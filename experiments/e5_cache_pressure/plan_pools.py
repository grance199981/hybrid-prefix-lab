"""Where to put the pools: run the block-level simulator (CPU only) on the exact
sessions and dispatch order the client will use, and report, per seed,

  peak_in_use      blocks in use at the peak with a pool too large to evict anything
  min_no_failure   smallest pool for which the simulator has no allocation failure
  min_no_loss      smallest pool for which every hit equals the hit with an unlimited pool (F*)

for the full-attention model (blocks of 64 tokens, 9 MiB) and the hybrid model
(blocks of 576 tokens, 18 MiB; unit 16 or 576; rule sets M0 and M1).

Closed mode has no fixed dispatch order (workers take sessions as they go); it
is approximated by waves: each group of C sessions sends turn k before turn k + 1.

Output: one JSON row per (seed, model variant), printed and optionally saved.

Usage (in an environment with transformers for the recorded workloads):
  python experiments/e5_cache_pressure/plan_pools.py --arch fa|hy --workload synth|smith|wildchat
      [--path P --tokenizer-dir D --thinking on|off] --seeds 1 2 3 --n-sessions N
      [--turns T --p0 P0 --step S] --max-len L --n-out O --mode closed|window
      --inflight C [--live N] [--json out.json]
"""
import argparse
import json

import lru_sim
from pressure_replay import build_schedule, session_queue
from pressure_sessions import fit, real_sessions, synth_sessions

BLOCK_MIB = {"fa": 9, "hy": 18}       # 24 GiB = 2,730 full-attention blocks = 1,365 hybrid blocks (vLLM start-up logs)
BLOCK_TOKENS = {"fa": lru_sim.FA_BLOCK, "hy": lru_sim.HY_BLOCK}


def closed_schedule(sessions, seed, inflight):
    queue, sched = session_queue(sessions, seed), []
    for g in range(0, len(queue), inflight):
        group = queue[g:g + inflight]
        for k in range(max(len(s["prompts"]) for s in group)):
            sched += [(s["sid"], k) for s in group if k < len(s["prompts"])]
    return sched


def min_pool(ok, lo, hi):
    """Smallest pool in [lo, hi] with ok(pool), assuming ok is monotone; None if ok(hi) is false."""
    if not ok(hi):
        return None
    if ok(lo):
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def plan(sessions, schedule, arch, unit, rules, n_out, overlap):
    events = lru_sim.events_from_schedule(sessions, schedule, n_out, overlap)
    B = BLOCK_TOKENS[arch]
    big = sum(lru_sim.cdiv(r["P"] + r["n_out"], B) + 5 for name, r in events if name == "start")
    _, s_big = lru_sim.simulate(events, big, arch, unit, rules)
    no_fail = min_pool(lambda p: lru_sim.simulate(events, p, arch, unit, rules)[1]["alloc_failures"] == 0, 1, big)
    no_loss = lru_sim.min_pool_without_loss(sessions, schedule, arch, unit, rules, n_out, 1, big, overlap)
    mib = BLOCK_MIB[arch]
    return {"n_requests": len(schedule), "peak_in_use_blocks": s_big["peak_in_use"],
            "min_no_failure_blocks": no_fail, "min_no_loss_blocks": no_loss,
            "peak_in_use_mib": s_big["peak_in_use"] * mib,
            "min_no_failure_mib": None if no_fail is None else no_fail * mib,
            "min_no_loss_mib": None if no_loss is None else no_loss * mib}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", required=True, choices=["fa", "hy"])
    ap.add_argument("--workload", required=True, choices=["synth", "smith", "wildchat"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--chat-template", default=None)
    ap.add_argument("--thinking", default="on", choices=["on", "off"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-sessions", type=int, required=True)
    ap.add_argument("--turns", type=int, default=None)
    ap.add_argument("--p0", type=int, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--n-out", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=["closed", "window"])
    ap.add_argument("--inflight", type=int, required=True)
    ap.add_argument("--live", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    live = a.live or a.inflight
    tok = None
    if a.workload != "synth":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    variants = [("fa", None, "M0")] if a.arch == "fa" else [("hy", u, r) for u in (16, 576) for r in ("M0", "M1")]
    out = []
    for seed in a.seeds:
        if a.workload == "synth":
            sessions = fit(synth_sessions(a.n_sessions, a.turns, a.p0, a.step, seed, list(range(1000, 9000))),
                           a.max_len, a.n_out)
        else:
            sessions = real_sessions(a.workload, a.path, a.tokenizer_dir, a.chat_template, seed, a.n_sessions,
                                     a.thinking, a.max_len, a.n_out, tok=tok)
        schedule = closed_schedule(sessions, seed, a.inflight) if a.mode == "closed" \
            else build_schedule(sessions, "window", live, seed)
        lens = [len(p) for s in sessions for p in s["prompts"]]
        for arch, unit, rules in variants:
            row = {"seed": seed, "arch": arch, "unit": unit, "rules": rules, "n_sessions": len(sessions),
                   "prompt_tokens_mean": round(sum(lens) / len(lens)), "prompt_tokens_max": max(lens)}
            row.update(plan(sessions, schedule, arch, unit, rules, a.n_out, a.inflight))
            out.append(row)
            print(json.dumps(row), flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
