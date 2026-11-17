"""Verdicts of the cache-pressure experiment from the records of pressure_replay.py.

Files: <tag>.jsonl, <tag>.summary.json, <tag>.metrics.jsonl with
  tag = <cfg>_<stage>_<workload>_kv<MiB>_out<n>_seed<seed>
  cfg   F0 (Qwen3-4B), C0 (Qwen3.5-4B), C1 (Qwen3.5-4B, --prefix-match-unit 16)
  stage S, I, L, W, or K for the control cells
A cell is (cfg, stage, workload, kv MiB, output length); its seeds are merged.

retention = sum(cached_tokens) / sum(pred_inf) over the steady window: how much
of what an unlimited cache would reuse the server still reused.

Controls come first and are printed first; if one fails there is no verdict:
  ample    largest stage-K pool of F0 and of C1: >= 99.5 % of the requests hit
           exactly pred_inf (known good: nothing may be lost);
  minimum  smallest stage-K pool: retention < 0.25 (known bad: the measurement
           must be able to see a loss);
  reconciliation  per run with no preemption: the per-request sums equal the
           /metrics deltas. Runs with preemptions are skipped (a preempted
           request is looked up again, so the counters run ahead).

Kill criterion "no concurrency-specific problem", all of:
  (S) at the two tightest stage-S pools (F0 and C1 paired by rank) and both
      output lengths: retention(C1) >= retention(F0) - 0.10 and
      preemptions(C1) <= preemptions(F0);
  (I) at the two tightest stage-I pools: retention(F0) - retention(C1) < 0.10
      and recompute median C1 / F0 < 1.5;
  (same bytes) wherever F0 and C1 ran with the same pool size:
      retention(C1) >= retention(F0).
The verdict is None while a needed cell is missing. The thresholds (10
percentage points, a ratio of 1.5) are the ones of the project's E1 kill
criterion.

Output: a per-cell table, the controls, the conditions and the verdict on
stdout; with --json the same as one JSON object. --sim-blocks also replays each
run through lru_sim.py and reports how often the measured hit equals the
simulated one.

Usage: python experiments/e5_cache_pressure/pressure_judge.py <dir>
           [--tokens-per-mib F0=7.1 C1=...] [--sim-blocks F0@4096=453 ...] [--json out.json]
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics

import lru_sim
from hplab.stats import percentile as pct
from hplab.stats import read_jsonl

TAG = re.compile(r"^(?P<cfg>F0|C0|C1)_(?P<stage>[SILWK])_(?P<wl>[a-z]+)_kv(?P<kv>\d+)"
                 r"_out(?P<out>\d+)_seed(?P<seed>\d+)$")
THRESHOLDS = {"retention_gap": 0.10, "recompute_ratio": 1.5, "n_tightest": 2, "n_out_lens": 2,
              "ample_share": 0.995, "minimum_retention": 0.25}
CLASSES = ["full", "older", "shared_only", "zero", "above"]
PREEMPT, QUERIES, HITS = "vllm:num_preemptions", "vllm:prefix_cache_queries", "vllm:prefix_cache_hits"
EVICT, USAGE = "vllm:kv_block_idle_before_evict_seconds_count", "vllm:kv_cache_usage_perc"


def load_dir(path):
    """{(cfg, stage, workload, kv, out): [run]}; run = {"seed", "rows", "summary", "metrics"}."""
    cells = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        stem = os.path.splitext(f)[0]
        m = TAG.match(os.path.basename(stem))
        if not m or not os.path.exists(stem + ".summary.json"):
            continue        # *.metrics.jsonl, and runs that stopped before the summary
        with open(stem + ".summary.json", encoding="utf-8") as g:
            summary = json.load(g)
        metrics = read_jsonl(stem + ".metrics.jsonl") if os.path.exists(stem + ".metrics.jsonl") else []
        cells[(m["cfg"], m["stage"], m["wl"], int(m["kv"]), int(m["out"]))].append(
            {"seed": int(m["seed"]), "rows": read_jsonl(f), "summary": summary, "metrics": metrics})
    return dict(cells)


def session_rows(rows):
    return [r for r in rows if r.get("sid") != "control"]


def steady_window(rows, pool_tokens):
    """Requests of one run that count: k > 0, no error, and
    window mode: dispatched after the recomputed tokens (P - cached) of the
      earlier requests add up to pool_tokens (the pool has been filled once)
      and before the last session is started (the live count is still full);
    closed mode: not in the first session of any worker. When no worker runs a
      second session (stage S: one session per worker) there is no warm-up
      session to cut, and every request with k > 0 counts, which is the rule
      of pressure_table.py."""
    rows = sorted((r for r in session_rows(rows) if not r.get("error")), key=lambda r: r["seq"])
    if any(r.get("worker") is not None for r in rows):
        first, sids = {}, collections.defaultdict(set)
        for r in rows:
            first.setdefault(r["worker"], r["sid"])
            sids[r["worker"]].add(r["sid"])
        if all(len(v) == 1 for v in sids.values()):
            return [r for r in rows if r["k"] > 0]
        return [r for r in rows if r["k"] > 0 and r["sid"] not in set(first.values())]
    last_start = max((r["seq"] for r in rows if r["k"] == 0), default=-1)
    out, recomputed = [], 0
    for r in rows:
        if recomputed >= pool_tokens and r["seq"] < last_start and r["k"] > 0:
            out.append(r)
        recomputed += r["P"] - r["cached_tokens"]
    return out


def classify(row, cross_prefix):
    """What the request reused, against pred_inf and the prefix its session
    shares with other sessions (cross_prefix)."""
    c, p = row["cached_tokens"], row["pred_inf"]
    if c > p:
        return "above"
    if c == p:
        return "full"
    if c == 0:
        return "zero"
    return "shared_only" if c <= cross_prefix else "older"


def retention(rows):
    den = sum(r["pred_inf"] for r in rows)
    return sum(r["cached_tokens"] for r in rows) / den if den else None


def _sum_delta(runs, name):
    vals = [run["summary"].get("delta", {}).get(name) for run in runs]
    return None if any(v is None for v in vals) else sum(vals)


def cell_stats(runs, pool_tokens=0, control=False):
    """Stats of one cell. Control cells (stage K) use every request with k > 0."""
    rows, classes = [], collections.Counter()
    for run in runs:
        if control:
            kept = [r for r in session_rows(run["rows"]) if r["k"] > 0 and not r.get("error")]
        else:
            kept = steady_window(run["rows"], pool_tokens)
        cross = run["summary"].get("cross_prefix", {})
        classes.update(classify(r, cross.get(r["sid"], 0)) for r in kept)
        rows += kept
    if not rows:
        return None
    rec = sorted(r["P"] - r["cached_tokens"] for r in rows)
    usage = [m[USAGE] for run in runs for m in run["metrics"] if USAGE in m]
    ttft = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
    return {"n": len(rows), "n_runs": len(runs), "retention": retention(rows),
            "recompute_median": statistics.median(rec), "recompute_p90": pct(rec, 0.9),
            "class_share": {c: classes[c] / len(rows) for c in CLASSES},
            "ttft_ms_p50": statistics.median(ttft) if ttft else None,
            "preemptions": _sum_delta(runs, PREEMPT), "evictions": _sum_delta(runs, EVICT),
            "kv_usage_max": max(usage) if usage else None,
            "n_errors": sum(1 for run in runs for r in session_rows(run["rows"]) if r.get("error"))}


# ------------------------------------------------------------------- controls
def controls(cells, stats, th=THRESHOLDS):
    """cells: load_dir output; stats: {key: cell_stats}. Returns the checks and
    the list of problems (empty = all controls passed)."""
    out = {"ample": [], "minimum": [], "reconciliation": [], "reconciliation_skipped": [], "problems": []}
    for cfg in sorted({k[0] for k in stats if k[1] == "K"} | {"F0", "C1"}):
        ks = sorted(k for k in stats if k[0] == cfg and k[1] == "K" and stats[k])
        kvs = sorted({k[3] for k in ks})
        if len(kvs) < 2:
            out["problems"].append(f"{cfg}: needs an ample and a minimum control cell (stage K), "
                                   f"found pools {kvs}")
            continue
        for k in ks:
            s = stats[k]
            if k[3] == kvs[-1]:
                ok = s["class_share"]["full"] >= th["ample_share"]
                out["ample"].append({"cell": k, "share_full": s["class_share"]["full"], "holds": ok})
                if not ok:
                    out["problems"].append(f"ample control {k}: only {s['class_share']['full']:.4f} "
                                           f"of the requests hit pred_inf")
            if k[3] == kvs[0]:
                ok = s["retention"] is not None and s["retention"] < th["minimum_retention"]
                out["minimum"].append({"cell": k, "retention": s["retention"], "holds": ok})
                if not ok:
                    out["problems"].append(f"minimum control {k}: retention {s['retention']} is not "
                                           f"below {th['minimum_retention']}")
    for key, runs in sorted(cells.items()):
        for run in runs:
            sm, name = run["summary"], (*key, run["seed"])
            d = sm.get("delta", {})
            if d.get(PREEMPT) != 0:
                out["reconciliation_skipped"].append({"run": name, "preemptions": d.get(PREEMPT)})
                continue
            ok = sm["sum_cached_tokens"] == d.get(HITS) and sm["sum_prompt_tokens"] == d.get(QUERIES)
            out["reconciliation"].append({"run": name, "holds": ok})
            if not ok:
                out["problems"].append(
                    f"reconciliation {name}: requests sum to {sm['sum_cached_tokens']} cached of "
                    f"{sm['sum_prompt_tokens']}, /metrics says {d.get(HITS)} of {d.get(QUERIES)}")
    return out


# ------------------------------------------------------------- kill criterion
def _paired(stats, stage, n_tightest):
    """[(workload, rank, out, F0 key or None, C1 key or None)] for the n tightest
    pools of a stage; F0 and C1 pools are paired by rank."""
    out = []
    for wl in sorted({k[2] for k in stats if k[1] == stage and k[0] in ("F0", "C1")}):
        pools = {c: sorted({k[3] for k in stats if k[:3] == (c, stage, wl)}) for c in ("F0", "C1")}
        outs = sorted({k[4] for k in stats if k[1:3] == (stage, wl) and k[0] in ("F0", "C1")})
        for rank in range(n_tightest):
            for o in outs:
                keys = [(c, stage, wl, pools[c][rank], o) if rank < len(pools[c]) else None
                        for c in ("F0", "C1")]
                out.append((wl, rank, o, *[k if k in stats and stats[k] else None for k in keys]))
    return out


def kill_criterion(stats, thresholds=THRESHOLDS):
    """stats: {(cfg, stage, workload, kv, out): cell_stats}. Every condition comes
    back with its numbers and `holds` (None = a cell or a number is missing)."""
    th, res, missing = thresholds, {"S": [], "I": [], "same_bytes": []}, []
    for stage in ("S", "I"):
        pairs = _paired(stats, stage, th["n_tightest"])
        if not pairs:
            missing.append(f"stage {stage}: no cells")
        if stage == "S" and pairs and len({p[2] for p in pairs}) < th["n_out_lens"]:
            missing.append(f"stage S: fewer than {th['n_out_lens']} output lengths")
        for wl, rank, o, fk, ck in pairs:
            row = {"stage": stage, "workload": wl, "rank": rank, "out": o,
                   "F0_kv": fk and fk[3], "C1_kv": ck and ck[3], "holds": None}
            if fk is None or ck is None:
                missing.append(f"stage {stage} {wl}: pool rank {rank}, out {o}: a cell is missing")
                res[stage].append(row)
                continue
            f, c = stats[fk], stats[ck]
            row.update({"retention_F0": f["retention"], "retention_C1": c["retention"]})
            if stage == "S":
                row.update({"preemptions_F0": f["preemptions"], "preemptions_C1": c["preemptions"]})
                vals = (f["retention"], c["retention"], f["preemptions"], c["preemptions"])
                if None not in vals:
                    row["holds"] = (c["retention"] >= f["retention"] - th["retention_gap"]
                                    and c["preemptions"] <= f["preemptions"])
            else:
                fm, cm = f["recompute_median"], c["recompute_median"]
                ratio = cm / fm if fm else (1.0 if cm == 0 else float("inf"))
                row.update({"recompute_median_F0": fm, "recompute_median_C1": cm, "ratio": ratio})
                if None not in (f["retention"], c["retention"]):
                    row["holds"] = (f["retention"] - c["retention"] < th["retention_gap"]
                                    and ratio < th["recompute_ratio"])
            if row["holds"] is None:
                missing.append(f"stage {stage} {wl}: pool rank {rank}, out {o}: a number is missing")
            res[stage].append(row)
    for fk in sorted(k for k in stats if k[0] == "F0" and k[1] != "K" and stats[k]):
        c = stats.get(("C1",) + fk[1:])
        if c:
            f = stats[fk]
            ok = None if None in (f["retention"], c["retention"]) else c["retention"] >= f["retention"]
            res["same_bytes"].append({"stage": fk[1], "workload": fk[2], "kv": fk[3], "out": fk[4],
                                      "retention_F0": f["retention"], "retention_C1": c["retention"],
                                      "holds": ok})
    if not res["same_bytes"]:
        missing.append("same bytes: no pool size that both F0 and C1 ran with")
    conds = res["S"] + res["I"] + res["same_bytes"]
    complete = not missing and all(r["holds"] is not None for r in conds)
    res.update({"missing": missing, "complete": complete,
                "no_concurrency_specific_problem": all(r["holds"] for r in conds) if complete else None})
    return res


def judge(cells, thresholds=THRESHOLDS, tokens_per_mib=None):
    """cells: load_dir output. tokens_per_mib: {cfg: tokens the pool holds per MiB},
    for the warm-up cut of the steady window (0 = no cut)."""
    tpm = tokens_per_mib or {}
    stats = {k: cell_stats(runs, pool_tokens=k[3] * tpm.get(k[0], 0), control=k[1] == "K")
             for k, runs in cells.items()}
    ctl = controls(cells, stats, thresholds)
    kc = kill_criterion(stats, thresholds)
    if ctl["problems"]:
        kc["verdict_withheld"] = kc["no_concurrency_specific_problem"]
        kc["no_concurrency_specific_problem"] = None
        kc["reason"] = "control failed: " + "; ".join(ctl["problems"])
    return {"cells": stats, "controls": ctl, "kill_criterion": kc}


def compare_sim(rows, sim_rows):
    """Share of requests whose measured cached_tokens equals the simulator's hit.
    sim_rows: {"M0": rows of lru_sim.simulate, "M1": ...}."""
    out = {}
    for rules, srows in sim_rows.items():
        hit = {(s["sid"], s["k"]): s["hit"] for s in srows}
        rs = [r for r in session_rows(rows) if not r.get("error") and (r["sid"], r["k"]) in hit]
        out[rules] = (sum(1 for r in rs if r["cached_tokens"] == hit[(r["sid"], r["k"])]) / len(rs)
                      if rs else None)
        out["n"] = len(rs)
    return out


def sim_check(cells, pool_blocks):
    """pool_blocks: {(cfg, kv): blocks}. Replays each run's timing through lru_sim
    under both rule sets (the rules only matter for the hybrid model)."""
    out = {}
    for key, runs in sorted(cells.items()):
        blocks = pool_blocks.get((key[0], key[3]))
        if blocks is None:
            continue
        arch, unit = ("fa", None) if key[0] == "F0" else ("hy", 16 if key[0] == "C1" else 576)
        for run in runs:
            events, _ = lru_sim.events_from_trace(run["rows"])
            sims = {m: lru_sim.simulate(events, blocks, arch, unit, m)[0] for m in ("M0", "M1")}
            out["/".join(map(str, (*key, run["seed"])))] = compare_sim(run["rows"], sims)
    return out


def _pairs_arg(items, cast):
    return {k: cast(v) for k, v in (it.split("=", 1) for it in items)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--tokens-per-mib", nargs="*", default=[], metavar="CFG=TOKENS")
    ap.add_argument("--sim-blocks", nargs="*", default=[], metavar="CFG@KVMIB=BLOCKS")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    cells = load_dir(a.dir)
    res = judge(cells, tokens_per_mib=_pairs_arg(a.tokens_per_mib, float))
    fmt = lambda v, spec: "-" if v is None else format(v, spec)      # noqa: E731
    print(f"{'cell':30s} {'n':>6s} {'retent':>7s} {'rec med':>8s} {'rec p90':>8s} {'full':>6s} "
          f"{'older':>6s} {'zero':>6s} {'preempt':>7s} {'evict':>7s} {'kv max':>6s} {'err':>4s}")
    for k, s in sorted(res["cells"].items()):
        name = "/".join(map(str, k))
        if not s:
            print(f"{name:30s} no request in the steady window")
            continue
        cs = s["class_share"]
        print(f"{name:30s} {s['n']:6d} {fmt(s['retention'], '7.3f')} {s['recompute_median']:8.1f} "
              f"{s['recompute_p90']:8d} {cs['full']:6.3f} {cs['older']:6.3f} {cs['zero']:6.3f} "
              f"{fmt(s['preemptions'], '7.0f')} {fmt(s['evictions'], '7.0f')} "
              f"{fmt(s['kv_usage_max'], '6.3f')} {s['n_errors']:4d}")
    ctl = res["controls"]
    print("== controls")
    for name in ("ample", "minimum"):
        for c in ctl[name]:
            print("   ", name, json.dumps(c))
    print(f"    reconciliation: {sum(1 for c in ctl['reconciliation'] if c['holds'])} of "
          f"{len(ctl['reconciliation'])} runs agree, {len(ctl['reconciliation_skipped'])} skipped "
          f"(preemptions)")
    for p in ctl["problems"]:
        print("    PROBLEM:", p)
    kc = res["kill_criterion"]
    print("== kill criterion")
    for name in ("S", "I", "same_bytes"):
        for row in kc[name]:
            print("   ", name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    for m in kc["missing"]:
        print("    missing:", m)
    print("== no_concurrency_specific_problem:", kc["no_concurrency_specific_problem"],
          ("(" + kc["reason"] + ")") if kc.get("reason") else "")
    if a.sim_blocks:
        blocks = {(k.split("@")[0], int(k.split("@")[1])): v
                  for k, v in _pairs_arg(a.sim_blocks, int).items()}
        res["sim"] = sim_check(cells, blocks)
        print("== measured hit equals the simulator's (share of requests)")
        for name, c in res["sim"].items():
            print("   ", name, json.dumps(c))
    if a.json:
        res["cells"] = {"/".join(map(str, k)): v for k, v in res["cells"].items()}
        with open(a.json, "w", encoding="utf-8", newline=chr(10)) as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
