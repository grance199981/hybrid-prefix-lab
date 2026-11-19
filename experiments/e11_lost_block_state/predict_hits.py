"""E11: predict the prefix hits of a hybrid model request by request from the prefill split rule.

The rule is the one of vLLM `2f70c32d` (`Scheduler._mamba_block_aligned_split`,
`get_mamba_prefill_checkpoint_position`, `is_mamba_prefill_checkpoint_valid`). Two builds:

  base  the merge base: the last chunk stops at the last block boundary and at the
        prompt checkpoint, and a Mamba state is cached where a chunk ends on a block
        boundary or on the checkpoint;
  pr    with PR #65186: when the checkpoint is valid for the last chunk, that chunk runs to
        the end of the prompt and only the checkpoint's state is exported.

A request's hit is the largest cached state position that lies inside its common prefix with
the prompt that produced the state. Each request's prefill starts at its measured hit (so
cross-session hits of the first turn are taken as measured), and only the states left by the
earlier turns of the same session are used. One request at a time, no eviction.

Usage:
  predict_hits.py r3 <R3 deployment dir> [--cfg C1a] [--unit 64] [--budget 2048] [--same-rule]
      compares the predicted hits with the measured hits of both builds (Q1-Q3);
  predict_hits.py sim <dir of replay records> [--unit 64] [--budget 2048]
      simulates both builds from scratch on recorded sessions and reports the difference (Q4).
`--same-rule` simulates the base with the PR's rule (the negative control of Q3).
"""
import argparse
import collections
import glob
import json
import os
import re

BLOCK = 576           # Mamba / attention block of Qwen3.5-4B on this commit
ALIGN = 64            # FLA chunk size: the PR's checkpoint alignment


def ckpt(num_tokens, unit):
    """get_mamba_prefill_checkpoint_position without EAGLE and without hash lookahead."""
    return max((num_tokens - 1) // unit * unit, 0)


def checkpoint_valid(start, end, c, unit, block=BLOCK, align=ALIGN):
    """is_mamba_prefill_checkpoint_valid."""
    col = -(-end // block) - 2
    return (start % unit == 0 and col >= start // block and start + unit <= c
            and start < c < end and (c - start) % align == 0)


def prefill(P, start, unit, budget, build, block=BLOCK):
    """Positions whose Mamba state is cached by one prefill of a P-token prompt from `start`."""
    partial = unit < block
    c = ckpt(P, unit if partial else block)
    cached, s = [], start
    while s < P:
        end = min(s + budget, P)
        last_cache = ckpt(P, block)
        internal = build == "pr" and end >= P and checkpoint_valid(s, end, c, unit if partial else block, block)
        if internal:
            last_cache = 0
        if end < P:
            aligned = end // block * block
            if aligned > s or block <= budget:
                end = aligned
        next_bb = (s // block + 1) * block
        tail = c if partial and not internal else 0
        stops = (next_bb if s % block and not internal else 0,
                 last_cache,
                 tail if last_cache < tail < P else 0)
        end = min((x for x in stops if s < x < end), default=end)
        if end <= s:                      # should not happen; guard against a stuck loop
            break
        if end < P and (end % block == 0 or end == c):
            cached.append(end)
        if internal:
            cached.append(c)
        s = end
    return cached


def predict_session(rows, unit, budget, build, use_measured_start=True):
    """rows: one session's requests sorted by k, each with P, D_R and (optionally) hit.
    Returns the predicted hits."""
    states = []                           # (position, turn that produced it)
    lcp_chain = []                        # lcp_chain[i] = common prefix of turn i and turn i - 1
    preds = []
    for i, r in enumerate(rows):
        P = r["P"]
        lcp_chain.append(r["D_R"] if i else 0)
        best = 0
        for x, j in states:
            lim = min(lcp_chain[j + 1:i + 1]) if j < i else P
            if x <= lim and x <= P - 1 and x > best:
                best = x
        preds.append(best)
        start = r["hit"] if use_measured_start and r.get("hit") is not None else best
        states += [(x, i) for x in prefill(P, start, unit, budget, build)]
    return preds


def load_records(paths):
    sessions = collections.defaultdict(list)
    for f in paths:
        tag = os.path.basename(f)[:-6]
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("sid") in (None, "control") or r.get("error") or "D_R" not in r:
                continue
            sessions[(tag, r["sid"])].append({"k": r["k"], "P": r["P"], "D_R": r["D_R"],
                                              "hit": r.get("cached_tokens")})
    for v in sessions.values():
        v.sort(key=lambda r: r["k"])
    return sessions


def r3(a):
    out = {}
    for build, arm in (("base", "main"), ("pr", "pr")):
        files = glob.glob(os.path.join(a.dir, arm, f"{a.cfg}_*.jsonl"))
        sess = load_records(files)
        rule = "pr" if (build == "pr" or a.same_rule) else "base"
        res = {}
        for key, rows in sess.items():
            pred = predict_session(rows, a.unit, a.budget, rule)
            for r, p in zip(rows, pred):
                res[(key[0], key[1], r["k"])] = (p, r["hit"])
        out[build] = res
    keys = sorted(set(out["base"]) & set(out["pr"]))
    scored = [k for k in keys if k[2] > 0]
    pred_lower = {k for k in scored if out["pr"][k][0] < out["base"][k][0]}
    meas_lower = {k for k in scored if out["pr"][k][1] < out["base"][k][1]}
    exact = {b: sum(1 for k in scored if out[b][k][0] == out[b][k][1]) for b in out}
    both_exact_on_meas = sum(1 for k in meas_lower if out["base"][k][0] == out["base"][k][1]
                             and out["pr"][k][0] == out["pr"][k][1])
    summary = {"cfg": a.cfg, "unit": a.unit, "budget": a.budget, "same_rule": a.same_rule,
               "requests_k_gt_0": len(scored),
               "measured_lower_with_pr": len(meas_lower), "predicted_lower_with_pr": len(pred_lower),
               "in_both": len(pred_lower & meas_lower), "predicted_only": len(pred_lower - meas_lower),
               "measured_only": len(meas_lower - pred_lower),
               "measured_lower_with_both_hits_exact": both_exact_on_meas,
               "exact_hit_predictions": exact,
               "mismatches_examples": {b: [[*k, *out[b][k]] for k in scored if out[b][k][0] != out[b][k][1]][:15]
                                       for b in out}}
    print(json.dumps(summary, indent=1))
    if a.json:
        json.dump(summary, open(a.json, "w", encoding="utf-8"), indent=1)


def sim(a):
    files = glob.glob(os.path.join(a.dir, "**", a.pattern), recursive=True)
    sess = load_records([f for f in files if not f.endswith(".metrics.jsonl")])
    per = collections.defaultdict(lambda: {"n": 0, "lower": 0, "higher": 0, "lost_tokens": [], "base_hit": 0})
    for key, rows in sess.items():
        base = predict_session(rows, a.unit, a.budget, "base", use_measured_start=False)
        pr = predict_session(rows, a.unit, a.budget, "pr", use_measured_start=False)
        group = re.sub(r"_seed\d+$", "", key[0].split("_R_")[0])
        for r, b, p in list(zip(rows, base, pr))[1:]:
            g = per[group]
            g["n"] += 1
            g["base_hit"] += b
            if p < b:
                g["lower"] += 1
                g["lost_tokens"].append(b - p)
            elif p > b:
                g["higher"] += 1
    out = {}
    for g, v in sorted(per.items()):
        lost = sorted(v["lost_tokens"])
        out[g] = {"requests": v["n"], "lower_with_pr": v["lower"], "share": round(v["lower"] / v["n"], 4) if v["n"] else None,
                  "higher_with_pr": v["higher"], "lost_median": lost[len(lost) // 2] if lost else 0,
                  "lost_mean": round(sum(lost) / len(lost), 1) if lost else 0,
                  "lost_per_request_all": round(sum(lost) / v["n"], 2) if v["n"] else None,
                  "lost_share_of_base_hits": round(sum(lost) / v["base_hit"], 5) if v["base_hit"] else None}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["r3", "sim"])
    ap.add_argument("dir")
    ap.add_argument("--cfg", default="C1a")
    ap.add_argument("--unit", type=int, default=64)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--same-rule", action="store_true")
    ap.add_argument("--pattern", default="*.jsonl", help="sim: which record files to read")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    (r3 if a.mode == "r3" else sim)(a)


if __name__ == "__main__":
    main()
