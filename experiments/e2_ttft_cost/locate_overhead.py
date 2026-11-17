"""Does the number of prefill scheduler steps explain the TTFT cost of
--prefix-match-unit 16?  Reads the same four-pass records as judge.py.

Step counts follow vllm/v1/core/sched/scheduler.py:_mamba_block_aligned_split
at commit 1479deb0 for a backend that cannot save the prompt checkpoint inside
the final chunk (the Triton GDN prefill kernel). Only the stops that a request
must make are counted, not the max_num_batched_tokens chunking (which is the same
in both configs):

  default:   stops strictly inside (s, P): L                    if s < L
  unit 16:   stops strictly inside (s, P): next block boundary  if s % block != 0
                                           L
                                           T                    if L < T
  with s = cached tokens, P = prompt tokens,
       L = floor((P - 1) / block) * block,  T = floor((P - 1) / 16) * 16.
  steps = 1 + number of distinct stops.

Output:
  M1  cold first requests (no cache hit in either config), grouped by the step
      difference: TTFT difference (unit 16 minus default) per group
  M2  per dataset, least-squares fit of the TTFT difference on the step
      difference and the recompute difference
  M3  per dataset, TTFT difference minus the recompute term, grouped by the
      step difference
  step_counts  "default steps -> unit-16 steps" histogram per dataset

Usage: python experiments/e2_ttft_cost/locate_overhead.py <dir with the records> \
           [--block 576] [--json out.json]
"""
import argparse
import json
import statistics as st

import numpy as np

from hplab.records import DATASETS, THINKS, load_dir, pairs


def steps(s, P, unit16, block):
    L = (P - 1) // block * block
    stops = {L}
    if unit16:
        if s % block:
            stops.add((s // block + 1) * block)
        T = (P - 1) // 16 * 16
        if L < T:
            stops.add(T)
    return 1 + sum(1 for x in stops if s < x < P)


def features(rows, block):
    """rows = [C0a, C1a, C1b, C0b] of one request."""
    c0a, c1a, c1b, c0b = rows
    d_ttft = (c1a["ttft_ms"] + c1b["ttft_ms"]) / 2 - (c0a["ttft_ms"] + c0b["ttft_ms"]) / 2
    s0 = steps(c0a["cached_tokens"], c0a["prompt_tokens"], False, block)
    s1 = steps(c1a["cached_tokens"], c1a["prompt_tokens"], True, block)
    d_rec = c0a["cached_tokens"] - c1a["cached_tokens"]      # recompute(C1) - recompute(C0)
    return d_ttft, s1 - s0, d_rec, s0, s1


def med(xs):
    return round(st.median(xs), 2) if xs else None


def ols(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    resid = np.asarray(y) - X @ beta
    ss_tot = float(((np.asarray(y) - np.mean(y)) ** 2).sum())
    return [round(float(b), 4) for b in beta], round(1 - float((resid ** 2).sum()) / ss_tot, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    ap.add_argument("--block", type=int, default=576,
                    help="block size the server reports for the hybrid model")
    a = ap.parse_args()
    cells = load_dir(a.dir)
    out = {}

    # M1: first requests, same step count vs one more step
    first = [r for ds in DATASETS for r in pairs(cells, ds, THINKS, "first", "ttft_ms")]
    # Some first requests do hit the cache (sessions of one dataset can share a
    # system prompt); M1 is stated for cold requests, so those are set aside.
    cold = [r for r in first if r[0]["cached_tokens"] == 0 and r[1]["cached_tokens"] == 0]
    out["M1_first_requests_not_cold"] = {
        "n_first": len(first), "n_cold": len(cold),
        "warm_by_dataset": {ds: sum(1 for r in pairs(cells, ds, THINKS, "first", "ttft_ms")
                                    if r[0]["cached_tokens"] or r[1]["cached_tokens"])
                            for ds in DATASETS}}
    groups = {}
    for r in cold:
        d_ttft, d_steps, _, _, _ = features(r, a.block)
        groups.setdefault(d_steps, []).append((d_ttft, r[0]["prompt_tokens"]))
    out["M1_first_requests_by_step_difference"] = {
        str(k): {"n": len(v), "ttft_diff_median": med([x[0] for x in v]),
                 "share_slower": round(sum(x[0] > 0 for x in v) / len(v), 3),
                 "prompt_tokens_min_max": [min(x[1] for x in v), max(x[1] for x in v)]}
        for k, v in sorted(groups.items())}

    # M2 / M3: regression per dataset over all requests (first and later)
    out["M2_regression"], out["M3_by_step_difference"], out["step_counts"] = {}, {}, {}
    for ds in DATASETS:
        rows = pairs(cells, ds, THINKS, "first", "ttft_ms") + \
            pairs(cells, ds, THINKS, "scored", "ttft_ms")
        f = [features(r, a.block) for r in rows]
        y = [x[0] for x in f]
        beta, r2 = ols(y, [np.array([x[1] for x in f], float), np.array([x[2] for x in f], float)])
        out["M2_regression"][ds] = {"n": len(f), "intercept_ms": beta[0], "ms_per_extra_step": beta[1],
                                    "ms_per_recomputed_token": beta[2], "r2": r2}
        by = {}
        for d_ttft, d_steps, d_rec, s0, s1 in f:
            by.setdefault(d_steps, []).append(d_ttft - beta[2] * d_rec)   # recompute term removed
        out["M3_by_step_difference"][ds] = {
            str(k): {"n": len(v), "ttft_diff_minus_recompute_term_median": med(v)}
            for k, v in sorted(by.items())}
        cnt = {}
        for _, _, _, s0, s1 in f:
            cnt[f"{s0}->{s1}"] = cnt.get(f"{s0}->{s1}", 0) + 1
        out["step_counts"][ds] = dict(sorted(cnt.items()))

    for k, v in out.items():
        print("==", k)
        for kk, vv in v.items():
            print("  ", kk, json.dumps(vv))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
