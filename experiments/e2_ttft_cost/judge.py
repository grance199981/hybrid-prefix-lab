"""Four-pass comparison of TTFT between two server configurations, with controls.

Question: does --prefix-match-unit 16 make the first token arrive later on a
hybrid model, on the same requests, in sequential R replay?

Four server blocks run in the order C0a, C1a, C1b, C0b (C0 = defaults,
C1 = --prefix-match-unit <unit>). Every request is sent once per block, so each
request has four TTFTs. Per request:

  main    = mean(C1a, C1b) - mean(C0a, C0b)      the effect under test
  null_C0 = C0b - C0a                            same config twice: must be ~0
  null_C1 = C1b - C1a                            same config twice: must be ~0
  inject  = (C0a + 30 ms) - C0b                  a known 30 ms shift: must be seen

The two nulls are the known-negative input and the injected shift the
known-positive input of the same statistic. If either fails in any cell the
script gives no verdict on the effect. The ABBA order cancels a steady drift of
the machine over the run.

Cells: first request of each session per dataset (thinking on and off pooled),
later requests per dataset x thinking; 9 cells, at least 20 pairs each.
Verdict: "confirmed" (every cell slower: median >= 15 ms and >= 80 % of pairs
slower, later requests also recompute less), "refuted" (more than half of the
cells at or below 5 ms), "mixed", "no conclusion (controls failed)" or
"incomplete" (a cell with fewer than 20 pairs).

Input: a directory of replay records <block>_<dataset>_think<on|off>_R_s1_seed<seed>.jsonl
(scripts/run_replay.sh with REPLAY_CONFIGS="C0a C1a C1b C0b").
Output: a table on stdout and, with --json, every number behind it.

Usage: python experiments/e2_ttft_cost/judge.py <dir with the request records> [--json out.json]
"""
import argparse
import json
import statistics as st

from hplab.records import BLOCKS, DATASETS, THINKS, load_dir, pairs
from hplab.stats import describe, line_fit

NULL_MS = 10.0        # |median| of a null difference must stay within this
EFFECT_MS = 15.0      # median paired slowdown needed to call a cell "slower"
SHARE = 0.80          # and this share of the pairs must be slower
NOT_SLOWER_MS = 5.0   # median at or below this: the cell is "not slower"
INJECT_MS = 30.0
MIN_PAIRS = 20


def recompute(r):
    return r["prompt_tokens"] - r["cached_tokens"]


def judge_cell(rows4, metric="ttft_ms"):
    """rows4: list of [C0a, C1a, C1b, C0b] rows of the same request."""
    v = [[r[metric] for r in rows] for rows in rows4]
    res = {
        "main": describe([(b + c) / 2 - (a + d) / 2 for a, b, c, d in v]),
        "null_C0": describe([d - a for a, b, c, d in v]),
        "null_C1": describe([c - b for a, b, c, d in v]),
        "inject": describe([a + INJECT_MS - d for a, b, c, d in v]),
        "recompute_diff": describe([(recompute(r[1]) + recompute(r[2])) / 2
                                    - (recompute(r[0]) + recompute(r[3])) / 2 for r in rows4]),
        "median_by_block": [round(st.median(x), 1) for x in zip(*v)] if v else None,
    }
    n = res["main"]["n"]
    res["enough_pairs"] = n >= MIN_PAIRS
    if not res["enough_pairs"]:
        res.update(null_ok=None, inject_seen=None, slower=None, not_slower=None)
        return res
    res["null_ok"] = (abs(res["null_C0"]["median"]) <= NULL_MS
                      and abs(res["null_C1"]["median"]) <= NULL_MS)
    res["inject_seen"] = (res["inject"]["median"] >= EFFECT_MS
                          and res["inject"]["share_positive"] >= SHARE)
    res["slower"] = (res["main"]["median"] >= EFFECT_MS
                     and res["main"]["share_positive"] >= SHARE)
    res["not_slower"] = res["main"]["median"] <= NOT_SLOWER_MS
    return res


def grid(cells, metric="ttft_ms"):
    out = {}
    for ds in DATASETS:       # first requests: thinking on and off pooled per dataset
        out[f"first/{ds}"] = judge_cell(pairs(cells, ds, THINKS, "first", metric), metric)
    for ds in DATASETS:
        for th in THINKS:
            out[f"scored/{ds}/{th}"] = judge_cell(pairs(cells, ds, (th,), "scored", metric), metric)
    return out


def verdict(g):
    cs = list(g.values())
    if not all(c["enough_pairs"] for c in cs):
        return {"verdict": "incomplete", "controls_pass": None,
                "cells_too_small": [k for k, c in g.items() if not c["enough_pairs"]]}
    failed = [k for k, c in g.items() if not (c["null_ok"] and c["inject_seen"])]
    if failed:
        return {"verdict": "no conclusion (controls failed)", "controls_pass": False,
                "cells_failing_controls": failed}
    first = [c for k, c in g.items() if k.startswith("first/")]
    scored = [c for k, c in g.items() if k.startswith("scored/")]
    v1 = all(c["slower"] for c in first)
    v2 = all(c["slower"] and c["recompute_diff"]["median"] < 0 for c in scored)
    n_not = sum(c["not_slower"] for c in cs)
    if v1 and v2:
        word = "confirmed"
    elif n_not > len(cs) / 2:
        word = "refuted"
    else:
        word = "mixed"
    return {"verdict": word, "controls_pass": True,
            "V1_fixed_overhead_on_first_requests": v1,
            "V2_slower_despite_less_recompute": v2,
            "cells_slower": sum(c["slower"] for c in cs),
            "cells_not_slower": n_not, "n_cells": len(cs)}


# ------------------------------------------------------------- report only
def fit(xs, ys):
    """TTFT against recomputed tokens; None with fewer than 3 points."""
    f = line_fit(xs, ys)
    if f is None:
        return None
    return {"intercept_ms": round(f["intercept"], 1), "ms_per_token": round(f["slope"], 4), "n": f["n"]}


def report(cells):
    out = {"per_block": {}, "main_by_prompt_length": {}, "ttft_missing": {}}
    for b in BLOCKS:
        for ds in DATASETS:
            rs = [r for th in THINKS for r in cells.get((b, ds, th), {}).values()]
            sc = [r for r in rs if r["k"] > 0]
            if not sc:
                continue
            tt = [r for r in rs if r.get("ttft_ms") is not None]
            out["per_block"][f"{b}/{ds}"] = {
                "n_scored": len(sc),
                "hit_rate": round(sum(r["cached_tokens"] for r in sc)
                                  / sum(r["prompt_tokens"] for r in sc), 4),
                "recompute_median": st.median(recompute(r) for r in sc),
                "ttft_fit": fit([recompute(r) for r in tt], [r["ttft_ms"] for r in tt]),
            }
            out["ttft_missing"][f"{b}/{ds}"] = len(rs) - len(tt)
    for ds in DATASETS:       # is the slowdown the same for short and long prompts?
        ps = pairs(cells, ds, THINKS, "scored", "ttft_ms")
        ps.sort(key=lambda rows: rows[0]["prompt_tokens"])
        third = len(ps) // 3
        if third < MIN_PAIRS:
            continue
        for name, part in (("short", ps[:third]), ("middle", ps[third:2 * third]),
                           ("long", ps[2 * third:])):
            d = [(r[1]["ttft_ms"] + r[2]["ttft_ms"]) / 2 - (r[0]["ttft_ms"] + r[3]["ttft_ms"]) / 2
                 for r in part]
            out["main_by_prompt_length"][f"{ds}/{name}"] = {
                "prompt_tokens_median": st.median(r[0]["prompt_tokens"] for r in part),
                "main_median": round(st.median(d), 2), "n": len(d)}
    return out


def judge(cells):
    g = grid(cells, "ttft_ms")
    res = {"thresholds": {"NULL_MS": NULL_MS, "EFFECT_MS": EFFECT_MS, "SHARE": SHARE,
                          "NOT_SLOWER_MS": NOT_SLOWER_MS, "INJECT_MS": INJECT_MS,
                          "MIN_PAIRS": MIN_PAIRS},
           "grid_ttft": g, "verdict": verdict(g)}
    for metric in ("first_chunk_ms", "e2e_ms"):      # report only
        if any(r.get(metric) is not None for c in cells.values() for r in c.values()):
            res["report_" + metric] = {k: {"main": c["main"], "null_C0": c["null_C0"],
                                           "null_C1": c["null_C1"]}
                                       for k, c in grid(cells, metric).items()}
    res["report"] = report(cells)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = judge(load_dir(a.dir))
    print(f"{'cell':22} {'n':>5} {'main med':>9} {'slower%':>8} {'null C0':>8} {'null C1':>8} "
          f"{'inject':>7} {'recomp':>7}  blocks C0a/C1a/C1b/C0b")
    for k, c in res["grid_ttft"].items():
        if not c["main"]["n"]:
            print(f"{k:22} {0:5}")
            continue
        print(f"{k:22} {c['main']['n']:5} {c['main']['median']:9.1f} "
              f"{100 * c['main']['share_positive']:8.1f} {c['null_C0']['median']:8.1f} "
              f"{c['null_C1']['median']:8.1f} {c['inject']['median']:7.1f} "
              f"{c['recompute_diff']['median']:7.0f}  {c['median_by_block']}")
    print("== verdict", json.dumps(res["verdict"], ensure_ascii=False))
    for k in ("report_first_chunk_ms", "report_e2e_ms"):
        if k in res:
            print("==", k, json.dumps({c: v["main"] for c, v in res[k].items()}))
    print("== report", json.dumps(res["report"], ensure_ascii=False))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
