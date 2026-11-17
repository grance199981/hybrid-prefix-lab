"""Pair the same request across two builds of vLLM (PR minus base) in a four-pass replay.

Each directory holds the four passes (C0a, C1a, C1b, C0b) of one build, as written by
scripts/run_replay.sh (C0 = default configuration, C1 = --prefix-match-unit). A request
enters a cell only if it is present, with a TTFT, in all four passes of both builds.

Per cell (first requests per dataset; later requests per dataset and thinking mode) this reports
  default      median over requests of mean(C0a, C0b) of (PR - base)
  unit16       the same for C1a, C1b (the unit is whatever C1 ran with)
  cost_main    unit minus default on the base build, on the same requests
  cost_pr      the same on the PR build
  cost_diff    paired difference of the two costs (PR - base)
and how many requests have a different number of cached tokens on the two builds.
`verdict` evaluates the pre-registered predictions F2 and F3 of the E8 follow-up
(F2: the default configuration of the PR is at least 5 ms faster in the four SWE cells).

Output: one line per cell on stdout and the verdict; with --json {"grid", "verdict"}.

Usage: python experiments/e8_pr61952/pair_envs.py <base_dir> <pr_dir> [--json out.json]
"""
import argparse
import json
import statistics as st

from hplab.records import BLOCKS, DATASETS, THINKS, load_dir

F2_CELLS = ("scored/smith/on", "scored/smith/off", "scored/rebench/on", "scored/rebench/off")
F2_MS = -5.0
SCORED = tuple(f"scored/{ds}/{th}" for ds in DATASETS for th in THINKS)


def describe(xs):
    if not xs:
        return {"n": 0, "median": None, "share_negative": None}
    return {"n": len(xs), "median": round(st.median(xs), 2),
            "share_negative": round(sum(x < 0 for x in xs) / len(xs), 3)}


def cell_rows(main, pr, ds, ths, subset):
    """[(base rows of the 4 passes, PR rows of the 4 passes)] of the same request."""
    out = []
    for th in ths:
        per = [c.get((b, ds, th), {}) for c in (main, pr) for b in BLOCKS]
        for key in sorted(set(per[0]).intersection(*per[1:])):
            if (key[2] == 0) != (subset == "first"):
                continue
            rows = [p[key] for p in per]
            if all(r.get("ttft_ms") is not None for r in rows):
                out.append((rows[:4], rows[4:]))
    return out


def judge_cell(rows):
    t = [([r["ttft_ms"] for r in m], [r["ttft_ms"] for r in p]) for m, p in rows]
    cost = lambda v: (v[1] + v[2]) / 2 - (v[0] + v[3]) / 2      # noqa: E731
    return {
        "default": describe([((p[0] - m[0]) + (p[3] - m[3])) / 2 for m, p in t]),
        "unit16": describe([((p[1] - m[1]) + (p[2] - m[2])) / 2 for m, p in t]),
        "default_by_pass": [describe([p[i] - m[i] for m, p in t])["median"] for i in (0, 3)],
        "unit16_by_pass": [describe([p[i] - m[i] for m, p in t])["median"] for i in (1, 2)],
        "cost_main": describe([cost(m) for m, p in t]),
        "cost_pr": describe([cost(p) for m, p in t]),
        "cost_diff": describe([cost(p) - cost(m) for m, p in t]),
        "cached_tokens_differ": sum(a["cached_tokens"] != b["cached_tokens"]
                                    for m, p in rows for a, b in zip(m, p)),
    }


def grid(main, pr):
    out = {}
    for ds in DATASETS:
        out[f"first/{ds}"] = judge_cell(cell_rows(main, pr, ds, THINKS, "first"))
    for ds in DATASETS:
        for th in THINKS:
            out[f"scored/{ds}/{th}"] = judge_cell(cell_rows(main, pr, ds, (th,), "scored"))
    return out


def verdict(g):
    f2 = {c: g[c]["default"]["median"] for c in F2_CELLS}
    smaller = {c: g[c]["cost_pr"]["median"] < g[c]["cost_main"]["median"] for c in SCORED}
    positive = {c: g[c]["cost_pr"]["median"] > 0 for c in SCORED}
    return {
        "F2_default_medians": f2,
        "F2_holds": all(v is not None and v <= F2_MS for v in f2.values()),
        "F3a_cost_smaller_cells": sum(smaller.values()),
        "F3a_holds": all(smaller.values()),
        "F3b_cost_positive_cells": sum(positive.values()),
        "F3b_holds": sum(positive.values()) >= 4,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("main_dir", help="four passes of the base build")
    ap.add_argument("pr_dir", help="four passes of the PR build")
    ap.add_argument("--json")
    a = ap.parse_args()
    g = grid(load_dir(a.main_dir), load_dir(a.pr_dir))
    res = {"grid": g, "verdict": verdict(g)}
    for c, v in g.items():
        print(f"{c:22s} n={v['default']['n']:4d}  default {v['default']['median']:+7.1f}  unit16 {v['unit16']['median']:+7.1f}  "
              f"cost main {v['cost_main']['median']:+6.1f}  PR {v['cost_pr']['median']:+6.1f}  diff {v['cost_diff']['median']:+6.1f}  "
              f"cached differ {v['cached_tokens_differ']}")
    print(json.dumps(res["verdict"], indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
