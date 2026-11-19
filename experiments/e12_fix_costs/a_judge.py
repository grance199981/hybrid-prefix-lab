"""E12 A: judge the A-B-B-A run (README.md).

Input: the run directory with seg<i>_<W>/<arm>/e5 (run_e12a.sh). Per cell (workload, pool, output):
  - each segment's numbers from perf_table.summarize;
  - AK: the two segments of the same arm differ by at most 5% in median TTFT and in requests/s;
  - the arm comparison uses both segments of each arm pooled.
Usage: a_judge.py <run_dir> [--order "main pr61282 pr61282 main"] [--json out.json]
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perf_table as T  # noqa: E402

KEYS = ("ttft_p50", "ttft_p90", "tpot_p50", "req_per_s", "out_tok_per_s", "preemptions", "retention", "usage_max")


def rel(a, b):
    return None if a is None or b is None or b == 0 else round(a / b - 1, 4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--order", default="main pr61282 pr61282 main")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    order = a.order.split()
    per_seg = collections.defaultdict(dict)       # cell -> seg index -> row
    pooled = collections.defaultdict(lambda: collections.defaultdict(list))   # cell -> arm -> dirs
    for i, arm in enumerate(order, 1):
        for w in ("S", "U", "L"):
            d = os.path.join(a.run_dir, f"seg{i}_{w}", arm, "e5")
            if not os.path.isdir(d):
                continue
            for r in T.summarize(T.collect([d])):
                cell = (r["workload"], w, r["kv_mib"], r["out"])
                per_seg[cell][i] = r
                pooled[cell][arm].append(d)
    out = []
    for cell in sorted(per_seg):
        segs = per_seg[cell]
        arms = {}
        for arm, dirs in pooled[cell].items():
            rows = [x for x in T.summarize(T.collect(dirs)) if (x["kv_mib"], x["out"]) == cell[2:]]
            arms[arm] = rows[0] if rows else None
        idx = collections.defaultdict(list)
        for i, arm in enumerate(order, 1):
            if i in segs:
                idx[arm].append(i)
        ak = {}
        for arm, ii in idx.items():
            if len(ii) == 2:
                x, y = segs[ii[0]], segs[ii[1]]
                ak[arm] = {"ttft_p50": rel(y["ttft_p50"], x["ttft_p50"]), "req_per_s": rel(y["req_per_s"], x["req_per_s"])}
        ak_ok = all(v is not None and abs(v) <= 0.05 for d in ak.values() for v in d.values())
        # the first arm of --order is the baseline ("main" in the output), the second the fix
        m, p = arms.get(order[0]), arms.get(order[1])
        cmp_ = {k: {"main": m[k] if m else None, "pr61282": p[k] if p else None,
                    "pr_vs_main": rel(p[k], m[k]) if m and p and isinstance(m[k], (int, float)) and k != "preemptions" else None}
                for k in KEYS}
        out.append({"workload": cell[0], "stage": cell[1], "kv_mib": cell[2], "out": cell[3], "AK_ok": ak_ok,
                    "AK": ak, "compare": cmp_,
                    "segments": {i: {k: segs[i][k] for k in KEYS} for i in sorted(segs)}})
    for r in out:
        c = r["compare"]
        print(f"{r['stage']} kv{r['kv_mib']} out{r['out']}: AK_ok={r['AK_ok']} AK={r['AK']}")
        print("   " + "  ".join(f"{k}: {c[k]['main']} -> {c[k]['pr61282']} ({c[k]['pr_vs_main']})"
                               for k in ("ttft_p50", "tpot_p50", "req_per_s", "preemptions", "retention")))
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
