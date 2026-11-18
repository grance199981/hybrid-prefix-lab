"""E10 R1: CPU-replay retention against the measured E5 S tier (C1), pool by pool.

Measured numbers: experiments/e5_cache_pressure/results/table_s.json and table_s_extra.json
(retention over k > 0, three seeds); the 8,640 MiB / output 256 cell was not measured in
E5, so it takes the P0 arm of E7 R2 (1.000, experiments/e7_upstream_prs/results/R2_pr66881/table_P0.json).
Usage: compare_cpu_e5.py <analysis_gap1.json> [--json out.json]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E5 = os.path.join(HERE, "..", "e5_cache_pressure", "results")
MIB_PER_BLOCK = 18
# cells E5's S tier did not measure: 8,640 MiB / 16 comes from E5's control stage K (seed 1 only,
# experiments/e5_cache_pressure/results/judge_k_and_s.json), 8,640 MiB / 256 from the P0 arm of E7 R2
R2_P0 = {(8640, 256): 1.0, (8640, 16): 1.0}


def measured():
    out = {}
    for name in ("table_s.json", "table_s_extra.json"):
        for r in json.load(open(os.path.join(E5, name), encoding="utf-8")):
            if r["cfg"] == "C1" and r["stage"] == "S":
                out[(r["kv_mib"], r["out"])] = r["retention"]
    for k, v in R2_P0.items():
        out.setdefault(k, v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    cells = json.load(open(a.analysis, encoding="utf-8"))["cells"]
    meas = measured()
    rows = []
    for cell, s in cells.items():
        blocks, out = cell.split("blocks_out")
        mib = int(blocks) * MIB_PER_BLOCK
        m = meas.get((mib, int(out)))
        rows.append({"mib": mib, "blocks": int(blocks), "out": int(out), "cpu": s["retention"], "measured": m,
                     "diff": None if m is None else round(s["retention"] - m, 4)})
    rows.sort(key=lambda r: (r["mib"], r["out"]))
    for r in rows:
        print(f"{r['mib']:5d} MiB ({r['blocks']:3d} blocks) out {r['out']:3d}: cpu {r['cpu']:.4f}  measured "
              f"{'-' if r['measured'] is None else format(r['measured'], '.4f')}  diff {r['diff']}")
    diffs = [abs(r["diff"]) for r in rows if r["diff"] is not None]
    verdict = {"cells_compared": len(diffs), "max_abs_diff": max(diffs), "R1_holds": max(diffs) <= 0.15}
    print(json.dumps(verdict))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"rows": rows, "verdict": verdict}, f, indent=1)


if __name__ == "__main__":
    main()
