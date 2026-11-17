"""Q2 and Q2b: one Nsight Systems trace (SQLite) plus the group log of the `groups`
or `long` pattern of probe_client.py -> per-group totals: how long, how much of it
the GPU was busy, kernel time by class, the heaviest kernels, and how many
scheduling steps the group took.

A group is one prefill request or one set of concurrent decode requests; the
client pauses between groups, so each group is one burst of kernels (split as in
analyze_q1.py). Totals over the whole burst are used, not per-step windows:
for prompts longer than one chunk the scheduler's step markers run ahead of the
GPU (the next chunk is scheduled while the previous one still executes), so a
marker-to-marker window is not one forward pass.

Usage: analyze_q2.py --sqlite trace.sqlite --requests h4_C0_nsys.jsonl --json out.json
"""
import argparse
import collections
import json

from hplab.nsys import CLASS_NAMES, STEP_MARK, bursts, classify, load_trace, union_ms
from hplab.stats import read_jsonl


def group_stats(burst, marks, memcpys):
    lo, hi = min(k[0] for k in burst), max(k[1] for k in burst)
    dur = collections.Counter()
    by_name_t, by_name_n = collections.Counter(), collections.Counter()
    for k in burst:
        d = (k[1] - k[0]) / 1e6
        dur[classify(k[2])] += d
        by_name_t[k[2]] += d
        by_name_n[k[2]] += 1
    total = sum(dur.values())
    mc = [m for m in memcpys if lo <= m[0] < hi]
    return {
        "burst_ms": (hi - lo) / 1e6, "busy_ms": union_ms([(k[0], k[1]) for k in burst]),
        "kernel_ms_sum": total, "n_kernels": len(burst),
        "n_graph_kernels": sum(1 for k in burst if k[3]),
        "graph_kernel_ms": sum((k[1] - k[0]) / 1e6 for k in burst if k[3]),
        "n_marks": sum(1 for m in marks if lo - int(200e6) <= m[1] <= hi),
        "class_ms": {c: round(dur.get(c, 0.0), 4) for c in CLASS_NAMES},
        "class_share": {c: round(dur.get(c, 0.0) / total, 4) if total else None for c in CLASS_NAMES},
        "memcpy_ms": sum(m[1] - m[0] for m in mc) / 1e6, "memcpy_bytes": sum(m[2] or 0 for m in mc),
        "top_kernels": [{"name": n[:100], "class": classify(n), "n": by_name_n[n],
                         "total_ms": round(t, 3), "avg_us": round(1000 * t / by_name_n[n], 2),
                         "share": round(t / total, 4)} for n, t in by_name_t.most_common(8)],
    }


def analyse(sqlite_path, requests_path, burst_gap_ms):
    t0, kernels, memcpys, nvtx, api = load_trace(sqlite_path)
    groups = [g for g in read_jsonl(requests_path) if g.get("kind") == "measure"]
    marks = [(s, e) for s, e, t in nvtx if t == STEP_MARK]
    bs = bursts(kernels, int(burst_gap_ms * 1e6))
    meta = {"n_kernels": len(kernels), "n_step_marks": len(marks), "n_bursts": len(bs),
            "n_groups": len(groups), "bursts_match_groups": len(bs) == len(groups)}
    out = []
    if meta["bursts_match_groups"]:
        for g, b in zip(groups, bs):
            row = {k: g[k] for k in ("phase", "P", "B", "rep", "wall_ms", "completion_tokens", "n_errors")}
            row.update(group_stats(b, marks, memcpys))
            row["start_minus_send_ms"] = (t0 + min(k[0] for k in b) - g["t_send_ns"]) / 1e6
            out.append(row)
    return {"meta": meta, "groups": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--burst-gap-ms", type=float, default=400.0)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    res = analyse(a.sqlite, a.requests, a.burst_gap_ms)
    with open(a.json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    m = res["meta"]
    print(f"kernels {m['n_kernels']} step marks {m['n_step_marks']} bursts {m['n_bursts']} "
          f"groups {m['n_groups']} matched {m['bursts_match_groups']}", flush=True)


if __name__ == "__main__":
    main()
