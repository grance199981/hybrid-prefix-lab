"""Q3: one Nsight Systems trace (SQLite) of a tensor-parallel server plus the group
log of the `groups` pattern of probe_client.py -> per-group totals for each worker
process (one per GPU): how long the GPU was busy, kernel time by class, and how
much of it was spent in communication kernels (the all-reduce between the GPUs).

What "communication" means here: the time of kernels whose name contains "nccl"
(or vLLM's own all-reduce kernel). An NCCL kernel also waits for its peer inside
the kernel, so this is an upper bound of the transfer itself.

A single-GPU trace gives one process and no NCCL kernels, so the same script
analyses the tensor-parallel-size 1 baseline.

Usage: analyze_q3.py --sqlite trace.sqlite --requests fa_T2n_nsys.jsonl --json out.json
"""
import argparse
import collections
import json
import re

from hplab import nsys
from hplab.nsys import STEP_MARK, bursts, load_trace, union_ms
from hplab.stats import read_jsonl

NCCL = re.compile(r"nccl|cross_device_reduce|all_?reduce", re.I)   # NCCL, or vLLM's own all-reduce kernel
CLASS_NAMES = ["nccl"] + nsys.CLASS_NAMES


def classify(name):
    return "nccl" if NCCL.search(name) else nsys.classify(name)


def process_stats(kernels):
    dur, by_name_t, by_name_n = collections.Counter(), collections.Counter(), collections.Counter()
    for k in kernels:
        d = (k[1] - k[0]) / 1e6
        dur[classify(k[2])] += d
        by_name_t[k[2]] += d
        by_name_n[k[2]] += 1
    total = sum(dur.values())
    return {
        "busy_ms": union_ms([(k[0], k[1]) for k in kernels]),
        "busy_without_nccl_ms": union_ms([(k[0], k[1]) for k in kernels if not NCCL.search(k[2])]),
        "kernel_ms_sum": total, "n_kernels": len(kernels),
        "n_nccl_kernels": sum(1 for k in kernels if NCCL.search(k[2])),
        "nccl_ms": dur.get("nccl", 0.0), "nccl_share": dur.get("nccl", 0.0) / total if total else None,
        "class_ms": {c: round(dur.get(c, 0.0), 4) for c in CLASS_NAMES},
        "top_kernels": [{"name": n[:100], "class": classify(n), "n": by_name_n[n],
                         "total_ms": round(t, 3), "avg_us": round(1000 * t / by_name_n[n], 2)}
                        for n, t in by_name_t.most_common(6)],
    }


def group_stats(burst, marks, memcpys):
    lo, hi = min(k[0] for k in burst), max(k[1] for k in burst)
    by_pid = collections.defaultdict(list)
    for k in burst:
        by_pid[k[4]].append(k)
    procs = [process_stats(ks) for _, ks in sorted(by_pid.items())]
    mc = collections.Counter()
    for m in memcpys:
        if lo <= m[0] < hi:
            mc[str(m[3])] += 1
    nccl = [p["nccl_share"] for p in procs if p["nccl_share"] is not None]
    return {
        "burst_ms": (hi - lo) / 1e6, "n_processes": len(procs),
        "busy_ms_max": max(p["busy_ms"] for p in procs),
        "busy_without_nccl_ms_max": max(p["busy_without_nccl_ms"] for p in procs),
        "nccl_share_mean": sum(nccl) / len(nccl) if nccl else None,
        "n_nccl_kernels_max": max(p["n_nccl_kernels"] for p in procs),
        "n_marks": sum(1 for m in marks if lo - int(200e6) <= m[1] <= hi),
        "memcpy_count_by_kind": dict(mc), "processes": procs,
    }


def analyse(sqlite_path, requests_path, burst_gap_ms):
    t0, kernels, memcpys, nvtx, api = load_trace(sqlite_path)
    groups = [g for g in read_jsonl(requests_path) if g.get("kind") == "measure"]
    marks = [(s, e) for s, e, t in nvtx if t == STEP_MARK]
    bs = bursts(kernels, int(burst_gap_ms * 1e6))
    names = collections.Counter(k[2] for k in kernels if NCCL.search(k[2]))
    meta = {"n_kernels": len(kernels), "n_step_marks": len(marks), "n_bursts": len(bs),
            "n_groups": len(groups), "bursts_match_groups": len(bs) == len(groups),
            "n_processes": len({k[4] for k in kernels}),
            "nccl_kernel_names": [[n[:120], c] for n, c in names.most_common(6)]}
    out = []
    if meta["bursts_match_groups"]:
        for g, b in zip(groups, bs):
            row = {k: g[k] for k in ("phase", "P", "B", "rep", "wall_ms", "completion_tokens", "n_errors")}
            row.update(group_stats(b, marks, memcpys))
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
    print(f"kernels {m['n_kernels']} processes {m['n_processes']} step marks {m['n_step_marks']} "
          f"bursts {m['n_bursts']} groups {m['n_groups']} matched {m['bursts_match_groups']} "
          f"nccl kernel names {len(m['nccl_kernel_names'])}", flush=True)


if __name__ == "__main__":
    main()
