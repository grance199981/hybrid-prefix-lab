"""Q1: one Nsight Systems trace (exported to SQLite) plus the request log of the
`cold` pattern of probe_client.py -> a per-request, per-scheduler-step breakdown.

How a request is found on the timeline: requests are sent one at a time with a
pause in between, so GPU kernels form one burst per request; bursts are split
where the GPU is idle for more than --burst-gap-ms. Burst i belongs to request
i. The profiler clock (session start + event offset) is used only as a check.

How a step is found: vLLM's scheduler emits the NVTX range
"schedule: update_after_schedule" once per scheduling step, right before the
GPU work of that step (the marker precedes the first kernel of the step by under
a millisecond). The GPU work of step i lies between the end of marker i and the
start of marker i + 1 (hplab.nsys.step_windows).

Step kinds, in order: the prefill steps, one short step right after the prefill
(a few dozen kernels, not a full forward pass), then one decode step per further
output token, so number of prefill steps = number of steps - completion_tokens.

For every step: wall time, time before the first kernel, GPU busy time (union
of kernel intervals), idle time between kernels, time after the last kernel,
kernel time by class, how much came from CUDA-graph nodes and how much was
launched one by one, and the CUDA API calls made on the CPU side.

Usage: analyze_q1.py --sqlite trace.sqlite --requests h4_C1_nsys.jsonl --json out.json
"""
import argparse
import collections
import json
import statistics as st

from hplab.nsys import (CLASS_NAMES, STEP_MARK, bursts, classify, label_steps, load_trace, step_stats,
                        step_windows, union_ms)
from hplab.stats import read_jsonl


def analyse(sqlite_path, requests_path, burst_gap_ms):
    t0, kernels, memcpys, nvtx, api = load_trace(sqlite_path)
    reqs = [r for r in read_jsonl(requests_path) if r.get("kind") == "measure" and not r.get("error")]
    marks = [(s, e) for s, e, t in nvtx if t == STEP_MARK]
    bs = bursts(kernels, int(burst_gap_ms * 1e6))

    name_time, name_n = collections.Counter(), collections.Counter()
    for k in kernels:
        name_time[k[2]] += k[1] - k[0]
        name_n[k[2]] += 1
    meta = {
        "session_start_utc_ns": t0, "n_kernels": len(kernels), "n_memcpy": len(memcpys),
        "n_api_calls": len(api), "n_step_marks": len(marks),
        "api_names": dict(collections.Counter(a[2].split("_v")[0] for a in api).most_common(10)),
        "nvtx_texts": dict(collections.Counter(t for _, _, t in nvtx).most_common(12)),
        "n_bursts": len(bs), "n_requests": len(reqs),
        "kernel_processes": len({k[4] for k in kernels}),
        "graph_kernel_share": round(sum(1 for k in kernels if k[3]) / len(kernels), 4) if kernels else None,
        "top_kernels": [{"name": n[:110], "class": classify(n), "n": name_n[n],
                         "total_ms": round(t / 1e6, 3)} for n, t in name_time.most_common(30)],
        "class_ms": {c: round(sum(t for n, t in name_time.items() if classify(n) == c) / 1e6, 3)
                     for c in CLASS_NAMES},
    }
    out_reqs = []
    matched = len(bs) == len(reqs)
    meta["bursts_match_requests"] = matched
    residual = []
    if matched:
        for r, b in zip(reqs, bs):
            lo = min(k[0] for k in b)
            hi = max(k[1] for k in b)
            residual.append((t0 + lo - r["t_send_ns"]) / 1e6)
            # the first step's marker comes shortly before the first kernel
            wins = step_windows(marks, lo - int(200e6), hi)
            steps = [step_stats(w, b, memcpys, api) for w in wins]
            n_pre, kinds = label_steps(len(steps), r.get("completion_tokens"))
            for s, kind in zip(steps, kinds):
                s["kind"] = kind
            out_reqs.append({
                "P": r["P"], "rep": r["rep"], "ttft_ms": r["ttft_ms"],
                "first_chunk_ms": r["first_chunk_ms"], "e2e_ms": r["e2e_ms"],
                "burst_ms": (hi - lo) / 1e6, "burst_busy_ms": union_ms([(k[0], k[1]) for k in b]),
                "n_kernels": len(b), "n_steps": len(steps), "n_prefill_steps": n_pre,
                "steps": steps})
        meta["burst_start_minus_send_ms"] = {
            "median": round(st.median(residual), 2), "min": round(min(residual), 2),
            "max": round(max(residual), 2)}
    return {"meta": meta, "requests": out_reqs}


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
          f"requests {m['n_requests']} matched {m['bursts_match_requests']}", flush=True)


if __name__ == "__main__":
    main()
