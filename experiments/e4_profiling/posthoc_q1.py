"""Q1, whole-request totals instead of per-step windows (written after the Q1
verdicts were read, so not one of them). For prompts longer than one chunk the
scheduler's step markers run ahead of the GPU (the next chunk is scheduled while
the previous one is still executing), so a marker-to-marker window is not one
forward pass. Totals over the whole request do not depend on that. Figure 1 and
the Q1 numbers of make_report.py come from this script's output.

For every config and prompt length (median over the repeats):
  burst  = first kernel start to last kernel end of the request
  busy   = union of all kernel intervals in the burst (GPU doing something)
  idle   = burst - busy (GPU waiting for the CPU)
and the differences unit16 - default for each model.

Usage: posthoc_q1.py <dir with *_nsys.analysis.json and *.jsonl> [--json q1_posthoc.json]
"""
import argparse
import json
import os

from hplab.stats import median, read_jsonl

PAIRS = {"h4": ("h4_C0", "h4_C1"), "h9": ("h9_C0", "h9_C1")}
CFGS = ("h4_C0", "h4_C1", "h9_C0", "h9_C1", "fa_F0")


def med(xs):
    return median(xs, 2)


def table(d, cfg):
    path = os.path.join(d, f"{cfg}_nsys.analysis.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        a = json.load(f)
    by = {}
    for r in a["requests"]:
        by.setdefault(r["P"], []).append(r)
    out = {}
    for P, rs in sorted(by.items()):
        first = [r["steps"][0] for r in rs if r["steps"]]
        dec = [s for r in rs for s in r["steps"][-2:]]        # the last two windows are plain decode steps
        out[P] = {
            "n": len(rs), "ttft_ms": med([r["ttft_ms"] for r in rs]),
            "burst_ms": med([r["burst_ms"] for r in rs]),
            "busy_ms": med([r["burst_busy_ms"] for r in rs]),
            "idle_ms": med([r["burst_ms"] - r["burst_busy_ms"] for r in rs]),
            "busy_share": med([r["burst_busy_ms"] / r["burst_ms"] for r in rs]),
            "n_kernels": med([r["n_kernels"] for r in rs]),
            "n_marks": sorted({r["n_steps"] for r in rs}),
            "first_window": {k: med([s[k] for s in first]) for k in ("wall_ms", "busy_ms", "gaps_ms", "n_kernels", "n_graph")},
            "decode_window": {k: med([s[k] for s in dec]) for k in ("wall_ms", "busy_ms", "n_kernels", "n_graph")},
        }
    return out


def plain_ttft(d, cfg):
    path = os.path.join(d, f"{cfg}_plain.jsonl")
    by = {}
    if os.path.exists(path):
        for r in read_jsonl(path):
            if r.get("kind") == "measure" and not r.get("error"):
                by.setdefault(r["P"], []).append(r["ttft_ms"])
    return {P: med(v) for P, v in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = {"configs": {c: table(a.dir, c) for c in CFGS},
           "plain_ttft_ms": {c: plain_ttft(a.dir, c) for c in CFGS}, "pairs": {}}
    for fam, (base, unit) in PAIRS.items():
        x, y = res["configs"][base], res["configs"][unit]
        px, py = res["plain_ttft_ms"][base], res["plain_ttft_ms"][unit]
        res["pairs"][fam] = {P: {
            "ttft_plain_diff_ms": round(py[P] - px[P], 2),
            "ttft_nsys_diff_ms": round(y[P]["ttft_ms"] - x[P]["ttft_ms"], 2),
            "burst_diff_ms": round(y[P]["burst_ms"] - x[P]["burst_ms"], 2),
            "busy_diff_ms": round(y[P]["busy_ms"] - x[P]["busy_ms"], 2),
            "idle_diff_ms": round(y[P]["idle_ms"] - x[P]["idle_ms"], 2),
            "kernels_diff": y[P]["n_kernels"] - x[P]["n_kernels"],
            "marks": [x[P]["n_marks"], y[P]["n_marks"]]} for P in sorted(set(x) & set(y))}
    for c, t in res["configs"].items():
        for P, r in t.items():
            print(f"{c:6} P={P:5} plain ttft {res['plain_ttft_ms'][c].get(P)}  nsys ttft {r['ttft_ms']}  "
                  f"burst {r['burst_ms']}  busy {r['busy_ms']}  idle {r['idle_ms']}  busy share {r['busy_share']}  "
                  f"kernels {r['n_kernels']}  first window {r['first_window']}  decode {r['decode_window']}")
    for fam, t in res["pairs"].items():
        for P, r in t.items():
            print(f"pair {fam} P={P}: {json.dumps(r)}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
