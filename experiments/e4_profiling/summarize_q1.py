"""Q1 summary and verdicts: where the extra prefill step of --prefix-match-unit 16
goes, from the plain request timings and the per-step trace breakdown.

Input: a directory with, per server config and mode,
  <cfg>_plain.jsonl, <cfg>_nsys.jsonl        client records (probe_client.py --pattern cold)
  <cfg>_nsys.analysis.json                   per-step breakdown (analyze_q1.py)
Configs: h4_C0 / h4_C1 (Qwen3.5-4B), h9_C0 / h9_C1 (Qwen3.5-9B),
         fa_F0 / fa_F16 (Qwen3-4B).  The second of each pair sets
         --prefix-match-unit 16.

Verdicts (thresholds fixed before the run):
  T1  observed prefill step counts equal the scheduler rule (hplab.cache_model)
  T2  without a profiler, a prompt length with one extra step is 25-45 ms slower
      under unit 16, and one without an extra step differs by at most 8 ms
  T3  the trace's extra prefill wall time is 0.5-1.5 x the 34.5 ms measured in E2
  T4  the full-attention pair has the same step counts and differs by at most 8 ms
  T5, T6  (report only) breakdown of the extra step, decode step time

Usage: summarize_q1.py <dir> [--json out.json]
"""
import argparse
import json
import os

from hplab.cache_model import MAX_CHUNK, count_steps
from hplab.nsys import CLASS_NAMES
from hplab.stats import median, read_jsonl

PAIRS = {"h4": ("h4_C0", "h4_C1"), "h9": ("h9_C0", "h9_C1"), "fa": ("fa_F0", "fa_F16")}
E2_STEP_MS = 34.5                      # E2, cold first requests with one extra stop
STEP_FIELDS = ("wall_ms", "pre_ms", "busy_ms", "busy_graph_ms", "busy_eager_ms", "gaps_ms",
               "post_ms", "n_kernels", "n_graph", "n_api_calls", "api_ms",
               "n_kernel_launch_calls", "n_graph_launch_calls")


def med(xs):
    return median(xs, 3)


def load_requests(path):
    if not os.path.exists(path):
        return []
    return [r for r in read_jsonl(path) if r.get("kind") == "measure" and not r.get("error")]


def ttft_table(d, cfg, mode):
    by = {}
    for r in load_requests(os.path.join(d, f"{cfg}_{mode}.jsonl")):
        by.setdefault(r["P"], []).append(r)
    return {P: {"n": len(rs), "ttft_ms": med([r["ttft_ms"] for r in rs]),
                "first_chunk_ms": med([r["first_chunk_ms"] for r in rs]),
                "e2e_ms": med([r["e2e_ms"] for r in rs]),
                "bad": sum(1 for r in rs if r["prompt_tokens"] != P or r["cached_tokens"])}
            for P, rs in sorted(by.items())}


def step_median(steps):
    out = {k: med([s.get(k) for s in steps]) for k in STEP_FIELDS}
    out["by_class_ms"] = {c: med([s.get("by_class", {}).get(c, 0.0) for s in steps])
                          for c in CLASS_NAMES}
    return out


def trace_table(d, cfg):
    path = os.path.join(d, f"{cfg}_nsys.analysis.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        a = json.load(f)
    by = {}
    for r in a["requests"]:
        by.setdefault(r["P"], []).append(r)
    out = {"meta": {k: a["meta"].get(k) for k in (
        "n_kernels", "n_step_marks", "n_bursts", "n_requests", "bursts_match_requests",
        "graph_kernel_share", "burst_start_minus_send_ms", "class_ms")}, "by_P": {}}
    for P, rs in sorted(by.items()):
        pre = [[s for s in r["steps"] if s["kind"] == "prefill"] for r in rs]
        out["by_P"][P] = {
            "n": len(rs),
            "n_prefill_steps": sorted({r["n_prefill_steps"] for r in rs}, key=lambda x: (x is None, x)),
            "prefill_wall_ms": med([sum(s["wall_ms"] for s in p) for p in pre]),
            "prefill_busy_ms": med([sum(s["busy_ms"] for s in p) for p in pre]),
            "prefill_gaps_ms": med([sum(s["gaps_ms"] for s in p) for p in pre]),
            "prefill_pre_post_ms": med([sum((s["pre_ms"] or 0) + (s["post_ms"] or 0) for s in p) for p in pre]),
            "last_prefill_step": step_median([p[-1] for p in pre if p]),
            "first_prefill_step": step_median([p[0] for p in pre if p]),
            "post_prefill_step": step_median([s for r in rs for s in r["steps"] if s["kind"] == "post_prefill"]),
            "decode_step": step_median([s for r in rs for s in r["steps"] if s["kind"] == "decode"]),
        }
    return out


def predicted_steps(cfg, P):
    """Prefill steps of a cold prompt of P tokens under the scheduler rule of vLLM 1479deb0
    with the Triton GDN backend. A model without Mamba layers is cut by the token budget only."""
    if cfg.startswith("fa"):
        return -(-P // MAX_CHUNK)
    return count_steps(0, P, unit=16 if cfg.endswith("C1") else None)


def judge(d):
    cfgs = [c for pair in PAIRS.values() for c in pair]
    res = {"ttft": {}, "trace": {}, "pairs": {}}
    for c in cfgs:
        res["ttft"][c] = {m: ttft_table(d, c, m) for m in ("plain", "nsys")}
        res["trace"][c] = trace_table(d, c)
    for fam, (base, unit) in PAIRS.items():
        rows = {}
        Ps = sorted(set(res["ttft"][base]["plain"]) & set(res["ttft"][unit]["plain"]))
        for P in Ps:
            row = {"predicted_steps": [predicted_steps(base, P), predicted_steps(unit, P)]}
            for m in ("plain", "nsys"):
                a, b = res["ttft"][base][m].get(P), res["ttft"][unit][m].get(P)
                row[f"ttft_diff_{m}_ms"] = None if not a or not b or a["ttft_ms"] is None \
                    or b["ttft_ms"] is None else round(b["ttft_ms"] - a["ttft_ms"], 2)
            ta, tb = res["trace"][base], res["trace"][unit]
            if ta and tb and P in ta["by_P"] and P in tb["by_P"]:
                x, y = ta["by_P"][P], tb["by_P"][P]
                row["observed_steps"] = [x["n_prefill_steps"], y["n_prefill_steps"]]
                for k in ("prefill_wall_ms", "prefill_busy_ms", "prefill_gaps_ms", "prefill_pre_post_ms"):
                    row[k.replace("prefill_", "prefill_diff_")] = None if x[k] is None or y[k] is None \
                        else round(y[k] - x[k], 3)
                row["last_step_unit16"] = y["last_prefill_step"]
            rows[P] = row
        res["pairs"][fam] = rows

    # ---- verdicts
    v = {}
    extra = [P for P in res["pairs"].get("h4", {}) if
             predicted_steps("h4_C1", P) > predicted_steps("h4_C0", P)]
    same = [P for P in res["pairs"].get("h4", {}) if P not in extra]
    h4 = res["pairs"].get("h4", {})

    def steps_ok(fam, rows):
        if not rows or any("observed_steps" not in r for r in rows.values()):
            return None
        return all(r["observed_steps"] == [[r["predicted_steps"][0]], [r["predicted_steps"][1]]]
                   for r in rows.values())

    v["T1_step_counts_match_the_scheduler_rules"] = {f: steps_ok(f, res["pairs"].get(f)) for f in PAIRS}
    d_plain = [h4[P]["ttft_diff_plain_ms"] for P in extra]
    d_same = [h4[P]["ttft_diff_plain_ms"] for P in same]
    v["T2_plain_ttft_cost_of_the_extra_step"] = {
        "with_extra_step": dict(zip(extra, d_plain)), "without": dict(zip(same, d_same)),
        "holds": None if not d_plain or None in d_plain or None in d_same else
        all(25 <= x <= 45 for x in d_plain) and all(abs(x) <= 8 for x in d_same)}
    d_wall = [h4[P].get("prefill_diff_wall_ms") for P in extra]
    v["T3_trace_agrees_with_request_timing"] = {
        "prefill_wall_diff_ms": dict(zip(extra, d_wall)), "reference_ms": E2_STEP_MS,
        "holds": None if not d_wall or None in d_wall else
        all(0.5 * E2_STEP_MS <= x <= 1.5 * E2_STEP_MS for x in d_wall)}
    over = {}
    for c in cfgs:
        for P, row in (res["ttft"][c]["plain"] or {}).items():
            n = (res["ttft"][c]["nsys"] or {}).get(P)
            if n and n["ttft_ms"] is not None and row["ttft_ms"] is not None:
                over[f"{c}/{P}"] = round(n["ttft_ms"] - row["ttft_ms"], 2)
    v["profiler_overhead_on_ttft_ms"] = {"min": min(over.values()) if over else None,
                                         "median": med(list(over.values())),
                                         "max": max(over.values()) if over else None, "cells": over}
    fa = res["pairs"].get("fa", {})
    d_fa = [r["ttft_diff_plain_ms"] for r in fa.values()]
    v["T4_full_attention_has_no_extra_step"] = {
        "ttft_diff_plain_ms": {P: r["ttft_diff_plain_ms"] for P, r in fa.items()},
        "holds": None if not d_fa or None in d_fa or steps_ok("fa", fa) is None else
        steps_ok("fa", fa) and all(abs(x) <= 8 for x in d_fa)}
    # the breakdown of the extra step: the last prefill step under unit 16
    comp = {}
    for fam in ("h4", "h9"):
        for P in extra:
            row = res["pairs"].get(fam, {}).get(P)
            if row and row.get("last_step_unit16"):
                s = row["last_step_unit16"]
                comp[f"{fam}/{P}"] = {k: s[k] for k in ("wall_ms", "busy_ms", "busy_graph_ms",
                                                         "busy_eager_ms", "gaps_ms", "pre_ms", "post_ms",
                                                         "n_kernels", "n_graph", "api_ms")}
                comp[f"{fam}/{P}"]["by_class_ms"] = s["by_class_ms"]
    v["T5_breakdown_of_the_extra_step"] = comp
    dec = {}
    for c in cfgs:
        t = res["trace"][c]
        if t:
            xs = [x["decode_step"]["busy_ms"] for x in t["by_P"].values() if x["decode_step"]["busy_ms"] is not None]
            ws = [x["decode_step"]["wall_ms"] for x in t["by_P"].values() if x["decode_step"]["wall_ms"] is not None]
            dec[c] = {"busy_ms": med(xs), "wall_ms": med(ws)}
    v["T6_decode_step_ms"] = dec
    res["verdicts"] = v
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = judge(a.dir)
    for c, modes in res["ttft"].items():
        for m, t in modes.items():
            if t:
                print(f"{c:7} {m:5} " + "  ".join(f"P={P}: {r['ttft_ms']} ms (n={r['n']}, bad={r['bad']})"
                                                  for P, r in t.items()))
    for fam, rows in res["pairs"].items():
        for P, r in rows.items():
            print(f"pair {fam} P={P}: " + json.dumps({k: v for k, v in r.items() if k != "last_step_unit16"}))
    for k, v in res["verdicts"].items():
        print("==", k, json.dumps(v))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
