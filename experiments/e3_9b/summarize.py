"""Predictions W1-W4 of E3 from the judge / locate outputs of two models.

Inputs are the JSON files written by experiments/e2_ttft_cost/judge.py and
locate_overhead.py, one directory per model, each with judge.json and locate.json:
  --small  Qwen3.5-4B (experiments/e2_ttft_cost/results)
  --large  Qwen3.5-9B (experiments/e3_9b/results)

W1  cost of one extra prefill stop: cold first requests, group with one more
    step vs group with the same number of steps
W2  slope of TTFT against recomputed tokens in the default config (C0a, C0b)
W3  break-even = W1 cost / W2 slope: tokens that must be saved to pay for a stop
W4  net effect on later requests, cell by cell, large vs small model

Usage: python experiments/e3_9b/summarize.py --small experiments/e2_ttft_cost/results \
           --large experiments/e3_9b/results [--json out.json]
"""
import argparse
import json
import os

DATASETS = ("wildchat", "smith", "rebench")


def load(d):
    with open(os.path.join(d, "judge.json"), encoding="utf-8") as f:
        judge = json.load(f)
    with open(os.path.join(d, "locate.json"), encoding="utf-8") as f:
        locate = json.load(f)
    return judge, locate


def facts(judge, locate):
    m1 = locate["M1_first_requests_by_step_difference"]
    slopes = {ds: [judge["report"]["per_block"][f"{b}/{ds}"]["ttft_fit"]["ms_per_token"]
                   for b in ("C0a", "C0b")] for ds in DATASETS}
    step = m1["1"]["ttft_diff_median"]
    grid = judge["grid_ttft"]
    return {
        "verdict": judge["verdict"]["verdict"],
        "controls_pass": judge["verdict"]["controls_pass"],
        "null_abs_max_ms": max(abs(c[k]["median"]) for c in grid.values()
                               for k in ("null_C0", "null_C1")),
        "inject_min_max_ms": [min(c["inject"]["median"] for c in grid.values()),
                              max(c["inject"]["median"] for c in grid.values())],
        "W1_same_steps": m1.get("0"), "W1_one_more_step": m1["1"],
        "W2_slope_default_ms_per_token": slopes,
        "W3_break_even_tokens": {ds: [round(step / s) for s in slopes[ds]] for ds in DATASETS},
        "net_scored_ms": {k: {"median": c["main"]["median"],
                              "share_slower": c["main"]["share_positive"],
                              "saved_tokens": -c["recompute_diff"]["median"], "n": c["main"]["n"]}
                          for k, c in grid.items() if k.startswith("scored/")},
        "first_ms": {k: c["main"]["median"] for k, c in grid.items() if k.startswith("first/")},
        "step_groups_ms": locate["M3_by_step_difference"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    small, large = facts(*load(a.small)), facts(*load(a.large))
    w4 = {k: {"small": small["net_scored_ms"][k]["median"], "large": v["median"],
              "large_is_lower": v["median"] < small["net_scored_ms"][k]["median"]}
          for k, v in large["net_scored_ms"].items()}
    step_l = large["W1_one_more_step"]["ttft_diff_median"]
    step_s = small["W1_one_more_step"]["ttft_diff_median"]
    slope_l = [s for ds in DATASETS for s in large["W2_slope_default_ms_per_token"][ds]]
    slope_s = [s for ds in DATASETS for s in small["W2_slope_default_ms_per_token"][ds]]
    out = {
        "small_4B": small, "large_9B": large,
        "W1": {"holds": step_l >= 15 and abs(large["W1_same_steps"]["ttft_diff_median"]) <= 8,
               "one_more_step_ms": step_l, "same_steps_ms": large["W1_same_steps"]["ttft_diff_median"],
               "n_same_steps": large["W1_same_steps"]["n"], "small_model_ms": step_s},
        "W2": {"holds": min(slope_l) > max(slope_s), "large_min_max": [min(slope_l), max(slope_l)],
               "small_min_max": [min(slope_s), max(slope_s)]},
        "W3": {"large_min_max": [round(step_l / max(slope_l)), round(step_l / min(slope_l))],
               "small_min_max": [round(step_s / max(slope_s)), round(step_s / min(slope_s))]},
        "W4": {"holds": all(v["large_is_lower"] for v in w4.values()), "cells": w4},
    }
    out["W3"]["holds"] = out["W3"]["large_min_max"][1] < out["W3"]["small_min_max"][0]
    for k in ("W1", "W2", "W3", "W4"):
        print("==", k, json.dumps(out[k]))
    print("== 9B verdict", large["verdict"], "controls", large["controls_pass"],
          "null max", large["null_abs_max_ms"], "inject", large["inject_min_max_ms"])
    print("== step groups 9B", json.dumps(large["step_groups_ms"]))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
