"""Comparisons on the E1 replay records that were chosen after the results were seen
(not part of the verdicts in judge.py).

1. Full attention at a 64-token block: share of scored R-replay requests whose
   cached_tokens equals floor(min(D_R, P - 1) / 64) * 64 (judge.py's H2 uses 16;
   vLLM 1479deb0 uses 64-token blocks by default, vllm/config/cache.py:185).
2. TTFT between hybrid configs, paired on the same request (dataset, thinking,
   seed, session, k) in sequential R replay: first requests of a session and
   scored requests, plus the per-config line fit of TTFT against recomputed tokens.
3. How far C4's (and F0's) hits reach past the previous prompt in O replay.
4. --e0b/--e0: the accounting redone with block 576 against the original block-528
   records (identical shared fields, then hy576 vs hy528).

Usage: python experiments/e1_prefix_hits/posthoc.py <dir with the replay records> \
           [--e0b <dir> --e0 <dir>] [--json out.json]
"""
import argparse
import collections
import glob
import json
import os
import statistics as st

from hplab.stats import line_fit, percentile

DATASETS = ("wildchat", "smith", "rebench")


def load(d):
    cells = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        cfg, ds, th, mode, s, seed = os.path.basename(f)[:-6].split("_")
        with open(f, encoding="utf-8") as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
        cells[(cfg, ds, th[5:], mode, int(s[1:]))].extend(
            dict(r, seed=int(seed[4:])) for r in rows if not r.get("error"))
    return cells


def med(xs):
    return round(st.median(xs), 1) if xs else None


def fit(xs, ys):
    f = line_fit(xs, ys)
    return round(f["intercept"], 1), round(f["slope"], 4)


def fa64(cells):
    out = {}
    for cfg in ("F0", "F1"):
        for ds in DATASETS:
            rs = [r for th in ("on", "off") for r in cells[(cfg, ds, th, "R", 1)] if r["k"] > 0]
            eq64 = sum(r["cached_tokens"] == min(r["D_R"], r["P"] - 1) // 64 * 64 for r in rs)
            eq16 = sum(r["cached_tokens"] == min(r["D_R"], r["P"] - 1) // 16 * 16 for r in rs)
            out[f"{cfg}/{ds}"] = {"n": len(rs), "equal_floor64": round(eq64 / len(rs), 4),
                                  "equal_floor16": round(eq16 / len(rs), 4)}
    return out


def ttft_pairs(cells, base, other):
    out = {}
    for ds in DATASETS:
        for th in ("on", "off"):
            a = {(r["seed"], r["sid"], r["k"]): r for r in cells[(base, ds, th, "R", 1)]}
            b = {(r["seed"], r["sid"], r["k"]): r for r in cells[(other, ds, th, "R", 1)]}
            keys = sorted(k for k in set(a) & set(b)
                          if a[k]["ttft_ms"] is not None and b[k]["ttft_ms"] is not None)
            res = {}
            for name, sel in (("first", [k for k in keys if k[2] == 0]),
                              ("scored", [k for k in keys if k[2] > 0])):
                d_t = [b[k]["ttft_ms"] - a[k]["ttft_ms"] for k in sel]
                d_r = [(b[k]["prompt_tokens"] - b[k]["cached_tokens"])
                       - (a[k]["prompt_tokens"] - a[k]["cached_tokens"])
                       for k in sel]
                res[name] = {
                    "n": len(sel),
                    "ttft_median_base": med([a[k]["ttft_ms"] for k in sel]),
                    "ttft_median_other": med([b[k]["ttft_ms"] for k in sel]),
                    "paired_ttft_diff_median": med(d_t),
                    "share_other_slower": round(sum(x > 0 for x in d_t) / len(sel), 3),
                    "paired_recompute_diff_median": med(d_r),
                    "ttft_p90_base": round(percentile(sorted(a[k]["ttft_ms"] for k in sel), 0.9), 1),
                    "ttft_p90_other": round(percentile(sorted(b[k]["ttft_ms"] for k in sel), 0.9), 1),
                }
            out[f"{ds}/{th}"] = res
    return out


def ttft_fits(cells):
    out = {}
    for cfg in ("F0", "F1", "C0", "C1", "C2", "C3", "C4"):
        for ds in DATASETS:
            every = [r for th in ("on", "off") for r in cells[(cfg, ds, th, "R", 1)]]
            rs = [r for r in every if r["ttft_ms"] is not None]
            if rs:
                i, b = fit([r["prompt_tokens"] - r["cached_tokens"] for r in rs],
                           [r["ttft_ms"] for r in rs])
                out[f"{cfg}/{ds}"] = {"n": len(rs), "intercept_ms": i, "ms_per_token": b,
                                      "ttft_missing": len(every) - len(rs)}
    return out


def past_prev(cells, cfg):
    out = {}
    for ds in DATASETS:
        over, n = [], 0
        for th in ("on", "off"):
            prev = {}
            for r in cells[(cfg, ds, th, "O", 1)]:
                key = (r["seed"], r["sid"])
                if r["k"] > 0 and key in prev:
                    n += 1
                    if r["cached_tokens"] > prev[key]:
                        over.append(r["cached_tokens"] - prev[key])
                prev[key] = r["prompt_tokens"]
        out[ds] = {"n": n, "n_past": len(over), "tokens_past_median": med(over)}
    return out


def e0b(new_dir, old_dir):
    same = ("P", "D_R", "D_I", "fa16_R", "fa16_I", "hy16", "hy_exact_end_I")
    n = diff = 0
    table = {}
    for f in sorted(glob.glob(os.path.join(new_dir, "*.jsonl"))):
        name = os.path.basename(f)
        with open(f, encoding="utf-8") as fh:
            a = [json.loads(x) for x in fh if x.strip()]
        with open(os.path.join(old_dir, name), encoding="utf-8") as fh:
            b = [json.loads(x) for x in fh if x.strip()]
        if len(a) != len(b):
            diff += abs(len(a) - len(b)) + 1
            continue
        for x, y in zip(a, b):
            n += 1
            if (x["sid"], x["k"]) != (y["sid"], y["k"]) or any(x.get(k) != y.get(k) for k in same):
                diff += 1
        ds, model, th, _ = name[:-6].split("_")
        t = table.setdefault((ds, model, th[5:]), collections.Counter())
        for x, y in zip(a, b):
            if x["k"] > 0:
                t["P"] += x["P"]
                t["hy576"] += x["hy576"]
                t["hy528"] += y["hy528"]
    rec = collections.defaultdict(lambda: ([], []))
    for f in sorted(glob.glob(os.path.join(new_dir, "*.jsonl"))):
        name = os.path.basename(f)
        ds, model, th, _ = name[:-6].split("_")
        with open(f, encoding="utf-8") as fh:
            a = [json.loads(x) for x in fh if x.strip()]
        with open(os.path.join(old_dir, name), encoding="utf-8") as fh:
            b = [json.loads(x) for x in fh if x.strip()]
        for x, y in zip(a, b):
            if x["k"] > 0:
                rec[(ds, model, th[5:])][0].append(x["P"] - x["hy576"])
                rec[(ds, model, th[5:])][1].append(y["P"] - y["hy528"])
    out = {"n_rows": n, "n_rows_differing": diff, "cells": {}}
    for key, t in sorted(table.items()):
        out["cells"]["/".join(key)] = {
            "hit_hy576": round(100 * t["hy576"] / t["P"], 2),
            "hit_hy528": round(100 * t["hy528"] / t["P"], 2),
            "recompute_median_hy576": med(rec[key][0]),
            "recompute_median_hy528": med(rec[key][1]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory with the E1 request records")
    ap.add_argument("--e0b", default=None)
    ap.add_argument("--e0", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cells = load(a.dir)
    res = {
        "fa_block_64": fa64(cells),
        "ttft_C1_vs_C0": ttft_pairs(cells, "C0", "C1"),
        "ttft_C2_vs_C0": ttft_pairs(cells, "C0", "C2"),
        "ttft_C4_vs_C1": ttft_pairs(cells, "C1", "C4"),
        "ttft_fits": ttft_fits(cells),
        "C4_hits_past_prev_prompt": past_prev(cells, "C4"),
        "F0_hits_past_prev_prompt": past_prev(cells, "F0"),
    }
    if a.e0b and a.e0:
        res["e0b"] = e0b(a.e0b, a.e0)
    for k, v in res.items():
        print("==", k)
        if isinstance(v, dict):
            for kk, vv in v.items():
                print("  ", kk, json.dumps(vv, ensure_ascii=False))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
