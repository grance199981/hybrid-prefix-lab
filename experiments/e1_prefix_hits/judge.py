"""Verdicts of E1 from the replay records, with the thresholds fixed before the run.

Input: a directory of replay records named
<cfg>_<dataset>_think<on|off>_<R|O>_s<S>_seed<seed>.jsonl (written by replay.py
through scripts/run_replay.sh). Scored requests: k > 0, no error. Seeds are merged.

Output (stdout table, optional --json):
  cells             hit rate, recompute median / p90, TTFT p50 / p90 per cell
  H2                full attention hits floor(D / 16) * 16 (within 16 tokens, >= 95 %)
  H3                default hybrid config hits the previous request's 576 checkpoint (>= 90 %)
  H4                with --prefix-match-unit 16 the hit is floor((P_prev - 1) / 16) * 16 (>= 90 %)
  H5                periodic misses caused by the chat template at unit 16
  H6                own-output replay: does the hit reach into the previous reply
  H8, H9            report only (TTFT vs recompute rank correlation; misses at 576)
  kill_criterion_1  hit-rate gap < 10 points and recompute-median ratio < 1.5 in every
                    dataset x thinking cell, R and O replay, F0 against the better of C1, C4

Usage: python experiments/e1_prefix_hits/judge.py <dir with *.jsonl> [--json out.json]
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics

from hplab.cache_model import checkpoint_position
from hplab.stats import percentile

TAG = re.compile(r"^(?P<cfg>[A-Z]\d)_(?P<ds>[a-z]+)_think(?P<think>on|off)_(?P<mode>[RO])_s(?P<S>\d+)_seed(?P<seed>\d+)$")
DATASETS = ["wildchat", "smith", "rebench"]
THINKS = ["on", "off"]
HYBRID_BEST_CANDIDATES = ["C1", "C4"]
GAP_PP_LIMIT = 10.0
RATIO_LIMIT = 1.5


def load_dir(path):
    """{(cfg, ds, think, mode, S): [rows of all seeds]} plus per-cell error counts."""
    cells, errors = collections.defaultdict(list), collections.Counter()
    for f in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        m = TAG.match(os.path.splitext(os.path.basename(f))[0])
        if not m:
            continue
        key = (m["cfg"], m["ds"], m["think"], m["mode"], int(m["S"]))
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("sid") == "control":
                continue
            r["seed"] = int(m["seed"])
            if r.get("error"):
                errors[key] += 1
            cells[key].append(r)
    return cells, errors


def scored(rows):
    return [r for r in rows if r["k"] > 0 and not r.get("error") and r.get("prompt_tokens")]


def cell_stats(rows):
    rs = scored(rows)
    if not rs:
        return None
    rec = sorted(r["prompt_tokens"] - r["cached_tokens"] for r in rs)
    out = {"n": len(rs), "n_sessions": len({(r["seed"], r["sid"]) for r in rs}),
           "hit_rate": sum(r["cached_tokens"] for r in rs) / sum(r["prompt_tokens"] for r in rs),
           "recompute_median": statistics.median(rec), "recompute_p90": percentile(rec, 0.9),
           "n_P_mismatch": sum(1 for r in rs if r.get("P") != r["prompt_tokens"])}
    tt = sorted(r["ttft_ms"] for r in rs if r.get("ttft_ms") is not None)
    if tt:
        out["ttft_ms_p50"], out["ttft_ms_p90"] = statistics.median(tt), percentile(tt, 0.9)
    return out


def frac(rows, pred):
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else None


# ----------------------------------------------------------------- hypotheses
def h2(cells, cfg="F0"):
    """Full attention: hit within 16 of floor(D / 16) * 16 (R, one session at a time)."""
    out, pooled = {}, []
    for ds in DATASETS:
        for th in THINKS:
            rs = scored(cells.get((cfg, ds, th, "R", 1), []))
            pooled += rs
            out[f"{ds}/{th}"] = frac(rs, lambda r: abs(r["cached_tokens"] - r["fa16_R"]) <= 16)
    val = frac(pooled, lambda r: abs(r["cached_tokens"] - r["fa16_R"]) <= 16)
    return {"cfg": cfg, "per_cell": out, "pooled": val, "n": len(pooled),
            "holds": None if val is None else val >= 0.95}


def h3(cells, block=576):
    pooled = []
    for ds in DATASETS:
        for th in THINKS:
            pooled += scored(cells.get(("C0", ds, th, "R", 1), []))
    val = frac(pooled, lambda r: r["cached_tokens"] == checkpoint_position(r["P_prev"], block))
    val_fb = frac(pooled, lambda r: r["cached_tokens"] == r.get(f"hy{block}"))
    return {"pooled_equal_prev_checkpoint": val, "pooled_equal_with_fallback_rule": val_fb,
            "n": len(pooled), "holds": None if val is None else val >= 0.90}


def h4(cells, cfg):
    """unit 16: hit == floor((P_prev - 1) / 16) * 16 whenever that is <= D."""
    pooled = []
    for ds in DATASETS:
        for th in THINKS:
            pooled += [r for r in scored(cells.get((cfg, ds, th, "R", 1), []))
                       if checkpoint_position(r["P_prev"], 16) <= r["D_R"]]
    val = frac(pooled, lambda r: r["cached_tokens"] == checkpoint_position(r["P_prev"], 16))
    return {"cfg": cfg, "pooled": val, "n_condition_met": len(pooled),
            "holds": None if val is None else val >= 0.90}


def h5(cells, cfg, ds="wildchat"):
    """Periodic misses at unit 16: requests with (P_prev - 1) mod 16 <= g - 2
    cannot use the previous request's checkpoint."""
    out = {}
    for th, expect in (("on", 1 / 16), ("off", 3 / 16)):
        rs = [r for r in scored(cells.get((cfg, ds, th, "R", 1), [])) if r.get("g_obs") is not None]
        bad = [r for r in rs if r["residue16"] <= r["g_obs"] - 2]
        good = [r for r in rs if r["residue16"] > r["g_obs"] - 2]
        share = len(bad) / len(rs) if rs else None
        miss = lambda r: r["cached_tokens"] < checkpoint_position(r["P_prev"], 16)  # noqa: E731
        res = {"n": len(rs), "bad_share": share, "expected_share": expect,
               "share_within_5pp": None if share is None else abs(share - expect) <= 0.05,
               "bad_miss_frac": frac(bad, miss), "good_miss_frac": frac(good, miss),
               "bad_recompute_median": statistics.median(
                   r["prompt_tokens"] - r["cached_tokens"] for r in bad) if bad else None,
               "good_recompute_median": statistics.median(
                   r["prompt_tokens"] - r["cached_tokens"] for r in good) if good else None}
        # "clearly lower": at least 90 % of the bad group misses the previous
        # checkpoint and at most 10 % of the good group does
        res["bad_clearly_lower"] = (None if not bad or not good else
                                    res["bad_miss_frac"] >= 0.9 and res["good_miss_frac"] <= 0.1)
        res["holds"] = (None if res["bad_clearly_lower"] is None else
                        bool(res["bad_clearly_lower"] and res["share_within_5pp"]))
        out[th] = res
    return {"cfg": cfg, "dataset": ds, "per_thinking": out}


def h6(cells):
    """O replay: how far the hit reaches past the previous prompt."""
    out = {}
    for cfg in ("F0", "F1", "C1", "C4"):
        for ds in DATASETS:
            rs = []
            for th in THINKS:
                rs += scored(cells.get((cfg, ds, th, "O", 1), []))
            if rs:
                out[f"{cfg}/{ds}"] = {
                    "n": len(rs),
                    "frac_hit_past_prev_prompt": frac(rs, lambda r: r["cached_tokens"] > r["P_prev"]),
                    "frac_D_past_prev_prompt": frac(rs, lambda r: r["D_I"] > r["P_prev"])}
    hyb = [v["frac_hit_past_prev_prompt"] for k, v in out.items() if k.startswith(("C1", "C4"))]
    f1 = out.get("F1/rebench", {}).get("frac_hit_past_prev_prompt")
    return {"per_cell": out,
            "hybrid_never_past_prev_prompt": None if not hyb else all(x == 0 for x in hyb),
            "F1_rebench_reaches_into_reply": None if f1 is None else f1 > 0}


def h8(cells):
    """TTFT against recomputed tokens (report only): Spearman rank correlation."""
    def spearman(xs, ys):
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for t in range(i, j + 1):
                    rk[order[t]] = (i + j) / 2
                i = j + 1
            return rk
        rx, ry = ranks(xs), ranks(ys)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        sx = sum((a - mx) ** 2 for a in rx) ** 0.5
        sy = sum((b - my) ** 2 for b in ry) ** 0.5
        return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy) if sx and sy else None
    out = {}
    for (cfg, ds, th, mode, S), rows in sorted(cells.items()):
        if mode != "R" or S != 1:
            continue
        rs = [r for r in scored(rows) if r.get("ttft_ms") is not None]
        if len(rs) >= 10:
            out[f"{cfg}/{ds}/{th}"] = spearman([r["prompt_tokens"] - r["cached_tokens"] for r in rs],
                                               [r["ttft_ms"] for r in rs])
    return out


def h9(cells, block=576):
    rs = []
    for ds in DATASETS:
        for th in THINKS:
            for mode in ("R", "O"):     # report only
                rs += [r for r in scored(cells.get(("C0", ds, th, mode, 1), []))
                       if r.get("g_obs") is not None]
    bad = [r for r in rs if 0 <= (r["P_prev"] - 1) % block <= r["g_obs"] - 2]
    return {"n": len(rs), "n_bad_residue": len(bad),
            "bad_miss_frac": frac(bad, lambda r: r["cached_tokens"] < checkpoint_position(r["P_prev"], block))}


# ------------------------------------------------------------ kill criterion 1
def pick_best_hybrid(cells):
    meds = {}
    for cfg in HYBRID_BEST_CANDIDATES:
        rec = []
        for ds in DATASETS:
            for th in THINKS:
                rec += [r["prompt_tokens"] - r["cached_tokens"]
                        for r in scored(cells.get((cfg, ds, th, "R", 1), []))]
        if rec:
            meds[cfg] = statistics.median(rec)
    if not meds:
        return None, meds
    return min(HYBRID_BEST_CANDIDATES, key=lambda c: (meds.get(c, float("inf")),
                                                      HYBRID_BEST_CANDIDATES.index(c))), meds


def kill_criterion_1(cells, fa="F0"):
    best, meds = pick_best_hybrid(cells)
    grid, complete = [], best is not None
    for ds in DATASETS:
        for th in THINKS:
            row = {"dataset": ds, "thinking": th}
            f_r, h_r = (cell_stats(cells.get((c, ds, th, "R", 1), [])) for c in (fa, best))
            f_o, h_o = (cell_stats(cells.get((c, ds, th, "O", 1), [])) for c in (fa, best))
            if f_r and h_r:
                row["gap_pp_R"] = 100 * (f_r["hit_rate"] - h_r["hit_rate"])
                row["ratio_R"] = h_r["recompute_median"] / f_r["recompute_median"]
            if f_o and h_o:
                row["ratio_O"] = h_o["recompute_median"] / f_o["recompute_median"]
            complete = complete and all(k in row for k in ("gap_pp_R", "ratio_R", "ratio_O"))
            grid.append(row)
    reached = [r for r in grid if r.get("gap_pp_R", 0) >= GAP_PP_LIMIT
               or r.get("ratio_R", 0) >= RATIO_LIMIT or r.get("ratio_O", 0) >= RATIO_LIMIT]
    return {"full_attention": fa, "best_hybrid": best, "candidate_recompute_medians": meds,
            "grid": grid, "complete": complete, "cells_reaching_a_threshold": reached,
            "triggered": (not reached) if complete else None}


def judge(cells, errors):
    stats = {}
    for key, rows in sorted(cells.items()):
        s = cell_stats(rows)
        if s:
            s["n_errors"] = errors.get(key, 0)
            s["error_rate_over_1pct"] = s["n_errors"] / (len(rows) or 1) > 0.01
            stats["/".join(map(str, key))] = s
    return {
        "cells": stats,
        "H2_F0": h2(cells, "F0"), "H2_F1": h2(cells, "F1"), "H3_C0": h3(cells),
        "H4": {c: h4(cells, c) for c in ("C1", "C3", "C4")},
        "H5": {c: h5(cells, c) for c in ("C1", "C3", "C4")},
        "H6": h6(cells), "H8_spearman_ttft_vs_recompute": h8(cells), "H9_C0": h9(cells),
        "kill_criterion_1": kill_criterion_1(cells, "F0"),
        "F1_report_only": kill_criterion_1(cells, "F1"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cells, errors = load_dir(a.dir)
    res = judge(cells, errors)
    print(f"{'cell':34s} {'n':>6s} {'hit%':>7s} {'recomp med':>10s} {'p90':>7s} {'ttft p50':>9s} {'err':>4s}")
    for k, s in res["cells"].items():
        print(f"{k:34s} {s['n']:6d} {100 * s['hit_rate']:7.2f} {s['recompute_median']:10.1f} "
              f"{s['recompute_p90']:7d} {s.get('ttft_ms_p50', float('nan')):9.1f} {s['n_errors']:4d}"
              + ("  ERR>1%" if s["error_rate_over_1pct"] else "")
              + (f"  P_MISMATCH={s['n_P_mismatch']}" if s["n_P_mismatch"] else ""))
    for name in ("H2_F0", "H2_F1", "H3_C0", "H4", "H5", "H6", "H9_C0"):
        print("==", name, json.dumps(res[name], ensure_ascii=False))
    for name in ("kill_criterion_1", "F1_report_only"):
        kc = res[name]
        print("==", name, "full attention:", kc["full_attention"], "best hybrid:", kc["best_hybrid"],
              kc["candidate_recompute_medians"], "complete:", kc["complete"], "triggered:", kc["triggered"])
        for r in kc["grid"]:
            print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline=chr(10)) as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
