"""Q2b summary and verdicts: decode step time against context length and batch
size, hybrid (h4_C0, Qwen3.5-4B) against full attention (fa_F0, Qwen3-4B), plus
the clock reading that checks whether sustained load lowers the GPU clock.

Input directory (scripts/run_probe.sh with probe_client.py --pattern long and analyze_q2.py):
  <cfg>_nsys.jsonl, <cfg>_nsys.analysis.json   groups under Nsight Systems
  <cfg>_plain.jsonl, <cfg>_plain_nvml.csv      groups without a profiler, NVML samples
  scaling.json                                  q2_scaling.py with clock samples

Verdicts (thresholds fixed before the run), GPU-busy time per decode step at batch 64:
  L1  hybrid / full attention falls with every longer context (577 and up)
  L2  that ratio is at most 0.90 at the longest context (2,305 tokens)
  L3  and at least 1.00 at 577 tokens
  L4  full attention rises by 14-38 ms from 577 to 2,305 tokens, the hybrid by at
      most half as much
  L5  at batch 1 both models change by at most 1.5 ms from 577 to 2,305 tokens
  K1  the SM clock at n >= 1024 tokens per pass is at most 0.90 x the clock at n <= 16

Usage: summarize_q2b.py <dir> [--json out.json]
"""
import argparse
import csv
import json
import os
import statistics as st

from hplab.nsys import CLASS_NAMES
from hplab.stats import median as med, read_jsonl, rnd

MODELS = {"h4_C0": "Qwen3.5-4B", "fa_F0": "Qwen3-4B"}
HYB, FULL = "h4_C0", "fa_F0"
SHORT_C, LONG_C, BIG_B = 577, 2305, 64
L2_MAX_RATIO = 0.90            # hybrid / full attention at the longest context, batch 64
L3_MIN_RATIO = 1.00            # the same ratio at the shortest long context
L4_FULL_RISE_MS = (14.0, 38.0)  # full attention: step time at LONG_C minus SHORT_C, batch 64
L4_HYBRID_SHARE = 0.50         # hybrid rise <= this share of the full-attention rise
L5_B1_MS = 1.5                 # batch 1: |step time at LONG_C - SHORT_C| for both models
K1_LOW_N, K1_HIGH_N, K1_DROP = 16, 1024, 0.90   # clock at n >= 1024 <= 0.90 * clock at n <= 16


def load_groups(path):
    if not os.path.exists(path):
        return []
    return [g for g in read_jsonl(path) if g.get("kind") == "measure"]


def load_nvml(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def nvml_mean(samples, t0, t1):
    rows = [s for s in samples if t0 + (t1 - t0) // 3 <= s["epoch_ns"] <= t1]     # skip the ramp-up third
    if len(rows) < 3:
        return None
    return {k: rnd(st.mean(r[k] for r in rows), 1) for k in rows[0] if k != "epoch_ns"} | {"n": len(rows)}


def trace_summary(d, cfg):
    path = os.path.join(d, f"{cfg}_nsys.analysis.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        a = json.load(f)
    by = {}
    for g in a["groups"]:
        by.setdefault((g["phase"], g["P"], g["B"]), []).append(g)
    out = {"meta": a["meta"], "decode": {}, "warm": {}, "prefill": {}}
    for (phase, P, B), gs in sorted(by.items()):
        row = {"n": len(gs), "burst_ms": rnd(med([g["burst_ms"] for g in gs])),
               "busy_ms": rnd(med([g["busy_ms"] for g in gs])),
               "class_share": {c: rnd(med([g["class_share"][c] for g in gs]), 4) for c in CLASS_NAMES}}
        if phase == "decode":
            steps = med([g["completion_tokens"] / g["B"] for g in gs])
            row.update(steps=steps, n_marks=med([g["n_marks"] for g in gs]),
                       busy_ms_per_step=rnd(med([g["busy_ms"] / (g["completion_tokens"] / g["B"]) for g in gs])),
                       busy_ms_per_step_reps=[rnd(g["busy_ms"] / (g["completion_tokens"] / g["B"])) for g in gs],
                       burst_ms_per_step=rnd(row["burst_ms"] / steps),
                       # report only: divided by the scheduling steps actually taken (a few more
                       # than `steps` when the B requests do not arrive in the same step)
                       busy_ms_per_mark=rnd(med([g["busy_ms"] / g["n_marks"] for g in gs if g["n_marks"]])),
                       class_ms_per_step={c: rnd(med([g["class_ms"][c] for g in gs]) / steps)
                                          for c in CLASS_NAMES})
        out[phase].setdefault(P, {})[B] = row
    return out


def plain_summary(d, cfg):
    groups = load_groups(os.path.join(d, f"{cfg}_plain.jsonl"))
    nv = load_nvml(os.path.join(d, f"{cfg}_plain_nvml.csv"))
    by = {}
    for g in groups:
        by.setdefault((g["phase"], g["P"], g["B"]), []).append(g)
    out = {"decode": {}, "warm": {}, "prefill": {}, "n_groups": len(groups),
           "n_bad": sum(1 for g in groups if g["n_errors"] or g["n_wrong_length"] or g.get("n_wrong_cached")),
           "nvml_samples": len(nv)}
    for (phase, P, B), gs in sorted(by.items()):
        nvm = [x for x in (nvml_mean(nv, g["t_send_ns"], g["t_end_ns"]) for g in gs) if x]
        row = {"n": len(gs), "wall_ms": rnd(med([g["wall_ms"] for g in gs])),
               "nvml": {k: rnd(med([x[k] for x in nvm]), 1) for k in nvm[0] if k != "n"} if nvm else None}
        if phase == "decode":
            steps = med([g["completion_tokens"] / g["B"] for g in gs])
            row.update(steps=steps, wall_ms_per_step=rnd(row["wall_ms"] / steps),
                       tokens_per_s=rnd(med([g["completion_tokens"] / g["wall_ms"] * 1e3 for g in gs]), 1))
        out[phase].setdefault(P, {})[B] = row
    return out


def clock_summary(d):
    path = os.path.join(d, "scaling.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        sc = json.load(f)
    sw = sc["sweeps"].get("bf16")
    if not sw:
        return None
    pts = [{"n": p["n"], "ms": p["ms_per_pass"], "tflops": p["tflops"], "gb_per_s": p["weights_gb_per_s"],
            "power_w": p["nvml"].get("power_w"), "clock_sm_mhz": p["nvml"].get("clock_sm_mhz"),
            "util_memory": p["nvml"].get("util_memory")} for p in sw["points"]]
    return {"points": pts, "plateau_gb_per_s": sw["plateau_gb_per_s"], "best_tflops": sw["best_tflops"],
            "knee_n": sw["knee_n"]}


def judge_decode(busy):
    """busy[(cfg, C, B)] = GPU-busy ms per decode step. Returns the L verdicts."""
    v = {}

    def ratio(C):
        h, f = busy.get((HYB, C, BIG_B)), busy.get((FULL, C, BIG_B))
        return None if h is None or f is None else h / f

    cs = sorted({C for (_, C, B) in busy if B == BIG_B and C >= SHORT_C})
    rs = [ratio(C) for C in cs]
    v["L1_ratio_falls_with_context"] = {
        "contexts": cs, "hybrid_over_full": [rnd(r) for r in rs],
        "holds": None if len(cs) < 2 or None in rs else all(a > b for a, b in zip(rs, rs[1:]))}
    r_long, r_short = ratio(LONG_C), ratio(SHORT_C)
    v["L2_hybrid_faster_at_long_context"] = {
        "ratio": rnd(r_long), "holds": None if r_long is None else r_long <= L2_MAX_RATIO}
    v["L3_hybrid_still_slower_at_577"] = {
        "ratio": rnd(r_short), "holds": None if r_short is None else r_short >= L3_MIN_RATIO}
    need = [busy.get((c, C, BIG_B)) for c in (FULL, HYB) for C in (SHORT_C, LONG_C)]
    if None in need:
        v["L4_rise_with_context"] = {"holds": None}
    else:
        d_full, d_hyb = need[1] - need[0], need[3] - need[2]
        v["L4_rise_with_context"] = {
            "full_rise_ms": rnd(d_full), "hybrid_rise_ms": rnd(d_hyb),
            "hybrid_share": rnd(d_hyb / d_full) if d_full else None,
            "holds": L4_FULL_RISE_MS[0] <= d_full <= L4_FULL_RISE_MS[1] and d_hyb <= L4_HYBRID_SHARE * d_full}
    need = [busy.get((c, C, 1)) for c in (FULL, HYB) for C in (SHORT_C, LONG_C)]
    if None in need:
        v["L5_batch_1_flat"] = {"holds": None}
    else:
        d = {"full_ms": rnd(need[1] - need[0]), "hybrid_ms": rnd(need[3] - need[2])}
        v["L5_batch_1_flat"] = {**d, "holds": all(abs(x) <= L5_B1_MS for x in d.values())}
    return v


def judge_clock(points):
    low = [p["clock_sm_mhz"] for p in points if p["n"] <= K1_LOW_N and p.get("clock_sm_mhz")]
    high = [p["clock_sm_mhz"] for p in points if p["n"] >= K1_HIGH_N and p.get("clock_sm_mhz")]
    if not low or not high:
        return {"holds": None}
    lo, hi = st.mean(low), st.mean(high)
    return {"clock_low_n_mhz": rnd(lo, 1), "clock_high_n_mhz": rnd(hi, 1), "high_over_low": rnd(hi / lo),
            "holds": hi <= K1_DROP * lo}


def judge(d):
    res = {"trace": {c: trace_summary(d, c) for c in MODELS},
           "plain": {c: plain_summary(d, c) for c in MODELS}, "clock": clock_summary(d)}
    busy = {}
    for c in MODELS:
        t = res["trace"][c]
        for C, per_b in (t["decode"] if t else {}).items():
            for B, row in per_b.items():
                busy[(c, C, B)] = row["busy_ms_per_step"]
    v = judge_decode(busy)
    v["K1_clock_drops_under_sustained_compute"] = judge_clock(res["clock"]["points"]) if res["clock"] else {"holds": None}
    res["verdicts"] = v
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = judge(a.dir)
    for c in MODELS:
        t, p = res["trace"][c], res["plain"][c]
        print(f"== {c} ({MODELS[c]}): plain groups {p['n_groups']}, bad {p['n_bad']}, nvml samples {p['nvml_samples']}"
              + (f"; trace {t['meta']}" if t else "; no trace"))
        for C, per_b in sorted((t["decode"] if t else {}).items()):
            for B, r in sorted(per_b.items()):
                pl = p["decode"].get(C, {}).get(B, {})
                print(f"  decode C={C:5d} B={B:3d}: busy/step {r['busy_ms_per_step']} ms {r['busy_ms_per_step_reps']} "
                      f"steps {r['steps']} class/step {r['class_ms_per_step']} | plain tok/s {pl.get('tokens_per_s')} "
                      f"nvml {pl.get('nvml')}")
        for P, per_b in sorted(p["prefill"].items()):
            print(f"  prefill P={P}: {per_b}")
    if res["clock"]:
        for q in res["clock"]["points"]:
            print("  scaling", q)
    for k, x in res["verdicts"].items():
        print("==", k, json.dumps(x))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
