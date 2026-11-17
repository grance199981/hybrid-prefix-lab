"""Q2 summary and verdicts: is decode limited by memory bandwidth and long prefill
by arithmetic, for a hybrid (h4_C0, Qwen3.5-4B) and a full-attention model
(fa_F0, Qwen3-4B), with prompts of 16 tokens in decode.

Input directory:
  scaling.json                     q2_scaling.py: the measured roofline
  <cfg>_plain.jsonl                probe_client.py --pattern groups, no profiler
  <cfg>_plain_nvml.csv             NVML samples during that run (nvml_sampler.py)
  <cfg>_nsys.jsonl, <cfg>_nsys.analysis.json   the same groups under Nsight Systems

For the matrix multiplies of a forward pass, by hand:
  work  = 2 * tokens * language-layer parameters      (floating point operations)
  bytes = language-layer weight bytes * forward passes (weights read once per pass)
and the trace gives the time those kernels took, so the achieved operations per
second and weight bytes per second follow without hardware counters.

Verdicts (thresholds fixed before the run):
  S1  the time per pass is flat (max / min <= 1.15) for n <= 64 and doubles at
      n in 190..500 tokens
  S2  fp32 weights take 1.8-2.2 x the bf16 time at n = 1 (twice the bytes)
  V1  GPU time per decode step at batch 16 / 64 is at most 1.2 / 1.5 x batch 1

Usage: summarize_q2.py <dir> [--json out.json]
"""
import argparse
import csv
import json
import os
import statistics as st

from hplab.cache_model import MAX_CHUNK, count_steps
from hplab.nsys import CLASS_NAMES
from hplab.stats import median as med, read_jsonl, rnd

MODELS = {"h4_C0": {"name": "Qwen3.5-4B", "lang_bytes": 7.381e9, "hybrid": True},
          "fa_F0": {"name": "Qwen3-4B", "lang_bytes": 7.267e9, "hybrid": False}}
KNEE_RANGE = (190.0, 500.0)      # n where the time per pass reaches twice the plateau
FLAT_UP_TO, FLAT_TOL = 64, 1.15  # plateau: max / min of the time for n <= 64
FP32_RATIO = (1.8, 2.2)          # fp32 time / bf16 time at n = 1
DECODE_RATIO = {16: 1.2, 64: 1.5}   # decode GPU time per step, relative to batch 1


def forward_passes(P, hybrid):
    """Prefill steps of a cold prompt of P tokens with default settings (vLLM 1479deb0,
    Triton GDN). A model without Mamba layers is cut by the token budget only."""
    return count_steps(0, P) if hybrid else -(-P // MAX_CHUNK)


def load_groups(path):
    if not os.path.exists(path):
        return []
    return [g for g in read_jsonl(path) if g.get("kind") == "measure"]


def load_nvml(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [(int(r["epoch_ns"]), float(r["util_gpu"]), float(r["util_memory"]), float(r["power_w"]))
                for r in csv.DictReader(f)]


def nvml_mean(samples, t0, t1):
    rows = [s for s in samples if t0 + (t1 - t0) // 3 <= s[0] <= t1]     # skip the ramp-up third
    if len(rows) < 3:
        return None
    return {"n": len(rows), "util_gpu": rnd(st.mean(r[1] for r in rows), 1),
            "util_memory": rnd(st.mean(r[2] for r in rows), 1), "power_w": rnd(st.mean(r[3] for r in rows), 1)}


def scaling_summary(d):
    path = os.path.join(d, "scaling.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        sc = json.load(f)
    out = {"nvml_error": sc.get("nvml_error")}
    for name, sw in sc["sweeps"].items():
        pts = sw["points"]
        low = [p["ms_per_pass"] for p in pts if p["n"] <= FLAT_UP_TO]
        out[name] = {"plateau_ms": sw["plateau_ms"], "plateau_gb_per_s": sw["plateau_gb_per_s"],
                     "knee_n": sw["knee_n"], "best_tflops": sw["best_tflops"],
                     "flat_max_over_min": rnd(max(low) / min(low)) if low else None,
                     "ms_at_n1": pts[0]["ms_per_pass"], "ms_at_largest_n": pts[-1]["ms_per_pass"],
                     "points": [{"n": p["n"], "ms": p["ms_per_pass"], "tflops": p["tflops"],
                                 "gb_per_s": p["weights_gb_per_s"], "nvml": p["nvml"]} for p in pts]}
    return out


def trace_summary(d, cfg, roof):
    path = os.path.join(d, f"{cfg}_nsys.analysis.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        a = json.load(f)
    m = MODELS[cfg]
    params = m["lang_bytes"] / 2
    by = {}
    for g in a["groups"]:
        by.setdefault((g["phase"], g["P"], g["B"]), []).append(g)
    out = {"meta": a["meta"], "prefill": {}, "decode": {}}
    for (phase, P, B), gs in sorted(by.items()):
        cls = {c: rnd(med([g["class_share"][c] for g in gs]), 4) for c in CLASS_NAMES}
        row = {"n": len(gs), "burst_ms": rnd(med([g["burst_ms"] for g in gs])),
               "busy_ms": rnd(med([g["busy_ms"] for g in gs])),
               "busy_share": rnd(med([g["busy_ms"] / g["burst_ms"] for g in gs])),
               "graph_time_share": rnd(med([g["graph_kernel_ms"] / g["kernel_ms_sum"] for g in gs])),
               "class_share": cls, "top_kernels": gs[0]["top_kernels"][:3]}
        gemm = med([g["class_ms"]["gemm"] for g in gs])
        if phase == "prefill":
            passes = forward_passes(P, m["hybrid"])
            row.update(forward_passes=passes, tokens_per_pass=rnd(P / passes, 1), gemm_ms=rnd(gemm),
                       gemm_tflops=rnd(2 * P * params / gemm * 1e3 / 1e12),
                       gemm_weights_gb_per_s=rnd(m["lang_bytes"] * passes / gemm * 1e3 / 1e9, 1))
            out["prefill"][P] = row
        else:
            steps = med([g["completion_tokens"] / g["B"] for g in gs])
            per = gemm / steps
            row.update(steps=steps, n_marks=med([g["n_marks"] for g in gs]),
                       busy_ms_per_step=rnd(row["busy_ms"] / steps), gemm_ms_per_step=rnd(per),
                       gemm_tflops=rnd(2 * B * params / per * 1e3 / 1e12),
                       gemm_weights_gb_per_s=rnd(m["lang_bytes"] / per * 1e3 / 1e9, 1))
            out["decode"][B] = row
        if roof:
            row["share_of_compute_roof"] = rnd(row["gemm_tflops"] / roof["tflops"], 4)
            row["share_of_bandwidth_roof"] = rnd(row["gemm_weights_gb_per_s"] / roof["gb_per_s"], 4)
    return out


def plain_summary(d, cfg):
    groups = load_groups(os.path.join(d, f"{cfg}_plain.jsonl"))
    nv = load_nvml(os.path.join(d, f"{cfg}_plain_nvml.csv"))
    by = {}
    for g in groups:
        by.setdefault((g["phase"], g["P"], g["B"]), []).append(g)
    out = {"prefill": {}, "decode": {}, "n_groups": len(groups),
           "n_bad": sum(1 for g in groups if g["n_errors"] or g["n_wrong_length"]), "nvml_samples": len(nv)}
    for (phase, P, B), gs in sorted(by.items()):
        nvm = [x for x in (nvml_mean(nv, g["t_send_ns"], g["t_end_ns"]) for g in gs) if x]
        row = {"n": len(gs), "wall_ms": rnd(med([g["wall_ms"] for g in gs])),
               "nvml": {k: rnd(med([x[k] for x in nvm]), 1) for k in ("util_gpu", "util_memory", "power_w")}
               if nvm else None}
        if phase == "prefill":
            row["ttft_ms"] = rnd(med([g["ttft_ms"][0] for g in gs if g["ttft_ms"]]))
            out["prefill"][P] = row
        else:
            steps = med([g["completion_tokens"] / g["B"] for g in gs])
            row.update(steps=steps, wall_ms_per_step=rnd(row["wall_ms"] / steps),
                       tokens_per_s=rnd(med([g["completion_tokens"] / g["wall_ms"] * 1e3 for g in gs]), 1))
            out["decode"][B] = row
    return out


def judge(d):
    sc = scaling_summary(d)
    roof = None
    if sc and "bf16" in sc:
        roof = {"tflops": sc["bf16"]["best_tflops"], "gb_per_s": sc["bf16"]["plateau_gb_per_s"]}
    res = {"scaling": sc, "roof": roof,
           "trace": {c: trace_summary(d, c, roof) for c in MODELS},
           "plain": {c: plain_summary(d, c) for c in MODELS}}
    v = {}
    if sc and "bf16" in sc:
        b = sc["bf16"]
        v["S1_plateau_then_knee"] = {
            "flat_max_over_min": b["flat_max_over_min"], "knee_n": b["knee_n"],
            "holds": b["knee_n"] is not None and b["flat_max_over_min"] <= FLAT_TOL
            and KNEE_RANGE[0] <= b["knee_n"] <= KNEE_RANGE[1]}
        if "fp32" in sc:
            r = sc["fp32"]["ms_at_n1"] / b["ms_at_n1"]
            v["S2_fp32_doubles_the_memory_bound_time"] = {
                "ratio_at_n1": rnd(r), "holds": FP32_RATIO[0] <= r <= FP32_RATIO[1]}
    for c in MODELS:
        t = res["trace"][c]
        if not t or 1 not in t["decode"]:
            v[f"V1_decode_step_vs_batch/{c}"] = {"holds": None}
            continue
        base = t["decode"][1]["busy_ms_per_step"]
        ratios = {B: rnd(t["decode"][B]["busy_ms_per_step"] / base) for B in t["decode"] if B != 1}
        v[f"V1_decode_step_vs_batch/{c}"] = {
            "busy_ms_per_step": {B: t["decode"][B]["busy_ms_per_step"] for B in t["decode"]},
            "ratio_to_batch_1": ratios,
            "holds": all(B in ratios and ratios[B] <= lim for B, lim in DECODE_RATIO.items())}
    res["verdicts"] = v
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = judge(a.dir)
    if res["scaling"]:
        for name in ("bf16", "fp32"):
            s = res["scaling"].get(name)
            if s:
                print(f"scaling {name}: plateau {s['plateau_ms']} ms = {s['plateau_gb_per_s']} GB/s, knee n={s['knee_n']}, "
                      f"best {s['best_tflops']} TFLOP/s, flat {s['flat_max_over_min']}")
                print("   " + "  ".join(f"n={p['n']}:{p['ms']}" for p in s["points"]))
    for c in MODELS:
        t, p = res["trace"][c], res["plain"][c]
        if t:
            for P, r in t["prefill"].items():
                print(f"{c} prefill P={P}: " + json.dumps({k: v for k, v in r.items() if k != "top_kernels"}))
            for B, r in t["decode"].items():
                print(f"{c} decode B={B}: " + json.dumps({k: v for k, v in r.items() if k != "top_kernels"}))
        print(f"{c} plain: " + json.dumps(p))
    for k, x in res["verdicts"].items():
        print("==", k, json.dumps(x))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
