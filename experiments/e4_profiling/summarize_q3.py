"""Q3 summary and verdicts: two-GPU tensor parallelism (h4_T2n, fa_T2n) against one
GPU (h4_C0, fa_F0) in the same batch, and the share of GPU kernel time spent in
NCCL kernels. The two-GPU configs switch vLLM's own all-reduce kernel off
(--disable-custom-all-reduce), so all communication goes through NCCL.

Input directory (scripts/run_probe.sh with probe_client.py --pattern groups and analyze_q3.py):
  <cfg>_nsys.analysis.json   per-group totals per worker process
  <cfg>_plain.jsonl          the same groups without a profiler (wall time, TTFT)

Verdicts (thresholds fixed before the run):
  C1  the NCCL share of GPU kernel time is higher in decode at batch 1 than in the
      prefill of 8,000 tokens
  C2  with two GPUs, wall time per decode step at batch 1 is at least 0.75 x one GPU
Groups that failed (errors, or no tokens) are left out of the trace summary.

Usage: summarize_q3.py <dir> [--json out.json]
"""
import argparse
import json
import os

from hplab.stats import median as med, read_jsonl, rnd

PAIRS = {"Qwen3.5-4B": ("h4_C0", "h4_T2n"), "Qwen3-4B": ("fa_F0", "fa_T2n")}
C1_DECODE_B, C1_PREFILL_P = 1, 8000      # NCCL share: decode at batch 1 above prefill of 8,000 tokens
C2_MIN_RATIO = 0.75                      # decode batch 1, wall per step, two GPUs / one GPU


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [g for g in read_jsonl(path) if g.get("kind") == "measure"]


def trace_summary(d, cfg):
    path = os.path.join(d, f"{cfg}_nsys.analysis.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        a = json.load(f)
    by = {}
    dropped = []
    for g in a["groups"]:
        if g["n_errors"] or not g["completion_tokens"]:   # a group that failed is not used
            dropped.append([g["phase"], g["P"], g["B"], g["rep"]])
            continue
        by.setdefault((g["phase"], g["P"] if g["phase"] == "prefill" else g["B"]), []).append(g)
    out = {"meta": a["meta"], "dropped_groups": dropped, "prefill": {}, "decode": {}}
    for (phase, key), gs in sorted(by.items()):
        steps = med([g["completion_tokens"] / g["B"] for g in gs]) if phase == "decode" else 1
        out[phase][key] = {
            "n": len(gs), "n_processes": med([g["n_processes"] for g in gs]),
            "burst_ms": rnd(med([g["burst_ms"] for g in gs])),
            "busy_ms_per_step": rnd(med([g["busy_ms_max"] for g in gs]) / steps),
            "busy_without_nccl_ms_per_step": rnd(med([g["busy_without_nccl_ms_max"] for g in gs]) / steps),
            "nccl_share": rnd(med([g["nccl_share_mean"] for g in gs]), 4),
            "nccl_kernels_per_mark": rnd(med([g["n_nccl_kernels_max"] / g["n_marks"] for g in gs if g["n_marks"]]), 1),
            "n_marks": med([g["n_marks"] for g in gs])}
    return out


def plain_summary(d, cfg):
    by = {}
    for g in load_jsonl(os.path.join(d, f"{cfg}_plain.jsonl")):
        by.setdefault((g["phase"], g["P"] if g["phase"] == "prefill" else g["B"]), []).append(g)
    out = {"prefill": {}, "decode": {}}
    for (phase, key), gs in sorted(by.items()):
        row = {"n": len(gs), "n_bad": sum(1 for g in gs if g["n_errors"] or g["n_wrong_length"]),
               "wall_ms": rnd(med([g["wall_ms"] for g in gs]))}
        if phase == "prefill":
            row["ttft_ms"] = rnd(med([g["ttft_ms"][0] for g in gs if g["ttft_ms"]]))
        else:
            steps = med([g["completion_tokens"] / g["B"] for g in gs])
            row.update(wall_ms_per_step=rnd(row["wall_ms"] / steps),
                       tokens_per_s=rnd(med([g["completion_tokens"] / g["wall_ms"] * 1e3 for g in gs]), 1))
        out[phase][key] = row
    return out


def judge_pair(tr2, pl1, pl2):
    """tr2: trace summary of the two-GPU config; pl1 / pl2: plain summaries."""
    v = {}
    dec = (tr2 or {}).get("decode", {}).get(C1_DECODE_B, {}).get("nccl_share")
    pre = (tr2 or {}).get("prefill", {}).get(C1_PREFILL_P, {}).get("nccl_share")
    v["C1_nccl_share_decode_above_prefill"] = {
        "decode_b1": dec, "prefill_8000": pre, "holds": None if dec is None or pre is None else dec > pre}
    w1 = (pl1 or {}).get("decode", {}).get(C1_DECODE_B, {}).get("wall_ms_per_step")
    w2 = (pl2 or {}).get("decode", {}).get(C1_DECODE_B, {}).get("wall_ms_per_step")
    v["C2_two_gpus_do_not_speed_up_decode_by_a_quarter"] = {
        "one_gpu_ms": w1, "two_gpu_ms": w2, "ratio": rnd(w2 / w1) if w1 and w2 else None,
        "holds": None if not (w1 and w2) else w2 / w1 >= C2_MIN_RATIO}
    return v


def judge(d):
    res = {"trace": {}, "plain": {}, "verdicts": {}, "ratios": {}}
    for model, (one, two) in PAIRS.items():
        for c in (one, two):
            res["trace"][c] = trace_summary(d, c)
            res["plain"][c] = plain_summary(d, c)
        res["verdicts"][model] = judge_pair(res["trace"][two], res["plain"][one], res["plain"][two])
        r = {}
        for P, row in res["plain"][two]["prefill"].items():
            base = res["plain"][one]["prefill"].get(P, {}).get("ttft_ms")
            if base and row.get("ttft_ms"):
                r[f"prefill_{P}_ttft"] = rnd(row["ttft_ms"] / base)
        for B, row in res["plain"][two]["decode"].items():
            base = res["plain"][one]["decode"].get(B, {}).get("wall_ms_per_step")
            if base:
                r[f"decode_{B}_wall_per_step"] = rnd(row["wall_ms_per_step"] / base)
        res["ratios"][model] = r           # two GPUs / one GPU; report only except decode batch 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = judge(a.dir)
    for c, t in res["trace"].items():
        print(f"== {c}: trace {t['meta'] if t else None}")
        for phase in ("prefill", "decode"):
            for key, row in (t[phase] if t else {}).items():
                print(f"   {phase} {key}: {row} | plain {res['plain'][c][phase].get(key)}")
    for model in PAIRS:
        print("== ratios two GPUs / one GPU", model, json.dumps(res["ratios"][model]))
        for k, x in res["verdicts"][model].items():
            print("==", model, k, json.dumps(x))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
