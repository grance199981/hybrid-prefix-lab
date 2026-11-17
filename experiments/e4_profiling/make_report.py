"""Build the E4 number table (e4_numbers.json) and figures (figures/*.svg) from the
summary files of the four collections (copies are in results/):
  --q1  q1_posthoc.json    posthoc_q1.py      the extra prefill step (figure 1)
  --q2  q2_summary.json    summarize_q2.py    roofline and decode at context 16 (figure 2a)
  --q2b q2b_summary.json   summarize_q2b.py   decode at long context (figure 2b)
  --q3  q3_summary.json    summarize_q3.py    two GPUs (figure 3; optional)
  --ncu ncu_medians.json   tools/ncu_extract.py --medians   single kernels under
                           Nsight Compute on a laptop GPU (optional)

--palette css writes the figures with CSS variables instead of fixed colours (for
embedding in an HTML page that defines them); the default is fixed colours on white.
Figure labels are in Chinese, like the README they belong to.

Usage (from the repository root, with R=experiments/e4_profiling/results):
  python experiments/e4_profiling/make_report.py --q1 $R/q1_posthoc.json
      --q2 $R/q2_summary.json --q2b $R/q2b_summary.json --q3 $R/q3_summary.json
      --ncu $R/ncu_medians.json --out-dir experiments/e4_profiling
"""
import argparse
import json
import math
import os

FIXED = {"ink": "#14181f", "ink2": "#47505e", "line": "#dde2ea", "axis": "#8a93a1",
         "s1": "#2a78d6", "s2": "#eb6834", "idle": "#b8c0cc", "surface": "#ffffff"}
CSS = {"ink": "var(--ink)", "ink2": "var(--ink-2)", "line": "var(--line)", "axis": "var(--line-strong)",
       "s1": "var(--s1)", "s2": "var(--s2)", "idle": "var(--line-strong)", "surface": "var(--surface)"}
FONT = "font-family=\"system-ui, 'Segoe UI', 'Microsoft YaHei', 'PingFang SC', sans-serif\""


def text(x, y, s, c, size=12, anchor="start", weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{c}"{w}>{s}</text>'


def svg(w, h, body, label, pal, standalone):
    bg = f'<rect width="{w}" height="{h}" fill="{pal["surface"]}"/>' if standalone else ""
    ns = ' xmlns="http://www.w3.org/2000/svg"' if standalone else ""
    return f'<svg{ns} viewBox="0 0 {w} {h}" role="img" aria-label="{label}" {FONT}>{bg}{body}</svg>\n'


def fig_extra_step(q1, pal, standalone):
    """Where the extra prefill step goes: GPU busy against GPU idle, 4B and 9B, prompt length 560."""
    rows = [("Qwen3.5-4B", q1["pairs"]["h4"]["560"]), ("Qwen3.5-9B", q1["pairs"]["h9"]["560"])]
    x0, scale, body = 150, 10.0, ""
    for i, (name, p) in enumerate(rows):
        y = 44 + i * 58
        busy, idle = p["busy_diff_ms"], p["idle_diff_ms"]
        body += text(x0 - 12, y + 19, name, pal["ink"], 13, "end")
        body += f'<rect x="{x0}" y="{y}" width="{busy * scale - 2:.1f}" height="28" rx="3" fill="{pal["s1"]}"/>'
        body += f'<rect x="{x0 + busy * scale:.1f}" y="{y}" width="{idle * scale:.1f}" height="28" rx="3" fill="{pal["idle"]}"/>'
        body += text(x0 + busy * scale / 2, y + 19, f"{busy:.1f}", "#ffffff", 12, "middle", 600)
        body += text(x0 + busy * scale + idle * scale / 2, y + 19, f"{idle:.1f}", pal["ink"], 12, "middle", 600)
        body += text(x0 + (busy + idle) * scale + 10, y + 19, f"共 {p['burst_diff_ms']:.1f} 毫秒", pal["ink2"], 12)
    body += f'<rect x="{x0}" y="14" width="11" height="11" rx="2" fill="{pal["s1"]}"/>' + text(x0 + 17, 24, "GPU 在跑 kernel", pal["ink2"])
    body += f'<rect x="{x0 + 130}" y="14" width="11" height="11" rx="2" fill="{pal["idle"]}"/>' + text(x0 + 147, 24, "GPU 空着，等 CPU 发起下一个 kernel", pal["ink2"])
    return svg(720, 156, body, "多停的那一步多出来的时间：GPU 在跑和 GPU 空着各占多少，4B 和 9B", pal, standalone)


def fig_roofline_decode(q2, q2b, pal, standalone):
    """Left: the measured roofline (time per pass against tokens per step, log-log).
    Right: decode step time against context length at batch 64, both models."""
    body = ""
    # ---- left panel
    X0, X1, Y0, Y1 = 52, 340, 236, 40
    pts = q2["scaling"]["bf16"]["points"]
    lo, hi = math.log10(4), math.log10(512)

    def xy(n, ms):
        return X0 + math.log2(n) / 12 * (X1 - X0), Y0 - (math.log10(ms) - lo) / (hi - lo) * (Y0 - Y1)

    for ms in (4, 16, 64, 256):
        y = xy(1, ms)[1]
        body += f'<line x1="{X0}" x2="{X1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{pal["line"]}"/>' + text(X0 - 6, y + 4, ms, pal["ink2"], 11, "end")
    for n in (1, 16, 256, 4096):
        body += text(xy(n, 4)[0], Y0 + 16, n, pal["ink2"], 11, "middle")
    knee = q2["scaling"]["bf16"]["knee_n"]
    kx = xy(knee, 4)[0]
    body += f'<line x1="{kx:.1f}" x2="{kx:.1f}" y1="{Y1}" y2="{Y0}" stroke="{pal["axis"]}" stroke-dasharray="4 3"/>'
    body += text(kx - 6, Y1 + 10, f"拐点 {knee:.0f}", pal["ink2"], 11, "end")
    line = " ".join("%.1f,%.1f" % xy(p["n"], p["ms"]) for p in pts)
    body += f'<polyline points="{line}" fill="none" stroke="{pal["s1"]}" stroke-width="2"/>'
    body += "".join('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (*xy(p["n"], p["ms"]), pal["s1"]) for p in pts)
    body += text(xy(2, pts[0]["ms"])[0], xy(2, pts[0]["ms"])[1] - 10, "受访存限制", pal["ink"], 12)
    body += text(xy(700, 40)[0], xy(700, 40)[1] + 22, "受算力限制", pal["ink"], 12)
    body += text(X0, 22, "(a) 缩放实验：每遍矩阵乘的时间，毫秒", pal["ink"], 12, "start", 600)
    body += text((X0 + X1) / 2, Y0 + 34, "一步处理的 token 数", pal["ink2"], 11, "middle")
    # ---- right panel
    A0, A1, B0, B1 = 418, 700, 236, 40
    series = {"h4_C0": ("混合（Qwen3.5-4B）", pal["s1"]), "fa_F0": ("全注意力（Qwen3-4B）", pal["s2"])}
    data = {c: sorted((int(C), per_b["64"]["busy_ms_per_step"]) for C, per_b in q2b["trace"][c]["decode"].items()
                      if "64" in per_b) for c in series}
    cmax = max(C for d in data.values() for C, _ in d)
    top = 10 * math.ceil(max(v for d in data.values() for _, v in d) / 10)

    def ab(C, ms):
        return A0 + C / cmax * (A1 - A0), B0 - ms / top * (B0 - B1)

    for ms in range(0, top + 1, 10):
        y = ab(0, ms)[1]
        body += f'<line x1="{A0}" x2="{A1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{pal["line"]}"/>' + text(A0 - 6, y + 4, ms, pal["ink2"], 11, "end")
    for C, _ in data["fa_F0"]:
        body += text(ab(C, 0)[0], B0 + 16, f"{C:,}", pal["ink2"], 11, "middle")
    for c, (name, col) in series.items():
        line = " ".join("%.1f,%.1f" % ab(C, v) for C, v in data[c])
        body += f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="2"/>'
        body += "".join('<circle cx="%.1f" cy="%.1f" r="3.5" fill="%s" stroke="%s" stroke-width="1.5"/>'
                        % (*ab(C, v), col, pal["surface"]) for C, v in data[c])
        C, v = data[c][-1]
        other = dict(data["fa_F0" if c == "h4_C0" else "h4_C0"])[C]
        body += text(ab(C, v)[0], ab(C, v)[1] + (-10 if v > other else 30), f"{name} {v:.1f}", pal["ink"], 12, "end")
    body += text(A0, 22, "(b) 同时 64 个请求：每步 GPU 在跑的时间，毫秒", pal["ink"], 12, "start", 600)
    body += text((A0 + A1) / 2, B0 + 34, "每个请求的上下文长度（token）", pal["ink2"], 11, "middle")
    return svg(720, 280, body, "左：量出来的屋顶线，拐点 217 个 token。右：同时 64 个请求时每步的时间随上下文长度的变化，两种模型",
               pal, standalone)


Q3_PAIRS = (("Qwen3.5-4B", "h4_C0", "h4_T2n"), ("Qwen3-4B", "fa_F0", "fa_T2n"))
Q3_CELLS = (("prefill", "512"), ("prefill", "2000"), ("prefill", "8000"), ("decode", "1"), ("decode", "16"), ("decode", "64"))


def q3_values(q3):
    """Per model: NCCL share of GPU kernel time (two GPUs) and two-GPU time / one-GPU time, per workload."""
    out = {}
    for model, one, two in Q3_PAIRS:
        share, ratio = [], []
        for phase, key in Q3_CELLS:
            share.append(q3["trace"][two][phase][key]["nccl_share"])
            field = "ttft_ms" if phase == "prefill" else "wall_ms_per_step"
            ratio.append(q3["plain"][two][phase][key][field] / q3["plain"][one][phase][key][field])
        out[model] = {"share": share, "ratio": ratio}
    return out


def fig_two_gpus(q3, pal, standalone):
    """Left: share of GPU kernel time spent in NCCL kernels with two GPUs.
    Right: time with two GPUs as a percentage of the time with one. Six workloads, both models."""
    vals = q3_values(q3)
    cols = {"Qwen3.5-4B": pal["s1"], "Qwen3-4B": pal["s2"]}
    names = {"Qwen3.5-4B": "混合（Qwen3.5-4B）", "Qwen3-4B": "全注意力（Qwen3-4B）"}
    ticks = ["512", "2,000", "8,000", "1", "16", "64"]

    def panel(x0, x1, key, top, step, title, ref=None):
        b, y0, y1 = "", 228, 44
        gw = (x1 - x0) / 6
        for v in range(0, top + 1, step):
            y = y0 - v / top * (y0 - y1)
            b += f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{pal["line"]}"/>' + text(x0 - 6, y + 4, v, pal["ink2"], 11, "end")
        if ref is not None:
            y = y0 - ref / top * (y0 - y1)
            b += f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{pal["axis"]}" stroke-dasharray="4 3"/>'
            b += text(x1, y - 5, "和单卡一样", pal["ink2"], 11, "end")
        for i in range(6):
            for j, model in enumerate(vals):
                v = vals[model][key][i] * 100
                h = v / top * (y0 - y1)
                x = x0 + i * gw + gw / 2 + (j - 1) * 19 + 1
                b += f'<rect x="{x:.1f}" y="{y0 - h:.1f}" width="17" height="{h:.1f}" rx="2" fill="{cols[model]}"/>'
                b += text(x + 8.5, y0 - h - 4, f"{v:.0f}", pal["ink"], 10, "middle")
            b += text(x0 + i * gw + gw / 2, y0 + 15, ticks[i], pal["ink2"], 11, "middle")
        b += f'<line x1="{x0 + 3 * gw:.1f}" x2="{x0 + 3 * gw:.1f}" y1="{y0 + 4}" y2="{y0 + 36}" stroke="{pal["line"]}"/>'
        b += text(x0 + 1.5 * gw, y0 + 32, "prefill（token 数）", pal["ink2"], 11, "middle")
        b += text(x0 + 4.5 * gw, y0 + 32, "decode（并发数）", pal["ink2"], 11, "middle")
        b += text(x0, 22, title, pal["ink"], 12, "start", 600)
        return b

    body = panel(52, 340, "share", 60, 20, "(a) 两卡时通信 kernel 占 GPU 时间，%")
    body += panel(418, 700, "ratio", 120, 20, "(b) 两卡耗时 ÷ 单卡耗时，%", ref=100)
    for j, model in enumerate(vals):
        x = 190 + j * 190
        body += f'<rect x="{x}" y="278" width="11" height="11" rx="2" fill="{cols[model]}"/>' + text(x + 17, 288, names[model], pal["ink2"])
    return svg(720, 300, body, "两张卡张量并行：左，通信 kernel 占 GPU 时间的比例；右，两卡耗时相对单卡的百分比；六种工况，两个模型",
               pal, standalone)


def numbers(q1, q2, q2b, src, ncu=None, q3=None):
    h4, h9 = q1["pairs"]["h4"]["560"], q1["pairs"]["h9"]["560"]
    b = q2["scaling"]["bf16"]
    dec = {c: {B: q2["trace"][c]["decode"][B]["busy_ms_per_step"] for B in ("1", "16", "64")} for c in ("h4_C0", "fa_F0")}
    tps = {c: {B: q2["plain"][c]["decode"][B]["tokens_per_s"] for B in ("1", "16", "64")} for c in ("h4_C0", "fa_F0")}
    out = {
        "sources": src,
        "q1_extra_step_prompt_560": {
            "Qwen3.5-4B": {"request_longer_ms": h4["burst_diff_ms"], "gpu_busy_ms": h4["busy_diff_ms"],
                           "gpu_idle_ms": h4["idle_diff_ms"], "ttft_plain_diff_ms": h4["ttft_plain_diff_ms"]},
            "Qwen3.5-9B": {"request_longer_ms": h9["burst_diff_ms"], "gpu_busy_ms": h9["busy_diff_ms"],
                           "gpu_idle_ms": h9["idle_diff_ms"], "ttft_plain_diff_ms": h9["ttft_plain_diff_ms"]}},
        "q2_roofline_l40_bf16_2560x9216": {
            "plateau_ms_per_pass": b["plateau_ms"], "plateau_weights_gb_per_s": b["plateau_gb_per_s"],
            "knee_tokens_per_step": b["knee_n"], "best_tflops": b["best_tflops"],
            "fp32_over_bf16_at_1_token": q2["verdicts"]["S2_fp32_doubles_the_memory_bound_time"]["ratio_at_n1"]},
        "q2_decode_context_16_gpu_busy_ms_per_step": dec,
        "q2_decode_context_16_tokens_per_s": tps,
        "q2_verdicts": {k: v.get("holds") for k, v in q2["verdicts"].items()},
    }
    if q2b:
        out["q2b_decode_gpu_busy_ms_per_step"] = {
            c: {C: {B: r["busy_ms_per_step"] for B, r in per_b.items()} for C, per_b in q2b["trace"][c]["decode"].items()}
            for c in ("h4_C0", "fa_F0")}
        out["q2b_decode_tokens_per_s"] = {
            c: {C: {B: r.get("tokens_per_s") for B, r in per_b.items()} for C, per_b in q2b["plain"][c]["decode"].items()}
            for c in ("h4_C0", "fa_F0")}
        out["q2b_verdicts"] = q2b["verdicts"]
    if ncu:      # single kernels under Nsight Compute on an RTX 5060 laptop (tools/ncu_extract.py medians)
        out["local_ncu_rtx5060_single_kernels"] = {
            case: {"n_kernels": r["n_kernels"], "compute_throughput_pct_of_peak": r["compute_pct"],
                   "memory_throughput_pct_of_peak": r["memory_pct"], "dram_bytes_read": r["dram_read"],
                   "dram_read_gb_per_s": round(r["dram_read_rate"] / 1e9, 1), "l2_hit_rate_pct": r["l2_hit_pct"],
                   "kernel_duration_ms_under_ncu": round(r["duration"] * 1e3, 3)}
            for case, r in ncu.items()}
    if q3:       # two GPUs, tensor parallel over NCCL, against one GPU in the same batch (250 W power limit)
        cells = [f"{ph}_{k}" for ph, k in Q3_CELLS]
        vals = q3_values(q3)
        out["q3_two_gpus_nccl"] = {
            "note": "two L40 over PCIe across NUMA nodes, --disable-custom-all-reduce, power limit 250 W; "
                    "the Qwen3-4B trace is from the pass that failed in its last group (decode_64 has n = 2)",
            "nccl_share_of_gpu_kernel_time": {m: dict(zip(cells, (round(x, 4) for x in v["share"]))) for m, v in vals.items()},
            "two_gpu_over_one_gpu": {m: dict(zip(cells, (round(x, 3) for x in v["ratio"]))) for m, v in vals.items()},
            "decode_64_tokens_per_s": {m: {"one_gpu": q3["plain"][one]["decode"]["64"]["tokens_per_s"],
                                           "two_gpus": q3["plain"][two]["decode"]["64"]["tokens_per_s"]}
                                       for m, one, two in Q3_PAIRS},
            "verdicts": {m: {k: x.get("holds") for k, x in v.items()} for m, v in q3["verdicts"].items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q1", required=True)
    ap.add_argument("--q2", required=True)
    ap.add_argument("--q2b", required=True)
    ap.add_argument("--ncu", default=None, help="ncu_medians.json of the local Nsight Compute run (optional)")
    ap.add_argument("--q3", default=None, help="q3_summary.json of the two-GPU run (optional; summarize_q3.py)")
    ap.add_argument("--out-dir", required=True, help="where e4_numbers.json and figures/ go")
    ap.add_argument("--palette", choices=["fixed", "css"], default="fixed")
    a = ap.parse_args()
    q1, q2, q2b = (json.load(open(p, encoding="utf-8")) for p in (a.q1, a.q2, a.q2b))
    pal, standalone = (FIXED, True) if a.palette == "fixed" else (CSS, False)
    os.makedirs(os.path.join(a.out_dir, "figures"), exist_ok=True)
    figs = {"fig1_extra_step.svg": fig_extra_step(q1, pal, standalone),
            "fig2_roofline_decode.svg": fig_roofline_decode(q2, q2b, pal, standalone)}
    q3 = json.load(open(a.q3, encoding="utf-8")) if a.q3 else None
    if q3:
        figs["fig3_two_gpus.svg"] = fig_two_gpus(q3, pal, standalone)
    for name, s in figs.items():
        with open(os.path.join(a.out_dir, "figures", name), "w", encoding="utf-8", newline="\n") as f:
            f.write(s)
    if a.palette == "fixed":
        src = {k: os.path.basename(p) for k, p in
               (("q1", a.q1), ("q2", a.q2), ("q2b", a.q2b)) + ((("q3", a.q3),) if a.q3 else ())}
        with open(os.path.join(a.out_dir, "e4_numbers.json"), "w", encoding="utf-8", newline="\n") as f:
            ncu = json.load(open(a.ncu, encoding="utf-8")) if a.ncu else None
            json.dump(numbers(q1, q2, q2b, src, ncu, q3), f, indent=1, ensure_ascii=False)
    print("wrote", ", ".join(figs), "and e4_numbers.json" if a.palette == "fixed" else "")


if __name__ == "__main__":
    main()
