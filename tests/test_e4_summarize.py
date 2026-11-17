"""experiments/e4_profiling/summarize_q1.py on synthetic inputs with known answers: a
config pair with a 35 ms extra step (must pass), one whose trace disagrees with the
request timing (must fail T3), and a full-attention pair with no difference."""
import json
import os

from experiments.e4_profiling import summarize_q1 as sq

LENGTHS = (16, 560, 580, 2000, 8000)


def step(kind, wall, busy):
    return {"kind": kind, "wall_ms": wall, "pre_ms": 0.5, "busy_ms": busy, "busy_graph_ms": busy / 2,
            "busy_eager_ms": busy / 2, "gaps_ms": wall - busy - 0.5, "post_ms": 0.0, "n_kernels": 100,
            "n_graph": 50, "n_api_calls": 60, "api_ms": 1.0, "n_kernel_launch_calls": 50,
            "n_graph_launch_calls": 1, "by_class": {"gemm": busy}}


def prefill_steps(cfg, P):
    """Prefill steps of the config and how many more than its default partner."""
    n = sq.predicted_steps(cfg, P)
    return n, n - sq.predicted_steps(cfg.replace("C1", "C0").replace("F16", "F0"), P)


def write(d, cfg, extra_ttft, extra_wall, reps=3):
    for mode, shift in (("plain", 0.0), ("nsys", 4.0)):
        with open(os.path.join(d, f"{cfg}_{mode}.jsonl"), "w", encoding="utf-8") as f:
            for rep in range(reps):
                for P in LENGTHS:
                    n, more = prefill_steps(cfg, P)
                    t = 80 + 0.05 * P + extra_ttft * more + shift
                    f.write(json.dumps({"kind": "measure", "error": None, "P": P, "rep": rep,
                                        "ttft_ms": t, "first_chunk_ms": t, "e2e_ms": t + 36,
                                        "prompt_tokens": P, "cached_tokens": 0,
                                        "completion_tokens": 4}) + "\n")
    reqs = []
    for rep in range(reps):
        for P in LENGTHS:
            n, more = prefill_steps(cfg, P)
            base = (60 + 0.05 * P) / (n - more)
            steps = [step("prefill", base, base * 0.5) for _ in range(n - more)]
            steps += [step("prefill", extra_wall, 15.0) for _ in range(more)]
            steps += [step("post_prefill", 4.0, 1.7)] + [step("decode", 11.5, 11.2)] * 3
            reqs.append({"P": P, "rep": rep, "ttft_ms": 0, "first_chunk_ms": 0, "e2e_ms": 0,
                         "n_steps": len(steps), "n_prefill_steps": n, "steps": steps})
    with open(os.path.join(d, f"{cfg}_nsys.analysis.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": {"bursts_match_requests": True}, "requests": reqs}, f)


def test_a_35_ms_extra_step_passes_every_check(tmp_path):
    d = str(tmp_path)
    write(d, "h4_C0", 0, 0)
    write(d, "h4_C1", 35.0, 38.0)
    write(d, "fa_F0", 0, 0)
    write(d, "fa_F16", 0, 0)
    v = sq.judge(d)["verdicts"]
    assert v["T1_step_counts_match_the_scheduler_rules"] == {"h4": True, "h9": None, "fa": True}
    t2 = v["T2_plain_ttft_cost_of_the_extra_step"]
    assert t2["holds"] is True and t2["with_extra_step"] == {560: 35.0, 2000: 35.0, 8000: 35.0}
    assert t2["without"] == {16: 0.0, 580: 0.0}
    assert v["T3_trace_agrees_with_request_timing"]["holds"] is True
    assert v["T4_full_attention_has_no_extra_step"]["holds"] is True
    assert v["profiler_overhead_on_ttft_ms"]["median"] == 4.0
    last = v["T5_breakdown_of_the_extra_step"]["h4/560"]
    assert last["wall_ms"] == 38.0 and last["busy_ms"] == 15.0 and last["by_class_ms"]["gemm"] == 15.0
    assert v["T6_decode_step_ms"]["h4_C1"] == {"busy_ms": 11.2, "wall_ms": 11.5}


def test_a_trace_far_from_the_request_timing_fails_t3(tmp_path):
    d = str(tmp_path)
    write(d, "h4_C0", 0, 0)
    write(d, "h4_C1", 35.0, 90.0)       # the trace says the step takes 90 ms
    v = sq.judge(d)["verdicts"]
    assert v["T2_plain_ttft_cost_of_the_extra_step"]["holds"] is True
    assert v["T3_trace_agrees_with_request_timing"]["holds"] is False


def test_no_extra_cost_fails_t2_and_missing_configs_give_no_verdict(tmp_path):
    d = str(tmp_path)
    write(d, "h4_C0", 0, 0)
    write(d, "h4_C1", 0.0, 38.0)
    v = sq.judge(d)["verdicts"]
    assert v["T2_plain_ttft_cost_of_the_extra_step"]["holds"] is False
    assert v["T4_full_attention_has_no_extra_step"]["holds"] is None
