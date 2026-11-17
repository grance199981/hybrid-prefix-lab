"""experiments/e4_profiling: analyze_q1.py on a hand-built trace with known answers, the
prefill step counts the Q1 summary predicts, and the prompts of probe_client.py."""
import json
import sqlite3

from experiments.e4_profiling import analyze_q1 as a
from experiments.e4_profiling import probe_client as c
from experiments.e4_profiling import summarize_q1 as sq
from hplab import nsys

MS = 1_000_000


def test_union_of_overlapping_intervals():
    assert nsys.union_ms([(0, 2 * MS), (1 * MS, 3 * MS), (5 * MS, 6 * MS)]) == 4.0
    assert nsys.union_ms([]) == 0.0


def test_bursts_split_on_long_idle_only():
    ks = [(0, MS, "k", 0, 1), (2 * MS, 3 * MS, "k", 0, 1), (900 * MS, 901 * MS, "k", 0, 1)]
    assert [len(b) for b in nsys.bursts(ks, 400 * MS)] == [2, 1]


def test_classes():
    cl = nsys.classify
    assert cl("ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_nn") == "gemm"
    assert cl("fused_recurrent_gated_delta_rule_fwd_kernel") == "gdn"
    assert cl("void at::native::vectorized_elementwise_kernel<...>") == "norm_elementwise"
    assert cl("something_else") == "other"
    # CUDA-graph nodes: only the demangled name tells what they are
    assert cl("std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16>") == "gemm"
    assert cl("void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16>") == "gemm"
    assert cl("flash_fwd_splitkv_kernel") == "attention"
    assert cl("void vllm::act_and_mul_kernel<c10::BFloat16>") == "norm_elementwise"


class Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [hash(w) % 50000 for w in text.split() for _ in (0, 1)]}


def test_prompt_has_exactly_the_requested_length_and_differs_by_seed():
    x, y = c.token_ids(Tok(), "a", 37), c.token_ids(Tok(), "b", 37)
    assert len(x) == 37 and len(y) == 37 and x != y
    assert c.token_ids(Tok(), "a", 37) == x


def test_prefill_steps_follow_the_scheduler_rules():
    n = sq.predicted_steps
    # default vs unit 16 on Qwen3.5 (576-token block, 2048-token budget)
    assert [n("h4_C0", P) for P in (16, 560, 580, 2000, 8000)] == [1, 1, 2, 2, 6]
    assert [n("h4_C1", P) for P in (16, 560, 580, 2000, 8000)] == [1, 2, 2, 3, 7]
    # a model without Mamba layers is only cut by the token budget
    assert [n("fa_F0", P) for P in (560, 2048, 2049, 8000)] == [1, 1, 2, 4]


def test_step_labels():
    assert nsys.label_steps(6, 4) == (2, ["prefill", "prefill", "post_prefill", "decode", "decode", "decode"])
    assert nsys.label_steps(5, 4) == (1, ["prefill", "post_prefill", "decode", "decode", "decode"])
    assert nsys.label_steps(3, 4)[0] is None


def build(path, n_requests=2, steps_per_request=(2, 3)):
    """Each request: steps of 10 ms; in every step a 1 ms marker, then two 2 ms
    kernels 1 ms apart starting 1 ms after the marker ends. Requests 2 s apart."""
    db = sqlite3.connect(path)
    db.executescript("""
      create table StringIds(id integer, value text);
      create table TARGET_INFO_SESSION_START_TIME(utcEpochNs integer, utcTime text, localTime text);
      create table CUPTI_ACTIVITY_KIND_KERNEL(start integer, end integer, shortName integer,
          demangledName integer, graphNodeId integer, globalPid integer);
      create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer, end integer, bytes integer, copyKind integer);
      create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer, end integer, nameId integer);
      create table NVTX_EVENTS(start integer, end integer, text text, textId integer);
    """)
    db.execute("insert into StringIds values (1, 'ampere_gemm_x'), (2, 'vectorized_elementwise_kernel'), "
               "(3, 'cudaLaunchKernel_v7000'), (4, 'kernel')")
    db.execute("insert into TARGET_INFO_SESSION_START_TIME values (1000000000000, '', '')")
    reqs = []
    for r in range(n_requests):
        base = r * 2000 * MS
        for s in range(steps_per_request[r]):
            t = base + s * 10 * MS
            db.execute("insert into NVTX_EVENTS values (?, ?, ?, null)", (t, t + MS, nsys.STEP_MARK))
            db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, 4, 1, 0, 7)", (t + 2 * MS, t + 4 * MS))
            db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, 4, 2, 5, 7)", (t + 5 * MS, t + 7 * MS))
            db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, 3)", (t + 1500000, t + 1800000))
        reqs.append({"kind": "measure", "error": None, "P": 100 * (r + 1), "rep": 0, "ttft_ms": 1.0,
                     "first_chunk_ms": 1.0, "e2e_ms": 2.0, "completion_tokens": 1,
                     "t_send_ns": 1000000000000 + base})
    db.commit()
    db.close()
    return reqs


def test_steps_are_recovered_from_a_known_trace(tmp_path):
    db = str(tmp_path / "t.sqlite")
    reqs = build(db)
    rq = tmp_path / "r.jsonl"
    rq.write_text("\n".join(json.dumps(r) for r in reqs) + "\n", encoding="utf-8")
    res = a.analyse(db, str(rq), 400.0)
    assert res["meta"]["bursts_match_requests"] and res["meta"]["n_step_marks"] == 5
    assert [r["n_steps"] for r in res["requests"]] == [2, 3]
    assert [r["n_prefill_steps"] for r in res["requests"]] == [1, 2]     # steps - completion tokens
    assert [s["kind"] for s in res["requests"][1]["steps"]] == ["prefill", "prefill", "post_prefill"]
    s = res["requests"][1]["steps"][0]
    assert s["n_kernels"] == 2 and s["n_graph"] == 1
    assert abs(s["pre_ms"] - 1.0) < 1e-9          # marker ends at +1 ms, first kernel at +2 ms
    assert abs(s["busy_ms"] - 4.0) < 1e-9 and abs(s["gaps_ms"] - 1.0) < 1e-9
    assert abs(s["post_ms"] - 3.0) < 1e-9         # last kernel ends at +7 ms, next marker at +10 ms
    assert s["by_class"] == {"gemm": 2.0, "norm_elementwise": 2.0}
    assert abs(s["busy_graph_ms"] - 2.0) < 1e-9 and abs(s["busy_eager_ms"] - 2.0) < 1e-9
    assert s["n_kernel_launch_calls"] == 1 and abs(s["api_ms"] - 0.3) < 1e-9
    # the profiler clock agrees with the client clock in this synthetic trace
    assert res["meta"]["burst_start_minus_send_ms"]["median"] == 2.0


def test_mismatched_burst_count_gives_no_per_request_rows(tmp_path):
    db = str(tmp_path / "t.sqlite")
    reqs = build(db)
    rq = tmp_path / "r.jsonl"
    rq.write_text(json.dumps(reqs[0]) + "\n", encoding="utf-8")
    res = a.analyse(db, str(rq), 400.0)
    assert res["meta"]["bursts_match_requests"] is False and res["requests"] == []


def test_long_pattern_warms_long_contexts_and_sends_short_ones_cold():
    assert c.long_plan([8000], [16, 577], [1, 64]) == [
        ("prefill", 8000, 1, True, 0),
        ("decode", 16, 1, True, 0), ("decode", 16, 64, True, 0),
        ("warm", 577, 1, True, 0), ("decode", 577, 1, False, 576),
        ("warm", 577, 64, True, 0), ("decode", 577, 64, False, 576),
    ]


def test_pattern_defaults():
    base = ["--url", "http://x:1", "--model", "m", "--tokenizer-dir", "d", "--label", "L", "--out", "o"]
    a1 = c.parse_args(base + ["--lengths", "16"])
    assert (a1.pattern, a1.reps, a1.out_tokens) == ("cold", 5, 4)
    a2 = c.parse_args(base + ["--pattern", "long", "--reps", "1"])
    assert (a2.reps, a2.out_tokens, a2.lengths) == (1, 1, [])
