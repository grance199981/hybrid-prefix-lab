"""E12 helpers: single-turn sessions in pressure_sessions.fit and perf_table.py (no vLLM needed)."""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "e5_cache_pressure"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "e12_fix_costs"))
import pressure_sessions as S  # noqa: E402
import perf_table as T  # noqa: E402


def test_fit_default_unchanged_and_single_turn():
    sess = S.synth_sessions(4, 1, 100, 50, seed=3, vocab_ids=range(10, 30))
    assert S.fit(sess, 16384, 16) == []                       # default: single-turn sessions dropped
    one = S.fit(sess, 16384, 16, min_requests=1)
    assert len(one) == 4 and all(len(s["prompts"]) == 1 for s in one)
    firsts = {tuple(s["prompts"][0][:2]) for s in one}
    assert len(firsts) == 4                                    # no two sessions share their head
    multi = S.synth_sessions(4, 3, 100, 50, seed=3, vocab_ids=range(10, 30))
    assert S.fit(multi, 16384, 16) == S.fit(multi, 16384, 16, min_requests=2)


def _write(d, label, rows, wall, preempt, usage):
    with open(os.path.join(d, label + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"label": label, **r}) + "\n")
    json.dump({"wall_seconds": wall, "n_errors": 0, "delta": {"vllm:num_preemptions": preempt}},
              open(os.path.join(d, label + ".summary.json"), "w"))
    with open(os.path.join(d, label + ".metrics.jsonl"), "w") as f:
        f.write(json.dumps({"vllm:kv_cache_usage_perc": usage}) + "\n")


def test_perf_table(tmp_path):
    d = str(tmp_path)
    rows = [{"sid": "control", "k": 0, "ttft_ms": 1.0, "e2e_ms": 2.0},
            {"sid": "s0", "k": 0, "P": 100, "prompt_tokens": 100, "cached_tokens": 0, "pred_inf": 0,
             "ttft_ms": 100.0, "e2e_ms": 400.0, "completion_tokens": 4},
            {"sid": "s0", "k": 1, "P": 200, "prompt_tokens": 200, "cached_tokens": 64, "pred_inf": 96,
             "ttft_ms": 50.0, "e2e_ms": 350.0, "completion_tokens": 4}]
    _write(d, "C1_S_synth_kv5184_out4_seed1", rows, wall=2.0, preempt=3, usage=0.5)
    _write(d, "C1_S_synth_kv5184_out4_seed2", rows[1:2], wall=2.0, preempt=1, usage=0.9)
    (r,) = T.summarize(T.collect([d]))
    assert (r["cfg"], r["kv_mib"], r["out"], r["seeds"], r["n"]) == ("C1", 5184, 4, 2, 3)
    assert r["preemptions"] == 4 and r["usage_max"] == 0.9
    assert r["tpot_p50"] == 100.0                              # (e2e - ttft) / 3 for all three rows
    assert r["req_per_s"] == 0.75 and r["out_tok_per_s"] == 3.0
    assert r["retention"] == round(64 / 96, 4)
    assert r["ttft_p50"] == 100.0 and r["ttft_later_p50"] == 50.0
