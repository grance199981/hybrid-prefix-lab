"""experiments/e9_two_gpu_debug: how graph_repro.py judges a run (the GPU path has its own
controls in the run itself: the `control` mode with a wrong checksum sent on purpose and
the `plain` mode without a profiler), the summary tables, and trace_counts.py on a
hand-built trace."""
import json
import sqlite3
import sys

from experiments.e9_two_gpu_debug import graph_repro as gr
from experiments.e9_two_gpu_debug import summarize as sm
from experiments.e9_two_gpu_debug import trace_counts as tc


def rank(mism=(), error=None):
    return {"mismatches": [{"step": s, "kind": k} for s, k in mism], "error": error}


def test_judge_normal_modes():
    assert gr.judge([rank(), rank()], [0, 0], False, "ok", -1) == ("ok", 0)
    assert gr.judge([rank(), rank([(3, "own_slot")])], [0, 0], False, "ok", -1) == ("mismatch", 10)
    assert gr.judge([rank(), rank(error="CUDA error")], [0, 2], False, "ok", -1) == ("error", 11)
    assert gr.judge([rank(), None], [0, -9], False, "ok", -1) == ("error", 11)
    assert gr.judge([rank(), rank()], [-15, -15], True, "ok", -1) == ("hang", 12)


def test_judge_control():
    both = [rank([(5, "hidden")]), rank([(5, "hidden")])]
    assert gr.judge(both, [0, 0], False, "injected", 5) == ("mismatch", 0)
    assert gr.judge([rank(), rank()], [0, 0], False, "injected", 5) == ("ok", 13)            # the checks missed it
    assert gr.judge([both[0], rank()], [0, 0], False, "injected", 5)[1] == 13                # one rank missed it
    assert gr.judge([both[0], rank([(6, "hidden")])], [0, 0], False, "injected", 5)[1] == 13  # at the wrong step
    assert gr.judge([both[0], rank([(5, "own_slot")])], [0, 0], False, "injected", 5)[1] == 13


def test_constants_match_the_nccl_header():
    assert (gr.NCCL_BF16, gr.NCCL_F32, gr.NCCL_SUM, gr.NCCL_MAX) == (9, 7, 0, 2)
    import ctypes
    assert ctypes.sizeof(gr.UniqueId) == 128


def test_redact_numbers_each_uuid_once():
    a, b = "01234567-89ab-cdef-0123-456789abcdef", "fedcba98-7654-3210-fedc-ba9876543210"
    text = json.dumps({"env": f"GPU-{a},GPU-{b}", "gpu": [a, b], "other": "1234-5678"})
    out = json.loads(sm.redact(text))
    assert out == {"env": "<gpu-uuid-0>,<gpu-uuid-1>", "gpu": ["<gpu-uuid-0>", "<gpu-uuid-1>"], "other": "1234-5678"}


def test_groups_finds_the_incomplete_group(tmp_path, capsys, monkeypatch):
    rows = [{"kind": "warmup", "phase": "decode", "B": 64, "rep": 0, "n_errors": 0, "completion_tokens": 0}]
    for rep in range(3):
        rows.append({"kind": "measure", "phase": "prefill", "B": 1, "rep": rep, "n_errors": 0, "completion_tokens": 1})
        rows.append({"kind": "measure", "phase": "decode", "B": 64, "rep": rep, "n_errors": 0,
                     "completion_tokens": 8192 if rep < 2 else 128})
    p = tmp_path / "fa_T2n_nsys.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["summarize.py", "groups", str(p)])
    sm.main()
    assert capsys.readouterr().out.strip() == \
        "fa_T2n_nsys.jsonl: 6 groups, 3 decode groups, incomplete: decode group 3 (128 of 8192 tokens)"


def test_trace_counts(tmp_path):
    path = str(tmp_path / "t.sqlite")
    db = sqlite3.connect(path)
    db.executescript("""
      create table StringIds(id integer, value text);
      create table CUPTI_ACTIVITY_KIND_KERNEL(start integer, end integer, shortName integer,
          graphNodeId integer, graphId integer, globalPid integer);
    """)
    db.execute("insert into StringIds values (1, 'ncclDevKernel_AllReduce_Sum_bf16_RING_LL'), (2, 'gemm')")
    pid = lambda p: p << 24  # noqa: E731  (the process id sits in bits 24-47 of globalPid)
    rows = [(0, 1, 1, 5, 1, pid(11)), (2, 3, 2, None, None, pid(11)), (0, 1, 1, 6, 1, pid(12))]
    db.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?, ?)", rows)
    db.commit()
    db.close()
    r = tc.counts(path)
    assert r["processes"] == [{"pid": 11, "kernels": 2, "nccl_kernels": 1}, {"pid": 12, "kernels": 1, "nccl_kernels": 1}]
    assert r["nccl_names"] == [["ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 2]]
    assert r["rows_with_graph_node"]["graphNodeId"] == {"rows": 3, "with_id": 2}
