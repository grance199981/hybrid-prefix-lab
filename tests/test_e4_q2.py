"""experiments/e4_profiling, Q2 scripts on inputs with known answers: the knee finder,
the per-group trace totals, and the summary verdicts (a memory-bound decode must
pass, a decode whose step time grows with the batch must fail)."""
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from experiments.e4_profiling import analyze_q2 as a2
from experiments.e4_profiling import summarize_q2 as s2
from hplab import nsys

MS = 1_000_000


def torch_importable():
    """q2_scaling imports torch at module level; on some machines that aborts the process,
    so it is tried in a subprocess first."""
    try:
        return subprocess.run([sys.executable, "-c", "import torch"], capture_output=True, timeout=300).returncode == 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not torch_importable(), reason="torch cannot be imported here")
def test_knee_is_where_the_time_doubles():
    from experiments.e4_profiling import q2_scaling as sc
    pts = [(1, 10.0), (2, 10.0), (4, 10.2), (64, 10.5), (256, 16.0), (512, 30.0), (1024, 60.0)]
    plateau, k = sc.knee(pts)
    assert plateau == 10.0 and abs(k - (256 + 256 * 4.0 / 14.0)) < 1e-6
    assert sc.knee([(1, 10.0), (2, 10.0), (4, 10.0), (8, 12.0)])[1] is None       # never doubles


def build(path):
    """Two groups 2 s apart; each has 3 step markers and 6 kernels of 2 ms:
    4 gemm (one of them from a CUDA graph) and 2 attention, with 1 ms gaps."""
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
    db.execute("insert into StringIds values (1, 'ampere_gemm_x'), (2, 'flash_fwd_splitkv_kernel'), (4, 'kernel')")
    db.execute("insert into TARGET_INFO_SESSION_START_TIME values (0, '', '')")
    groups = []
    for g in range(2):
        base = g * 2000 * MS
        for i in range(6):
            t = base + i * 3 * MS
            db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, 4, ?, ?, 7)",
                       (t, t + 2 * MS, 2 if i >= 4 else 1, 9 if i == 0 else 0))
        for i in range(3):
            db.execute("insert into NVTX_EVENTS values (?, ?, ?, null)",
                       (base + i * 5 * MS - MS, base + i * 5 * MS - MS // 2, nsys.STEP_MARK))
        groups.append({"kind": "measure", "phase": "decode", "P": 16, "B": 2, "rep": g, "wall_ms": 20.0,
                       "completion_tokens": 6, "n_errors": 0, "n_wrong_length": 0, "t_send_ns": base})
    db.commit()
    db.close()
    return groups


def test_group_totals_from_a_known_trace(tmp_path):
    db = str(tmp_path / "t.sqlite")
    groups = build(db)
    rq = tmp_path / "g.jsonl"
    rq.write_text("\n".join(json.dumps(g) for g in groups) + "\n", encoding="utf-8")
    res = a2.analyse(db, str(rq), 400.0)
    assert res["meta"]["bursts_match_groups"] and len(res["groups"]) == 2
    g = res["groups"][0]
    assert abs(g["burst_ms"] - 17.0) < 1e-9 and abs(g["busy_ms"] - 12.0) < 1e-9
    assert g["n_kernels"] == 6 and g["n_graph_kernels"] == 1 and g["n_marks"] == 3
    assert g["class_ms"]["gemm"] == 8.0 and g["class_ms"]["attention"] == 4.0
    assert abs(g["class_share"]["gemm"] - 2 / 3) < 1e-4
    assert g["top_kernels"][0]["class"] == "gemm" and g["top_kernels"][0]["n"] == 4


def write_inputs(d, step_ms):
    """A hybrid model whose decode step takes step_ms[B] of matrix multiply."""
    sc = {"nvml_error": None, "sweeps": {}}
    for name, scale in (("bf16", 1.0), ("fp32", 2.0)):
        pts = [{"n": n, "ms_per_pass": scale * 6.4 * max(1.0, n / 240), "tflops": 1.0 * n,
                "weights_gb_per_s": 600.0, "nvml": None} for n in (1, 2, 4, 64, 256, 512, 1024)]
        sc["sweeps"][name] = {"points": pts, "plateau_ms": scale * 6.4, "plateau_gb_per_s": 650.0 / scale,
                              "knee_n": 480.0, "best_tflops": 190.0}
    json.dump(sc, open(os.path.join(d, "scaling.json"), "w", encoding="utf-8"))
    groups, rows = [], []
    for B, ms in step_ms.items():
        for rep in range(3):
            groups.append({"phase": "decode", "P": 16, "B": B, "rep": rep, "wall_ms": 128 * ms + 60,
                           "completion_tokens": 128 * B, "n_errors": 0, "burst_ms": 128 * ms + 50,
                           "busy_ms": 128 * ms, "kernel_ms_sum": 128 * ms, "n_kernels": 128 * 370,
                           "n_graph_kernels": 128 * 350, "graph_kernel_ms": 120 * ms, "n_marks": 130,
                           "class_ms": {"gemm": 128 * ms * 0.9, "gdn": 0, "attention": 0, "sampling": 0,
                                        "norm_elementwise": 128 * ms * 0.1, "other": 0},
                           "class_share": {"gemm": 0.9, "gdn": 0, "attention": 0, "sampling": 0,
                                           "norm_elementwise": 0.1, "other": 0},
                           "top_kernels": [{"name": "gemvx", "class": "gemm", "n": 1, "total_ms": 1.0,
                                            "avg_us": 1.0, "share": 0.9}]})
            rows.append({"kind": "measure", "phase": "decode", "P": 16, "B": B, "rep": rep,
                         "wall_ms": 128 * ms + 60, "completion_tokens": 128 * B, "n_errors": 0,
                         "n_wrong_length": 0, "t_send_ns": 0, "t_end_ns": 1, "ttft_ms": [1.0], "e2e_ms": [2.0]})
    json.dump({"meta": {"bursts_match_groups": True}, "groups": groups},
              open(os.path.join(d, "h4_C0_nsys.analysis.json"), "w", encoding="utf-8"))
    with open(os.path.join(d, "h4_C0_plain.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")


def test_memory_bound_decode_passes_and_reports_the_roof_shares(tmp_path):
    d = str(tmp_path)
    write_inputs(d, {1: 12.0, 16: 13.0, 64: 16.0})
    res = s2.judge(d)
    v = res["verdicts"]
    assert v["S1_plateau_then_knee"]["holds"] is True
    assert v["S2_fp32_doubles_the_memory_bound_time"]["holds"] is True
    assert v["V1_decode_step_vs_batch/h4_C0"]["holds"] is True
    assert v["V1_decode_step_vs_batch/fa_F0"]["holds"] is None          # no data for that model
    dec = res["trace"]["h4_C0"]["decode"][1]
    assert dec["busy_ms_per_step"] == 12.0 and dec["gemm_ms_per_step"] == 10.8
    # 7.381 GB of weights read in 10.8 ms
    assert abs(dec["gemm_weights_gb_per_s"] - 683.4) < 0.2
    assert abs(dec["share_of_bandwidth_roof"] - 683.4 / 650.0) < 1e-3
    assert res["plain"]["h4_C0"]["decode"][64]["tokens_per_s"] == round(128 * 64 / (128 * 16.0 + 60) * 1e3, 1)


def test_decode_that_scales_with_the_batch_fails_v1(tmp_path):
    d = str(tmp_path)
    write_inputs(d, {1: 12.0, 16: 30.0, 64: 100.0})
    v = s2.judge(d)["verdicts"]
    assert v["V1_decode_step_vs_batch/h4_C0"]["holds"] is False


def test_forward_passes():
    assert [s2.forward_passes(P, True) for P in (512, 2000, 8000)] == [1, 2, 6]
    assert [s2.forward_passes(P, False) for P in (512, 2000, 8000)] == [1, 1, 4]
