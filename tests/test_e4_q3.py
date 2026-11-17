"""experiments/e4_profiling, Q3 scripts on inputs with known answers: a two-process trace
where one third of the kernel time is NCCL, a one-process trace with none, and the
verdicts (a decode step that two GPUs make much faster must fail C2)."""
import json
import sqlite3

from experiments.e4_profiling import analyze_q3 as a3
from experiments.e4_profiling import summarize_q3 as s3
from hplab import nsys

MS = 1_000_000


def build(path, pids):
    """One group. Each process runs 6 kernels of 2 ms with 1 ms gaps: 4 gemm and,
    with two processes, 2 NCCL all-reduce kernels (otherwise 2 attention)."""
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
    db.execute("insert into StringIds values (1, 'ampere_gemm_x'), (2, 'flash_fwd_splitkv_kernel'), "
               "(3, 'ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)'), (4, 'kernel')")
    db.execute("insert into TARGET_INFO_SESSION_START_TIME values (0, '', '')")
    for pid in pids:
        for i in range(6):
            t = i * 3 * MS
            name = 1 if i < 4 else (3 if len(pids) > 1 else 2)
            db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, 4, ?, 0, ?)", (t, t + 2 * MS, name, pid))
    for i in range(2):
        db.execute("insert into NVTX_EVENTS values (?, ?, ?, null)", (i * 9 * MS - MS, i * 9 * MS - MS // 2, nsys.STEP_MARK))
    db.commit()
    db.close()
    return [{"kind": "measure", "phase": "decode", "P": 16, "B": 1, "rep": 0, "wall_ms": 20.0,
             "completion_tokens": 2, "n_errors": 0, "n_wrong_length": 0, "t_send_ns": 0}]


def run(tmp_path, pids):
    db = str(tmp_path / "t.sqlite")
    groups = build(db, pids)
    rq = tmp_path / "g.jsonl"
    rq.write_text("\n".join(json.dumps(g) for g in groups) + "\n", encoding="utf-8")
    return a3.analyse(db, str(rq), 400.0)


def test_two_processes_with_a_third_of_the_time_in_nccl(tmp_path):
    res = run(tmp_path, [11, 12])
    assert res["meta"]["bursts_match_groups"] and res["meta"]["n_processes"] == 2
    assert res["meta"]["nccl_kernel_names"][0][1] == 4
    g = res["groups"][0]
    assert g["n_processes"] == 2 and g["n_nccl_kernels_max"] == 2 and g["n_marks"] == 2
    assert abs(g["nccl_share_mean"] - 1 / 3) < 1e-9
    assert abs(g["busy_ms_max"] - 12.0) < 1e-9 and abs(g["busy_without_nccl_ms_max"] - 8.0) < 1e-9
    assert g["processes"][0]["class_ms"]["nccl"] == 4.0 and g["processes"][0]["class_ms"]["gemm"] == 8.0


def test_one_process_has_no_nccl(tmp_path):
    g = run(tmp_path, [11])["groups"][0]
    assert g["n_processes"] == 1 and g["n_nccl_kernels_max"] == 0 and g["nccl_share_mean"] == 0.0
    assert g["processes"][0]["class_ms"]["attention"] == 4.0


def summaries(dec_share, pre_share, one_ms, two_ms):
    tr2 = {"decode": {1: {"nccl_share": dec_share}}, "prefill": {8000: {"nccl_share": pre_share}}}
    return tr2, {"decode": {1: {"wall_ms_per_step": one_ms}}}, {"decode": {1: {"wall_ms_per_step": two_ms}}}


def test_verdicts():
    v = s3.judge_pair(*summaries(0.40, 0.15, 12.0, 11.0))
    assert v["C1_nccl_share_decode_above_prefill"]["holds"] is True
    assert v["C2_two_gpus_do_not_speed_up_decode_by_a_quarter"]["holds"] is True
    v = s3.judge_pair(*summaries(0.05, 0.15, 12.0, 6.5))            # cheap all-reduce, near-ideal scaling
    assert v["C1_nccl_share_decode_above_prefill"]["holds"] is False
    assert v["C2_two_gpus_do_not_speed_up_decode_by_a_quarter"]["holds"] is False
    v = s3.judge_pair(None, {}, {})
    assert all(x["holds"] is None for x in v.values())
