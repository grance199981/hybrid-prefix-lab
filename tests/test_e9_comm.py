"""experiments/e9_two_gpu_debug/comm_test.py, the two-GPU communication test.

The first tests need no torch. The last three run the real loop on the CPU (gloo backend):
a clean run must pass (known good), a run with two wrong elements sent on purpose must be
reported with exactly those steps (known bad), and the same run judged as a normal group
must fail. They are skipped where torch with gloo cannot form a two-process group (for
example where importing torch aborts, or where gloo finds no usable network device).
"""
import json
import os
import subprocess
import sys

import pytest

from experiments.e9_two_gpu_debug import comm_test as ct

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO, "experiments", "e9_two_gpu_debug", "comm_test.py")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)


def test_plan_covers_every_op_size_and_dtype():
    k = len(ct.SIZES)
    seen = {ct.plan(s) for s in range(4 * k)}
    assert seen == {(op, n, d) for op in ("all_reduce", "all_gather") for n in ct.SIZES for d in ct.DTYPES}
    assert ct.plan(2) == ("all_reduce", ct.SIZES[2], "float32")
    assert ct.plan(7) == ("all_gather", ct.SIZES[2], "float32")


def test_sums_are_exact_in_float32():
    assert 2 * (ct.MOD - 1) < 2 ** 24
    assert ct.coeffs(0, 5) != ct.coeffs(1, 5)


def rank(mism=(), error=None):
    return {"mismatches": [{"step": s, "n_diff": d} for s, d in mism], "error": error}


def test_judge_normal_groups():
    assert ct.judge([rank(), rank()], [0, 0], False, "ok", set()) == ("ok", 0)
    assert ct.judge([rank(), rank([(3, 5)])], [0, 0], False, "ok", set()) == ("mismatch", 10)
    assert ct.judge([rank(), rank(error="CUDA error")], [0, 2], False, "ok", set()) == ("error", 11)
    assert ct.judge([rank(), None], [0, -9], False, "ok", set()) == ("error", 11)
    assert ct.judge([rank(), rank()], [-15, -15], True, "ok", set()) == ("hang", 12)


def test_judge_control():
    both = [rank([(2, 1), (7, 1)]), rank([(2, 1), (7, 1)])]
    assert ct.judge(both, [0, 0], False, "injected", {2, 7}) == ("mismatch", 0)
    # the checker missed the wrong elements: the control fails
    assert ct.judge([rank(), rank()], [0, 0], False, "injected", {2, 7}) == ("ok", 13)
    # one rank missed one of them
    assert ct.judge([both[0], rank([(2, 1)])], [0, 0], False, "injected", {2, 7})[1] == 13
    # more than the one element that was sent wrong
    assert ct.judge([both[0], rank([(2, 1), (7, 9)])], [0, 0], False, "injected", {2, 7})[1] == 13


GLOO_PROBE = """
import socket, sys, torch.distributed as d
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
d.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
d.destroy_process_group()
"""


def gloo_ok():
    try:
        return subprocess.run([sys.executable, "-c", GLOO_PROBE], capture_output=True, timeout=300,
                              env=ENV).returncode == 0
    except Exception:  # noqa: BLE001
        return False


needs_gloo = pytest.mark.skipif(not gloo_ok(), reason="torch with gloo cannot form a process group here")


def run(tmp_path, *extra):
    r = subprocess.run([sys.executable, SCRIPT, "--out", str(tmp_path), "--backend", "gloo", "--seconds", "600",
                        "--max-steps", "40", "--grace", "60", "--sizes", "64,1000,4096,8192,20000", *extra],
                       capture_output=True, text=True, timeout=600, env=ENV)
    with open(tmp_path / "summary.json", encoding="utf-8") as f:
        return r.returncode, json.load(f)


@needs_gloo
def test_cpu_known_good(tmp_path):
    rc, s = run(tmp_path)
    assert rc == 0 and s["status"] == "ok"
    assert s["steps"] == [40, 40] and s["mismatch_records"] == [0, 0]
    r0 = json.load(open(tmp_path / "rank0.json", encoding="utf-8"))
    assert {(x["op"], x["n"]) for x in r0["latency"]} == {(op, n) for op in ("all_reduce", "all_gather")
                                                         for n in (64, 1000, 4096, 8192, 20000)}


@needs_gloo
def test_cpu_known_bad_is_found(tmp_path):
    rc, s = run(tmp_path, "--inject-steps", "2,7", "--expect", "injected")
    assert rc == 0 and s["status"] == "mismatch"
    assert s["steps"] == [10, 10]              # both ranks stop at the first check after the mismatch
    for r in (0, 1):
        m = json.load(open(tmp_path / f"rank{r}.json", encoding="utf-8"))["mismatches"]
        assert [(x["step"], x["op"], x["slot"], x["n_diff"]) for x in m] == \
            [(2, "all_reduce", None, 1), (7, "all_gather", 1, 1)]
        assert all(x["first_index"] == x["n"] // 2 for x in m)


@needs_gloo
def test_cpu_known_bad_fails_a_normal_group(tmp_path):
    rc, s = run(tmp_path, "--inject-steps", "3")
    assert rc == 10 and s["status"] == "mismatch"
