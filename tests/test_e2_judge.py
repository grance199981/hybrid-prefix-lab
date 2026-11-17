"""experiments/e2_ttft_cost/judge.py on synthetic records with known answers: a real
slowdown, no slowdown, drift between blocks, and missing data; plus, when
HPLAB_E1_RECORDS points to the E1 replay records, the same statistic on E1's
C0 / C2 / C1 / C3."""
import json
import os

import pytest

from experiments.e2_ttft_cost import judge as j
from helpers import four_pass_records as make



def test_a_real_30_ms_overhead_is_confirmed():
    res = j.judge(make(overhead=30.0, saved=100))
    v = res["verdict"]
    assert v["verdict"] == "confirmed" and v["controls_pass"] is True
    assert v["V1_fixed_overhead_on_first_requests"] and v["V2_slower_despite_less_recompute"]
    assert abs(res["grid_ttft"]["first/wildchat"]["main"]["median"] - 30.0) < 4.0
    assert res["grid_ttft"]["scored/wildchat/on"]["recompute_diff"]["median"] == -100


def test_overhead_mostly_paid_back_by_saved_prefill_is_not_confirmed():
    # 30 ms overhead, 200 tokens saved (18 ms): first requests 30 ms slower,
    # later requests only 12 ms slower, below the 15 ms line
    v = j.judge(make(overhead=30.0, saved=200))["verdict"]
    assert v["verdict"] == "mixed" and v["V1_fixed_overhead_on_first_requests"] is True
    assert v["V2_slower_despite_less_recompute"] is False


def test_net_effect_on_scored_requests_is_overhead_minus_saved_prefill():
    # 40 ms overhead, 100 tokens saved (9 ms): scored requests about 31 ms slower
    res = j.judge(make(overhead=40.0, saved=100))
    assert res["verdict"]["verdict"] == "confirmed"
    assert abs(res["grid_ttft"]["scored/smith/off"]["main"]["median"] - 31.0) < 4.0


def test_no_overhead_is_refuted_with_passing_controls():
    v = j.judge(make(overhead=0.0, saved=0))["verdict"]
    assert v["controls_pass"] is True and v["verdict"] == "refuted" and v["cells_slower"] == 0


def test_a_small_overhead_is_neither_confirmed_nor_refuted():
    v = j.judge(make(overhead=10.0, saved=0))["verdict"]
    assert v["controls_pass"] is True and v["verdict"] == "mixed"


def test_a_jump_between_repeats_of_one_config_gives_no_conclusion():
    # the machine got 25 ms slower before the last block: C0b - C0a is far from 0
    v = j.judge(make(overhead=30.0, drift=(0.0, 0.0, 0.0, 25.0)))["verdict"]
    assert v["verdict"].startswith("no conclusion") and v["controls_pass"] is False


def test_abba_order_cancels_a_steady_drift():
    res = j.judge(make(overhead=30.0, saved=100, drift=(0.0, 2.0, 4.0, 6.0)))
    assert res["verdict"]["verdict"] == "confirmed"
    assert abs(res["grid_ttft"]["first/rebench"]["main"]["median"] - 30.0) < 4.0


def test_noise_too_large_to_see_the_injected_shift_gives_no_conclusion():
    v = j.judge(make(overhead=30.0, noise=60.0))["verdict"]
    assert v["verdict"].startswith("no conclusion")


def test_missing_block_or_too_few_pairs_is_incomplete():
    assert j.judge(make(overhead=30.0, skip_block="C1b"))["verdict"]["verdict"] == "incomplete"
    assert j.judge(make(overhead=30.0, sessions=2))["verdict"]["verdict"] == "incomplete"


def test_requests_without_ttft_or_with_errors_are_left_out(tmp_path):
    cells = make(overhead=30.0)
    n0 = j.judge(cells)["grid_ttft"]["scored/smith/on"]["main"]["n"]
    cells[("C1a", "smith", "on")][(1, "smith-0", 3)]["ttft_ms"] = None
    assert j.judge(cells)["grid_ttft"]["scored/smith/on"]["main"]["n"] == n0 - 1
    # load_dir: control rows and failed requests are dropped, names are parsed
    p = tmp_path / "C0a_smith_thinkon_R_s1_seed4.jsonl"
    rows = [{"sid": "control", "control": "A", "k": 0, "prompt_tokens": 9, "cached_tokens": 0},
            {"sid": "smith-1", "k": 0, "prompt_tokens": 9, "cached_tokens": 0, "ttft_ms": 1.0, "error": None},
            {"sid": "smith-1", "k": 1, "prompt_tokens": 9, "cached_tokens": 0, "ttft_ms": None, "error": "boom"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "C0b_diag_smith_thinkoff_seed4.jsonl").write_text(json.dumps(rows[1]) + "\n", encoding="utf-8")
    loaded = j.load_dir(str(tmp_path))
    assert list(loaded) == [("C0a", "smith", "on")] and list(loaded[("C0a", "smith", "on")]) == [(4, "smith-1", 0)]


E1 = os.environ.get("HPLAB_E1_RECORDS", "")


@pytest.mark.skipif(not os.path.isdir(E1), reason="set HPLAB_E1_RECORDS to run")
def test_on_the_formal_e1_records_as_a_dry_run():
    """E1's C0, C1, C3, C2 stand in for C0a, C1a, C1b, C0b (C2 behaved like C0
    and C3 like C1 there). This only checks that the script runs on real
    records and that its controls behave; it is not the E2 result."""
    cells = {}
    for f in os.listdir(E1):
        for src, dst in (("C0", "C0a"), ("C1", "C1a"), ("C3", "C1b"), ("C2", "C0b")):
            if f.startswith(src + "_") and f.endswith(".jsonl") and "_R_s1_" in f:
                _, ds, th, _, _, seed = f[:-6].split("_")
                d = cells.setdefault((dst, ds, th[5:]), {})
                for line in open(os.path.join(E1, f), encoding="utf-8"):
                    r = json.loads(line)
                    if r.get("sid") != "control" and not r.get("error"):
                        d[(int(seed[4:]), r["sid"], r["k"])] = r
    res = j.judge(cells)
    assert res["verdict"]["controls_pass"] is True
    assert all(c["main"]["n"] >= j.MIN_PAIRS for c in res["grid_ttft"].values())
