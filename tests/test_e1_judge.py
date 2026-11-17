"""experiments/e1_prefix_hits/judge.py on synthetic records with known answers (a case
that must trigger kill criterion 1, cases that must not, an incomplete grid), plus,
when HPLAB_E1_SMOKE_RECORDS points to a directory of C0 replay records (one seed,
both modes), the H3 re-check on real records."""
import os

import pytest

from experiments.e1_prefix_hits import judge as j


def rows(cached, n=20, P=1000, P_prev=900, D=898, mode="R"):
    out = [{"sid": "s", "k": 0, "P": 500, "prompt_tokens": 500, "cached_tokens": 0, "seed": 1,
            "error": None}]
    for k in range(1, n + 1):
        r = {"sid": "s", "k": k, "P": P, "prompt_tokens": P, "cached_tokens": cached, "seed": 1,
             "error": None, "P_prev": P_prev, "D_R": D, "D_I": D, "g_obs": P_prev - D,
             "residue16": (P_prev - 1) % 16, "fa16_R": min(D, P - 1) // 16 * 16, "hy576": 576,
             "hy16": 896, "ttft_ms": 50.0 + (P - cached)}
        out.append(r)
    return out


def grid(f0=960, c1=944, c4=940, override=None, drop=None):
    cells = {}
    for ds in j.DATASETS:
        for th in j.THINKS:
            for mode in ("R", "O"):
                for cfg, c in (("F0", f0), ("C1", c1), ("C4", c4)):
                    key = (cfg, ds, th, mode, 1)
                    if drop and key == drop:
                        continue
                    cells[key] = rows((override or {}).get(key, c), mode=mode)
    return cells


def test_kill_criterion_triggers_when_every_cell_is_below_both_thresholds():
    kc = j.kill_criterion_1(grid())
    assert kc["best_hybrid"] == "C1" and kc["complete"] and kc["triggered"] is True
    assert all(abs(r["gap_pp_R"] - 1.6) < 1e-6 and abs(r["ratio_R"] - 1.4) < 1e-6
               and abs(r["ratio_O"] - 1.4) < 1e-6 for r in kc["grid"])


def test_one_cell_reaching_the_ratio_in_O_replay_blocks_the_trigger():
    kc = j.kill_criterion_1(grid(override={("C1", "rebench", "off", "O", 1): 900}))
    assert kc["triggered"] is False
    assert [(r["dataset"], r["thinking"]) for r in kc["cells_reaching_a_threshold"]] == [("rebench", "off")]
    assert abs(kc["cells_reaching_a_threshold"][0]["ratio_O"] - 2.5) < 1e-6


def test_hit_rate_gap_of_ten_points_blocks_the_trigger():
    # F0 recomputes 380, C1 recomputes 500: ratio 1.32 (< 1.5) but the gap is 12 pp
    kc = j.kill_criterion_1(grid(f0=620, c1=500, c4=500))
    assert kc["best_hybrid"] == "C1"                      # tie goes to C1
    assert all(abs(r["gap_pp_R"] - 12.0) < 1e-6 and r["ratio_R"] < 1.5 for r in kc["grid"])
    assert kc["triggered"] is False and len(kc["cells_reaching_a_threshold"]) == 6


def test_incomplete_grid_gives_no_verdict():
    kc = j.kill_criterion_1(grid(drop=("F0", "smith", "on", "O", 1)))
    assert kc["complete"] is False and kc["triggered"] is None


def test_best_hybrid_is_the_candidate_with_the_smaller_recompute_median():
    assert j.pick_best_hybrid(grid(c1=930, c4=950))[0] == "C4"


def test_errors_and_first_requests_are_not_scored():
    rs = rows(960)
    rs.append(dict(rs[-1], k=99, error="boom", cached_tokens=0))
    s = j.cell_stats(rs)
    assert s["n"] == 20 and abs(s["hit_rate"] - 0.96) < 1e-9 and s["recompute_median"] == 40


def test_h2_h4_on_synthetic_rows():
    cells = grid(f0=896, c1=896, c4=896)        # 896 = floor(898 / 16) * 16 = floor(899 / 16) * 16
    assert j.h2(cells, "F0")["holds"] is True and j.h2(cells, "F0")["pooled"] == 1.0
    assert j.h4(cells, "C1")["holds"] is True
    bad = grid(f0=800, c1=800, c4=800)
    assert j.h2(bad, "F0")["holds"] is False and j.h4(bad, "C1")["holds"] is False


SMOKE = os.environ.get("HPLAB_E1_SMOKE_RECORDS", "")


@pytest.mark.skipif(not os.path.isdir(SMOKE), reason="set HPLAB_E1_SMOKE_RECORDS to run")
def test_on_the_c0_smoke_records():
    cells, errors = j.load_dir(SMOKE)
    assert len(cells) == 12 and sum(errors.values()) == 0
    res = j.judge(cells, errors)
    assert res["H3_C0"]["pooled_equal_with_fallback_rule"] == 1.0 and res["H3_C0"]["holds"] is True
    assert res["kill_criterion_1"]["complete"] is False and res["kill_criterion_1"]["triggered"] is None
    assert res["H9_C0"]["n_bad_residue"] >= 0
