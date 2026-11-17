"""Checks for the offline accounting math (hplab.cache_model) on synthetic token sequences.

Each case has a hand-computed expected value, and there is a known-good input
(identical prompt) and a known-bad input (unrelated prompt) through the same
code path as the real run.
"""
import numpy as np

from hplab.cache_model import checkpoint_position, lcp, predict_hits, summarize


def arr(*xs):
    return np.asarray(xs, dtype=np.int64)


def seq(n, start=0):
    return np.arange(start, start + n, dtype=np.int64)


def test_lcp():
    assert lcp(arr(1, 2, 3), arr(1, 2, 4)) == 2
    assert lcp(arr(1, 2, 3), arr(1, 2, 3, 4)) == 3
    assert lcp(arr(9), arr(1)) == 0
    assert lcp(arr(), arr(1)) == 0


def test_checkpoint_position_matches_pr50551_numbers():
    # prompts 61 / 3729 / 8151 / 11145 -> hits 48 / 3728 / 8144 / 11136 (PR #50551 body)
    assert [checkpoint_position(p, 16) for p in (61, 3729, 8151, 11145)] == [48, 3728, 8144, 11136]
    assert checkpoint_position(528, 528) == 0      # needs one token past the boundary
    assert checkpoint_position(529, 528) == 528
    assert checkpoint_position(16, 16) == 0
    assert checkpoint_position(17, 16) == 16


def test_known_good_identical_prompt():
    p = seq(1000)
    rows = predict_hits([p, p.copy()], [None, None])
    r = rows[1]
    assert r["D_R"] == 1000
    assert r["fa16_R"] == 992            # floor(999/16)*16
    assert r["hy528"] == 528             # floor(999/528)*528
    assert r["hy16"] == 992
    assert rows[0]["fa16_R"] == 0 and rows[0]["hy16"] == 0


def test_known_bad_unrelated_prompt():
    rows = predict_hits([seq(1000), seq(1000, start=50_000)], [None, None])
    r = rows[1]
    assert r["D_R"] == 0
    assert all(r[k] == 0 for k in ("fa16_R", "fa16_I", "hy528", "hy16", "hy_exact_end_I"))


def test_template_gap_makes_tail_checkpoint_unusable():
    # prompt 0 has 801 tokens; its unit-16 checkpoint sits at 800.
    p0 = seq(801)
    # g = 2: next prompt shares only the first 799 tokens -> checkpoint at 800 unusable.
    p1 = np.concatenate([seq(799), seq(300, start=90_000)])
    r = predict_hits([p0, p1], [None, None])[1]
    assert r["g_obs"] == 2 and r["residue16"] == 0
    assert r["fa16_R"] == 784            # floor(799/16)*16
    assert r["hy16"] == 0                # no earlier state to fall back to
    assert r["hy528"] == 528             # 528 <= 799, still usable
    # same gap, residue 1: prompt of 802 tokens -> checkpoint at 800, shared 800 -> usable.
    q0 = seq(802)
    q1 = np.concatenate([seq(800), seq(300, start=90_000)])
    r = predict_hits([q0, q1], [None, None])[1]
    assert r["g_obs"] == 2 and r["residue16"] == 1
    assert r["hy16"] == 800


def test_fallback_to_older_checkpoint():
    p0 = seq(401)                                             # checkpoint at 400
    p1 = seq(801)                                             # checkpoint at 800
    p2 = np.concatenate([seq(799), seq(100, start=90_000)])   # shares 799 with p1, 401 with p0
    r = predict_hits([p0, p1, p2], [None, None, None])[2]
    assert r["hy16"] == 400


def test_ideal_round_trip_extends_reuse_for_full_attention_only():
    p0 = seq(500)
    ideal0 = seq(700)                                         # prompt + 200 re-rendered reply tokens
    p1 = np.concatenate([seq(700), seq(50, start=90_000)])
    r = predict_hits([p0, p1], [ideal0, None])[1]
    assert r["D_R"] == 500 and r["D_I"] == 700
    assert r["fa16_R"] == 496 and r["fa16_I"] == 688
    assert r["hy16"] == 496                                   # main keeps no decode-end state
    assert r["hy_exact_end_I"] == 700


def test_summarize_excludes_first_request():
    p = seq(1000)
    rows = []
    for sid in ("a", "b"):
        for r in predict_hits([p, p.copy()], [None, None]):
            r.update({"sid": sid, "after_user_query": True})
            r.setdefault("g_obs", None)
            rows.append(r)
    s = summarize(rows)
    assert s["n_requests"] == 2 and s["n_sessions"] == 2
    assert abs(s["fa16_R"]["hit_rate"] - 0.992) < 1e-9
    assert abs(s["hy528"]["hit_rate"] - 0.528) < 1e-9
    assert s["hy528"]["recompute_median"] == 472
    assert abs(s["gap_pp_fa16_R_minus_hy528"] - 46.4) < 1e-9
    assert abs(s["recompute_median_ratio_hy528_over_fa16_R"] - 59.0) < 1e-9
