"""experiments/e4_profiling/summarize_q2b.py on inputs with known answers: a full-attention
step time that grows with context must make the ratio fall, a flat one must not, and a
clock that does not drop must not be called a drop."""
from experiments.e4_profiling import summarize_q2b as s


def busy(full, hyb, b1=(12.0, 12.2, 12.3, 12.4)):
    """full / hyb: step ms at contexts 577, 1153, 2305 with batch 64."""
    out = {}
    for C, f, h in zip((577, 1153, 2305), full, hyb):
        out[(s.FULL, C, 64)] = f
        out[(s.HYB, C, 64)] = h
    out[(s.FULL, 577, 1)], out[(s.FULL, 2305, 1)], out[(s.HYB, 577, 1)], out[(s.HYB, 2305, 1)] = b1
    return out


def test_crossover_as_predicted_passes_every_verdict():
    v = s.judge_decode(busy(full=(24.0, 32.0, 47.0), hyb=(28.0, 30.0, 33.0)))
    assert all(v[k]["holds"] is True for k in v), v
    assert v["L2_hybrid_faster_at_long_context"]["ratio"] == round(33.0 / 47.0, 3)


def test_no_crossover_fails_l2_and_l4():
    # full attention barely grows with context: hybrid stays slower everywhere
    v = s.judge_decode(busy(full=(16.5, 17.0, 18.0), hyb=(27.0, 27.5, 28.0)))
    assert v["L1_ratio_falls_with_context"]["holds"] is True       # 1.636 > 1.618 > 1.556
    assert v["L2_hybrid_faster_at_long_context"]["holds"] is False
    assert v["L3_hybrid_still_slower_at_577"]["holds"] is True
    assert v["L4_rise_with_context"]["holds"] is False             # rise 1.5 ms, below 14


def test_hybrid_growing_as_fast_as_full_fails_l1_and_l4():
    v = s.judge_decode(busy(full=(24.0, 32.0, 47.0), hyb=(28.0, 40.0, 60.0)))
    assert v["L1_ratio_falls_with_context"]["holds"] is False
    assert v["L4_rise_with_context"]["holds"] is False             # hybrid rise 32 > 0.5 * 23


def test_batch_1_must_stay_flat():
    v = s.judge_decode(busy(full=(24.0, 32.0, 47.0), hyb=(28.0, 30.0, 33.0), b1=(12.0, 14.0, 12.3, 12.4)))
    assert v["L5_batch_1_flat"]["holds"] is False


def test_missing_cells_give_no_verdict():
    v = s.judge_decode({(s.FULL, 577, 64): 20.0})
    assert v["L2_hybrid_faster_at_long_context"]["holds"] is None
    assert v["L4_rise_with_context"]["holds"] is None and v["L5_batch_1_flat"]["holds"] is None


def test_clock_verdict():
    pts = lambda lo, hi: [{"n": 1, "clock_sm_mhz": lo}, {"n": 16, "clock_sm_mhz": lo},      # noqa: E731
                          {"n": 256, "clock_sm_mhz": 1.0}, {"n": 1024, "clock_sm_mhz": hi},
                          {"n": 4096, "clock_sm_mhz": hi}]
    assert s.judge_clock(pts(2250.0, 1550.0))["holds"] is True
    assert s.judge_clock(pts(2250.0, 2200.0))["holds"] is False
    assert s.judge_clock([{"n": 1, "clock_sm_mhz": None}, {"n": 4096, "clock_sm_mhz": None}])["holds"] is None
