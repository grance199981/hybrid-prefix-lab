"""experiments/e7_upstream_prs/steps_from_log.py: matching the server's iteration log
to the replayed requests."""
from experiments.e7_upstream_prs import steps_from_log as sl


def line(i, nreq, tok, gen=0):
    return (f"(APIServer pid=1) INFO 11-17 04:44:19 [loggers.py:201] Engine 000: Iteration({i}): {nreq} context requests, "
            f"{tok} context tokens, {gen} generation requests, {gen} generation tokens, iteration elapsed time: 9.00 ms")


def req(cached, prompt, k=0):
    return {"file": "x.jsonl", "sid": "s", "k": k, "control": None, "prompt": prompt, "cached": cached}


def test_parse_keeps_only_prefill_iterations():
    lines = [line(0, 1, 1728), line(1, 1, 256), line(2, 1, 16), line(3, 0, 0, 1), "noise", line(4, 1, 300)]
    assert sl.parse_log(lines) == [(0, 1, 1728), (1, 1, 256), (2, 1, 16), (4, 1, 300)]


def test_known_good_matches_main_with_unit_64():
    """The pattern seen in the smoke run of the base build: a cold prompt of 2,000 tokens under unit 64
    runs as 1,728 + 256 + 16, which is what the scheduler rule without internal checkpoints gives."""
    it = sl.parse_log([line(0, 1, 1728), line(1, 1, 256), line(2, 1, 16), line(3, 0, 0, 1)])
    matched, err = sl.match(it, [req(0, 2000)])
    assert err is None and matched[0]["steps"] == 3 and matched[0]["chunks"] == [1728, 256, 16]
    rows, res = sl.compare(matched, 64, None, "66938")
    assert res["same"] == 1 and rows[0]["sim"] == 3


def test_adjacent_requests_without_a_generation_gap_are_separated_by_token_count():
    # two requests whose prefill iterations follow each other directly
    it = sl.parse_log([line(50, 1, 576), line(51, 1, 384), line(52, 1, 40), line(53, 1, 192), line(54, 1, 16)])
    matched, err = sl.match(it, [req(0, 1000), req(1792, 2000, 1)])
    assert err is None
    assert [m["steps"] for m in matched] == [3, 2]


def test_known_bad_overshoot_is_reported_and_nothing_is_guessed():
    it = sl.parse_log([line(0, 1, 576), line(1, 1, 500)])
    matched, err = sl.match(it, [req(0, 1000)])
    assert matched == [] and "needs 1000 tokens" in err


def test_known_bad_leftover_and_batched_iterations():
    it = sl.parse_log([line(0, 1, 300), line(1, 1, 64)])
    matched, err = sl.match(it, [req(0, 300)])
    assert len(matched) == 1 and "left over" in err
    it = sl.parse_log([line(0, 2, 600)])
    matched, err = sl.match(it, [req(0, 300), req(0, 300)])
    assert matched == [] and "not one request at a time" in err


def test_disagreement_with_the_simulator_is_counted():
    # with the PR (alignment 64) a cold prompt of 2,000 tokens should take one step; three were logged
    it = sl.parse_log([line(0, 1, 1728), line(1, 1, 256), line(2, 1, 16)])
    matched, _ = sl.match(it, [req(0, 2000)])
    rows, res = sl.compare(matched, 64, 64, "66938")
    assert res["same"] == 0 and res["simulated_vs_measured"] == {"1->3": 1}
