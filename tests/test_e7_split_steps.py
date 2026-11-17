"""The scheduler's split rule (hplab.cache_model.count_steps) and the grouping of
experiments/e7_upstream_prs/split_steps.py."""
import json

from experiments.e2_ttft_cost import locate_overhead as lo
from experiments.e7_upstream_prs import split_steps as ss
from hplab.cache_model import BLOCK, count_steps



def test_matches_the_e2_step_model_on_main():
    """Known-good reference: locate_overhead.steps (validated in E2 against measured TTFT) is the same
    rule for a backend without internal checkpoints, as long as one chunk covers the tail."""
    n = 0
    for cached in list(range(0, 4000, 16)) + [5760, 11520, 23040]:
        for tail in (1, 5, 15, 16, 17, 63, 64, 65, 100, 300, 511, 512, 513, 575, 576, 577, 640, 1151, 1152, 1153, 2000):
            prompt = cached + tail
            if cached % BLOCK == 0:
                # the default configuration only ever hits on a block boundary; off the boundary the
                # scheduler would also stop at the next one, which the E2 model leaves out
                assert count_steps(cached, prompt, None, None) == lo.steps(cached, prompt, False, BLOCK), (cached, prompt)
            assert count_steps(cached, prompt, 16, None) == lo.steps(cached, prompt, True, BLOCK), (cached, prompt)
            n += 1
    assert n > 5000


def test_port_default_never_stops_at_a_block_boundary():
    # main stops at the last block boundary; with an alignment the final chunk saves it internally
    assert count_steps(0, 580, None, None) == 2
    assert count_steps(0, 580, None, 64) == 1
    assert count_steps(1152, 1900, None, None) == 2
    assert count_steps(1152, 1900, None, 64) == 1
    assert count_steps(0, 560, None, 64) == 1      # no boundary inside: one step either way
    assert count_steps(0, 560, None, None) == 1


def test_port_unit64_cold_prompts():
    # at most one block: the checkpoint column is not above the initial state's, so the stop stays
    for prompt in (100, 300, 500, 560, 576):
        assert count_steps(0, prompt, 64, 64) == 2, prompt
    # shorter than one hash block plus one token: nothing to save, no stop
    assert count_steps(0, 40, 64, 64) == 1
    assert count_steps(0, 64, 64, 64) == 1
    # more than one block: saved inside the final chunk
    for prompt in (577, 580, 1000, 2000):
        assert count_steps(0, prompt, 64, 64) == 1, prompt
    # main with unit 64 stops everywhere above 64 tokens
    for prompt in (100, 560, 580, 2000):
        assert count_steps(0, prompt, 64, None) >= 2, prompt


def test_port_unit64_after_a_hit():
    # start on a block boundary: the tail must be longer than one block
    assert count_steps(1152, 1152 + 300, 64, 64) == 2
    assert count_steps(1152, 1152 + 576, 64, 64) == 2
    assert count_steps(1152, 1152 + 577, 64, 64) == 1
    # start inside a block: the prompt must end past the second boundary after the start's block
    assert count_steps(1216, 1216 + 300, 64, 64) == 2            # same block
    assert count_steps(1216, 2304, 64, 64) > 1                   # ends on the second boundary
    assert count_steps(1216, 2305, 64, 64) == 1
    # tail of at most 64 tokens: the checkpoint is the hit itself
    assert count_steps(1216, 1216 + 64, 64, 64) == 1


def test_port_unit16_needs_the_checkpoint_on_the_64_grid():
    assert count_steps(0, 2000, 16, 64) == 1      # checkpoint 1,984 = 31 x 64
    assert count_steps(0, 8000, 16, 64) > count_steps(0, 8000, None, 64)   # checkpoint 7,984 is off the grid


def test_chunk_cap():
    # a long cold prompt is cut into block-aligned chunks of at most 2,048 tokens in every configuration
    assert count_steps(0, 8000, None, 64) == count_steps(0, 8000, 64, 64)
    assert count_steps(0, 8000, None, 64) >= 4


def test_rule_of_pr66938():
    """A resume inside a block may save the checkpoint once the prompt ends in a later block.
    The first case is the example in the description of PR #66938 (block 1,280, unit 128,
    resume at 1,920, prompt 2,880): 3 passes before, 1 pass after."""
    kw = dict(unit=128, alignment=128, block=1280, max_chunk=16384)
    assert count_steps(1920, 2880, rule="1479deb0", **kw) == 3
    assert count_steps(1920, 2880, rule="66938", **kw) == 1
    # block 576, unit 64, start inside a block
    assert count_steps(1216, 1216 + 300, 64, 64, rule="66938") == 2      # ends in the same block: still stops
    assert count_steps(1216, 1900, 64, 64, rule="1479deb0") == 3         # ends in the next block
    assert count_steps(1216, 1900, 64, 64, rule="66938") == 1
    # a start on a block boundary and cold prompts are not changed by the PR
    for cached, prompt in ((1152, 1452), (1152, 1728), (1152, 1729), (0, 300), (0, 576), (0, 577)):
        assert (count_steps(cached, prompt, 64, 64, rule="66938")
                == count_steps(cached, prompt, 64, 64, rule="1479deb0"))


def test_grouping_of_four_pass_records(tmp_path):
    """Two requests in four passes: a 300-token cold prompt (unit 64 adds a step on main) and a
    2,000-token cold prompt (no extra step with alignment 64). The rule must sort them apart."""
    reqs = [("s0", 0, 300), ("s1", 0, 2000)]
    for cfg, extra in (("C0a", 0.0), ("C1a", 30.0), ("C1b", 32.0), ("C0b", 2.0)):
        rows = [{"sid": "control", "k": 0}] + [
            {"sid": sid, "k": k, "prompt_tokens": P, "cached_tokens": 0, "error": None,
             "ttft_ms": 100.0 + (extra if P == 300 else extra / 10)} for sid, k, P in reqs]
        (tmp_path / f"{cfg}_wildchat_thinkoff_R_s1_seed4.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rows = ss.paired_rows(str(tmp_path))
    assert sorted(r["prompt"] for r in rows) == [300, 2000]
    by_rule = ss.group_by_steps(rows, 64, 64, "66938")["wildchat"]
    assert by_rule["1"]["n"] == 1 and by_rule["1"]["ttft_diff_median"] == 30.0
    assert by_rule["0"]["n"] == 1 and by_rule["0"]["ttft_diff_median"] == 3.0
    # known bad: without the alignment (main's rule) both requests take an extra step
    assert ss.group_by_steps(rows, 64, None, "66938")["wildchat"]["1"]["n"] == 2
