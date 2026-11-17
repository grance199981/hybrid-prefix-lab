"""The shared helpers in hplab: hit prediction, the scheduler step rule, metrics parsing."""
import numpy as np

from hplab import cache_model as cm
from hplab import client, stats


def arr(xs):
    return np.asarray(xs, dtype=np.int64)


def test_lcp():
    assert cm.lcp(arr([1, 2, 3]), arr([1, 2, 4])) == 2
    assert cm.lcp(arr([1, 2]), arr([1, 2, 3])) == 2
    assert cm.lcp(arr([]), arr([1])) == 0
    assert cm.lcp([1, 2, 3], [1, 2, 3]) == 3


def test_checkpoint_position():
    assert cm.checkpoint_position(577, 576) == 576
    assert cm.checkpoint_position(576, 576) == 0
    assert cm.checkpoint_position(1, 16) == 0


def test_predict_hits_growing_session():
    # every prompt extends the previous one by 700 tokens
    base = np.arange(5000) % 97
    prompts = [arr(base[:n]) for n in (700, 1400, 2100)]
    rows = cm.predict_hits(prompts, [None] * 3, aligns=(576, 16))
    assert rows[0]["fa16_R"] == 0 and rows[0]["hy576"] == 0
    # request 1 reuses the whole of request 0: full attention on 16-token blocks,
    # the hybrid model only at request 0's prompt-tail checkpoint
    assert rows[1]["D_R"] == 700
    assert rows[1]["fa16_R"] == 688
    assert rows[1]["hy576"] == 576
    assert rows[1]["hy16"] == 688
    assert rows[2]["hy576"] == 1152


def test_count_steps_extra_stop():
    # 560-token cold prompt: the default takes one step; unit 16 stops at the
    # 544-token checkpoint first, which is the extra step that costs ~30 ms on L40
    assert cm.count_steps(0, 560) == 1
    assert cm.count_steps(0, 560, unit=16) == 2
    # a cold 1000-token prompt with unit 64: three steps on main (stops at the 576-token
    # block and at the 960-token checkpoint), one step when the backend can save the
    # checkpoint inside the chunk on a 64-token grid (PR #65186)
    assert cm.count_steps(0, 1000, unit=64) == 3
    assert cm.count_steps(0, 1000, unit=64, alignment=64) == 1
    # a hit that ends inside a block: the old validity rule still splits, PR #66938 does not
    assert cm.count_steps(640, 1200, unit=64, alignment=64) == 2
    assert cm.count_steps(640, 1200, unit=64, alignment=64, rule="66938") == 1


def test_parse_metrics():
    text = "\n".join(["# HELP x", 'vllm:prefix_cache_hits_total{a="1"} 3',
                      'vllm:prefix_cache_hits_total{a="2"} 4.5', "vllm:num_preemptions_total 2"])
    names = ["vllm:prefix_cache_hits", "vllm:num_preemptions", "vllm:absent"]
    assert client.parse_metrics(text, names) == {"vllm:prefix_cache_hits": 7.5, "vllm:num_preemptions": 2.0}
    assert client.parse_metrics(text, names, fill_missing=True)["vllm:absent"] == 0.0


def test_stats():
    assert stats.median([3, None, 1, 2]) == 2
    assert stats.median([]) is None
    assert stats.describe([1, -1, 2]) == {"n": 3, "median": 1, "share_positive": 0.667}
    fit = stats.line_fit([0, 1, 2, 3], [1, 3, 5, 7])
    assert abs(fit["slope"] - 2) < 1e-12 and abs(fit["intercept"] - 1) < 1e-12
    assert stats.line_fit([1, 1, 1], [1, 2, 3]) is None
