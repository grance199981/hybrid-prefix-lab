"""experiments/e5_cache_pressure/lru_sim.py on inputs with known answers: a pool
too large to evict must give the infinite-cache prediction, and small pools must
lose exactly the blocks a hand walk through the free queue says they lose."""
import json
import os
import random
import sys

import pytest

from hplab.cache_model import lcp as np_lcp

# the experiment's modules import each other by plain name, as when run as scripts
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "e5_cache_pressure"))
import lru_sim as L  # noqa: E402
from pressure_sessions import cross_session_prefix, infinite_cache_prediction, synth_sessions  # noqa: E402

ARCH = {"fa64": ("fa", None), "hy576": ("hy", 576), "hy16": ("hy", 16)}
N_OUT = 16


def sess(sid, lengths, base):
    """One session whose prompts are prefixes of each other; ids start at `base`."""
    return {"sid": sid, "prompts": [list(range(base, base + n)) for n in lengths]}


def diverging_sessions(n, seed):
    """Each prompt drops the last g tokens of the previous one before it grows,
    the way a chat template's generation prompt is replaced by the reply."""
    rng, out = random.Random(seed), []
    for i in range(n):
        prompt, prompts = [i] + [rng.randrange(10, 500) for _ in range(rng.randrange(40, 900))], []
        for _ in range(rng.randrange(2, 7)):
            prompts.append(prompt)
            keep = len(prompt) - rng.choice([0, 1, 3, 7, 20, 40])
            prompt = prompt[:keep] + [rng.randrange(10, 500) for _ in range(rng.randrange(60, 900))]
        out.append({"sid": f"div-{i}", "prompts": prompts})
    return out


def serial(sessions):
    return [(s["sid"], k) for s in sessions for k in range(len(s["prompts"]))]


def interleaved(sessions):
    longest = max(len(s["prompts"]) for s in sessions)
    return [(s["sid"], k) for k in range(longest) for s in sessions if k < len(s["prompts"])]


def hits(rows):
    return {(r["sid"], r["k"]): r["hit"] for r in rows}


# ----------------------------------------------------- unlimited pool = E0 rule
@pytest.mark.parametrize("arch", sorted(ARCH))
@pytest.mark.parametrize("rules", ["M0", "M1"])
def test_huge_pool_gives_the_infinite_cache_prediction(arch, rules):
    synth = synth_sessions(6, 5, 700, 333, seed=3, vocab_ids=range(1000, 3000))
    for sessions in (synth, diverging_sessions(8, seed=5)):
        want = infinite_cache_prediction(sessions, arch)
        assert any(v > 0 for v in want.values())
        for schedule, overlap in ((serial(sessions), 1), (interleaved(sessions), 1), (interleaved(sessions), 3)):
            events = L.events_from_schedule(sessions, schedule, N_OUT, overlap)
            rows, summary = L.simulate(events, 10**5, *ARCH[arch], rules=rules)
            assert hits(rows) == want
            assert summary["evictions"] == 0 and summary["alloc_failures"] == 0


def test_synth_sessions_are_deterministic_and_share_nothing():
    a = synth_sessions(5, 3, 100, 50, seed=1, vocab_ids=range(50, 60))
    assert a == synth_sessions(5, 3, 100, 50, seed=1, vocab_ids=range(50, 60))
    assert a != synth_sessions(5, 3, 100, 50, seed=2, vocab_ids=range(50, 60))
    assert [len(p) for p in a[0]["prompts"]] == [100, 150, 200]
    assert all(s["prompts"][1][:100] == s["prompts"][0] for s in a)
    assert len({s["prompts"][0][0] for s in a}) == 5            # 10 ids >= 5 sessions: first token differs
    assert set(cross_session_prefix(a).values()) == {0}
    # 3 distinct ids for 5 sessions: the index takes two tokens, so at most one is shared
    b = synth_sessions(5, 3, 100, 50, seed=1, vocab_ids=[7, 8, 9])
    assert len({tuple(s["prompts"][0][:2]) for s in b}) == 5
    assert max(cross_session_prefix(b).values()) <= 1


def test_real_sessions_render_like_e1_and_the_simulator_agrees_on_them(monkeypatch):
    """Recorded sessions through the stand-in tokenizer of tests/helpers.py (no transformers)."""
    pytest.importorskip("jinja2")
    import pressure_sessions as es
    from hplab.sessions import render_session
    from helpers import TOOLS, StandInTokenizer, chat_session, struct_session
    raw = [("chat-0", chat_session(5), None), ("chat-1", chat_session(3), None),
           ("struct-0", struct_session(4), TOOLS), ("short-0", chat_session(1), None)]
    monkeypatch.setitem(es.LOADERS, "wildchat", lambda path, n, seed, tools_json=None: iter(raw[:n]))
    tok = StandInTokenizer("qwen3.5-4b_chat_template.jinja")
    got = es.real_sessions("wildchat", "x", "x", None, 1, 4, "on", 10**6, 8, tok=tok)
    order = [sid for sid, _, _ in raw]
    random.Random(1).shuffle(order)                       # the seed fixes the order, as in E1
    assert [s["sid"] for s in got] == [sid for sid in order if sid != "short-0"]   # one request only
    for s in got:
        messages, tools = next((m, t) for sid, m, t in raw if sid == s["sid"])
        want = [p.tolist() for p in render_session(tok, messages, tools, True, 10**6)[0]]
        assert s["prompts"] == want and all(type(x) is int for x in s["prompts"][0])
    # requests that leave no room for the output are dropped, then sessions with fewer than two
    chat0 = next(s for s in got if s["sid"] == "chat-0")
    limit = len(chat0["prompts"][2]) + 8
    cut = es.real_sessions("wildchat", "x", "x", None, 1, 4, "on", limit, 8, tok=tok)
    assert len(next(s for s in cut if s["sid"] == "chat-0")["prompts"]) == 3
    assert all(len(p) + 8 <= limit for s in cut for p in s["prompts"])
    # the two chat sessions open with the same first turn: all of it is shared
    cross = cross_session_prefix(got)
    assert cross["chat-0"] == cross["chat-1"] == len(chat0["prompts"][0]) and cross["struct-0"] < 10
    for arch in sorted(ARCH):
        want = infinite_cache_prediction(got, arch)
        rows, _ = L.simulate(L.events_from_schedule(got, interleaved(got), N_OUT, 2), 10**5, *ARCH[arch])
        assert hits(rows) == want


def test_lcp_and_events():
    assert L.lcp([1, 2, 3], [1, 2, 4, 5]) == 2 and L.lcp([1, 2], [1, 2, 3]) == 2 and L.lcp([], [1]) == 0
    big = list(range(9000))
    assert L.lcp(big, big[:8500] + [-1] + big[8501:]) == 8500
    # the list version gives what hplab's numpy version gives
    rng = random.Random(7)
    for _ in range(200):
        a = [rng.randrange(3) for _ in range(rng.randrange(0, 50))]
        b = a[:rng.randrange(0, len(a) + 1)] + [rng.randrange(3) for _ in range(rng.randrange(0, 20))]
        assert L.lcp(a, b) == np_lcp(a, b)
    s = [sess("A", [100, 200], 0), sess("B", [100], 10**6)]
    name = lambda ev: [(n, r["sid"], r["k"]) for n, r in ev]      # noqa: E731
    assert name(L.events_from_schedule(s, [("A", 0), ("B", 0), ("A", 1)], 4, overlap=2)) == [
        ("start", "A", 0), ("prompt_done", "A", 0), ("start", "B", 0), ("prompt_done", "B", 0),
        ("end", "A", 0),                 # A's next turn may not start while this one is in flight
        ("start", "A", 1), ("prompt_done", "A", 1), ("end", "B", 0), ("end", "A", 1)]
    assert L.make_requests(s, 4)[("A", 1)] == {"sid": "A", "k": 1, "P": 200, "n_out": 4, "lcp_prev": {0: 100}}


# ------------------------------------------------- full attention, by hand
def test_full_attention_second_visit_loses_exactly_the_evicted_tail_blocks():
    """Pool of 8 blocks of 64. A and B alternate; prompts 256 then 320 tokens, 16 out.
    A request of 256 + 16 tokens uses 5 blocks: 4 full ones (hashed) and a tail (not).

    A0: takes b0..b4. Freed tail first: the tail goes to the front, A3 A2 A1 A0 to the back.
        queue: [b4 b5 b6 b7 A3 A2 A1 A0]
    B0: prompt takes b4..b7, its output block pops A3 (eviction 1).
        queue after: [tail, A2 A1 A0 B3 B2 B1 B0]
    A1 (320 tokens, 256 shared with A0): chain A0 A1 A2, A3 is gone -> hit 3 * 64 = 192
        (unlimited cache: 256). Needs 5 - 3 prompt blocks: the free tail and B3 (eviction 2),
        then one output block: B2 (eviction 3). Holds 3 + 3 = 6 blocks.
        queue after: [tail, B1 B0 A4 A3' A2 A1 A0]
    B1: chain B0 B1, B2 is gone -> hit 128. Takes the tail, A4 and A3' (evictions 4, 5),
        then A2 for its output (eviction 6)."""
    sessions = [sess("A", [256, 320], 0), sess("B", [256, 320], 10**6)]
    events = L.events_from_schedule(sessions, [("A", 0), ("B", 0), ("A", 1), ("B", 1)], 16)
    rows, summary = L.simulate(events, 8, "fa")
    assert [r["hit"] for r in rows] == [0, 0, 192, 128]
    assert [r["blocks_allocated"] for r in rows] == [5, 5, 3, 4]
    assert summary == {"evictions": 6, "evictions_state": 0, "alloc_failures": 0, "peak_in_use": 6}
    want = infinite_cache_prediction(sessions, "fa64")
    assert want[("A", 1)] == 256 and want[("B", 1)] == 256


def test_a_request_that_does_not_fit_is_marked_failed_and_leaves_the_cache_alone():
    sessions = [sess("A", [256, 320], 0), sess("B", [1000], 10**6)]
    events = L.events_from_schedule(sessions, [("A", 0), ("B", 0), ("A", 1)], 16)
    rows, summary = L.simulate(events, 8, "fa")           # B needs 16 blocks
    assert [(r["hit"], r["failed"]) for r in rows] == [(0, False), (0, True), (256, False)]
    assert summary["alloc_failures"] == 1 and summary["evictions"] == 0


# ------------------------------------------------------- hybrid, by hand
def test_losing_the_one_state_block_drops_the_hybrid_hit_to_zero():
    """Hybrid, 576-token blocks, unit 576, rules M1, pool 9; 16 output tokens.
    A0 (1200 tokens): 3 attention blocks (two full, one tail) and 2 state blocks;
    checkpoint at 1152. Free queue after A0:
        [state, tail, b5 b6 b7 b8, S(1152), A1, A0]      (S went back at prompt end)
    B0 (2400 tokens): 5 attention + 2 state = 7 blocks: the six without a hash and S.
    So exactly one cached block is evicted, the state block; both attention blocks
    of A stay. A1 (1800 tokens, 1200 shared) then hits 0 instead of 1152.

    Full attention with the same token budget (81 blocks of 64): A0 leaves 19
    cached blocks behind 62 free ones; B0 takes 38 of those: nothing is evicted and
    A1 hits 1152. With C0 (2000 tokens, 32 blocks) after B0, 25 free blocks are
    left, so 7 of A's tail blocks go (A18..A12) and A1 hits 12 * 64 = 768: a part."""
    sessions = [sess("A", [1200, 1800], 0), sess("B", [2400], 10**6), sess("C", [2000], 2 * 10**6)]
    assert infinite_cache_prediction(sessions, "hy576")[("A", 1)] == 1152
    assert infinite_cache_prediction(sessions, "fa64")[("A", 1)] == 1152
    ev = lambda schedule: L.events_from_schedule(sessions, schedule, 16)      # noqa: E731

    _, s = L.simulate(ev([("A", 0), ("B", 0)]), 9, "hy", 576, "M1")
    assert (s["evictions"], s["evictions_state"]) == (1, 1)
    rows, s = L.simulate(ev([("A", 0), ("B", 0), ("A", 1)]), 9, "hy", 576, "M1")
    assert rows[-1]["hit"] == 0 and s["alloc_failures"] == 0
    rows, s = L.simulate(ev([("A", 0), ("B", 0), ("A", 1)]), 81, "fa")
    assert rows[-1]["hit"] == 1152 and s["evictions"] == 0

    schedule = [("A", 0), ("B", 0), ("C", 0), ("A", 1)]
    assert L.simulate(ev(schedule), 9, "hy", 576, "M1")[0][-1]["hit"] == 0
    rows, s = L.simulate(ev(schedule), 81, "fa")
    assert rows[-1]["hit"] == 768 and s["alloc_failures"] == 0


def test_losing_the_newest_state_block_falls_back_to_the_older_checkpoint():
    """Rules M1, pool 12. After A0 (1200) and A1 (1800, hits 1152, checkpoint 1728):
        [state, tail, b7..b11, S(1728), A2, A1, A0, S(1152)]
    S(1728) went back at A1's prompt end; S(1152) was held as a hit until A1 ended.
    B0 (3000 tokens: 6 attention + 2 state) takes the seven blocks without a hash
    and S(1728). A2 (2400, 1800 shared) cannot use 1728 any more and falls back to
    1152, whose state block and two attention blocks are all still cached.
    Full attention with the same budget (108 blocks) evicts nothing: 1792."""
    sessions = [sess("A", [1200, 1800, 2400], 0), sess("B", [3000], 10**6)]
    assert infinite_cache_prediction(sessions, "hy576")[("A", 2)] == 1728
    ev = lambda schedule: L.events_from_schedule(sessions, schedule, 16)      # noqa: E731
    _, s = L.simulate(ev([("A", 0), ("A", 1), ("B", 0)]), 12, "hy", 576, "M1")
    assert (s["evictions"], s["evictions_state"]) == (1, 1)
    rows, s = L.simulate(ev([("A", 0), ("A", 1), ("B", 0), ("A", 2)]), 12, "hy", 576, "M1")
    assert [r["hit"] for r in rows] == [0, 1152, 0, 1152] and s["alloc_failures"] == 0
    rows, s = L.simulate(ev([("A", 0), ("A", 1), ("B", 0), ("A", 2)]), 108, "fa")
    assert [r["hit"] for r in rows] == [0, 1152, 0, 1792] and s["evictions"] == 0


def test_m0_and_m1_differ_when_a_request_starts_during_another_one_s_decode():
    """Pool 12, unit 576. D0 and A0 (1200 tokens each: 3 attention + 2 state) overlap;
    B0 (1800 tokens: 4 + 2) starts while A0 is still decoding.

    M1: checkpoints go back when the prompt is done. When B0 starts the queue is
        [state, tail, b10 b11, S_D, S_A, D1, D0]: B0 takes six blocks, among them
        S_D and S_A. A0's checkpoint is gone before A0 has finished, and A1 hits 0.
    M0: S_A is held until A0 ends. When B0 starts the queue is
        [state, tail, b10 b11, D1, D0, S_D]: B0 evicts D's two attention blocks.
        A0 then ends, [.. A1 A0 S_A] are all cached, and A1 hits 1152."""
    sessions = [sess("D", [1200], 2 * 10**6), sess("A", [1200, 1800], 0), sess("B", [1800], 10**6)]
    reqs = L.make_requests(sessions, 16)
    order = [("start", "D", 0), ("prompt_done", "D", 0), ("start", "A", 0), ("prompt_done", "A", 0),
             ("end", "D", 0), ("start", "B", 0), ("prompt_done", "B", 0), ("end", "A", 0),
             ("end", "B", 0), ("start", "A", 1), ("prompt_done", "A", 1), ("end", "A", 1)]
    events = [(name, reqs[(sid, k)]) for name, sid, k in order]
    m0, s0 = L.simulate(events, 12, "hy", 576, "M0")
    m1, s1 = L.simulate(events, 12, "hy", 576, "M1")
    assert hits(m0)[("A", 1)] == 1152 and hits(m1)[("A", 1)] == 0
    assert s0["alloc_failures"] == 0 and s1["alloc_failures"] == 0
    assert (s0["evictions"], s0["evictions_state"]) == (2, 0)
    assert s1["evictions_state"] == 2


def test_unit_16_hit_inside_a_block_costs_the_copies():
    """A0: 1000 tokens, unit 16: checkpoint at 992 inside block 1 -> 2 attention
    blocks + 2 state blocks + 1 for the checkpoint inside a block = 5.
    A1: 1300 tokens, hits 992: copies block 1 (attention blocks 1 and 2 are new)
    and takes 2 state blocks + 1 for the copied state + 1 for its checkpoint (1296) = 6.
    With unit 576 the same session hits 576 and takes 2 + 2 blocks."""
    sessions = [sess("A", [1000, 1300], 0)]
    events = L.events_from_schedule(sessions, [("A", 0), ("A", 1)], 16)
    rows, _ = L.simulate(events, 100, "hy", 16)
    assert [(r["hit"], r["blocks_allocated"]) for r in rows] == [(0, 5), (992, 6)]
    rows, _ = L.simulate(events, 100, "hy", 576)
    assert [(r["hit"], r["blocks_allocated"]) for r in rows] == [(0, 4), (576, 4)]


# ------------------------------------------------------------------ pool search
@pytest.mark.parametrize("arch,rules", [("fa64", "M0"), ("hy576", "M0"), ("hy16", "M0"), ("hy16", "M1")])
def test_min_pool_without_loss_is_tight(arch, rules):
    sessions = synth_sessions(5, 4, 700, 400, seed=2, vocab_ids=range(1000, 3000))
    schedule = interleaved(sessions)
    a, unit = ARCH[arch]
    p = L.min_pool_without_loss(sessions, schedule, a, unit, rules, N_OUT, 1, 4000)
    events = L.events_from_schedule(sessions, schedule, N_OUT)
    want = infinite_cache_prediction(sessions, arch)

    def lossless(pool):
        rows, summary = L.simulate(events, pool, a, unit, rules)
        return summary["alloc_failures"] == 0 and hits(rows) == want

    assert p is not None and 1 < p < 4000
    assert lossless(p) and not lossless(p - 1)
    assert L.min_pool_without_loss(sessions, schedule, a, unit, rules, N_OUT, 1, p - 1) is None


# ------------------------------------------------------------------ trace CLI
def test_trace_cli_rebuilds_the_events_from_client_records(tmp_path, capsys):
    """Records of the hand-computed full-attention case, sent one after another:
    the simulator must reproduce the measured hits from the timestamps alone."""
    recs, t = [], 10**9
    for seq, (sid, k, P, cached, prev) in enumerate([("A", 0, 256, 0, {}), ("B", 0, 256, 0, {}),
                                                     ("A", 1, 320, 192, {"0": 256}),
                                                     ("B", 1, 320, 128, {"0": 256})]):
        recs.append({"sid": sid, "k": k, "seq": seq, "P": P, "n_out": 16, "t_send_ns": t,
                     "t_end_ns": t + 9 * 10**6, "ttft_ms": 3.0, "cached_tokens": cached,
                     "lcp_prev": prev, "error": None})
        t += 10**7
    recs.append({"sid": "control", "control": "A", "k": 0, "P": 1500, "cached_tokens": 0, "error": None})
    trace, out = tmp_path / "t.jsonl", tmp_path / "sim.json"
    trace.write_text("\n".join(json.dumps(r) for r in recs[::-1]) + "\n", encoding="utf-8")
    assert L.main(["--trace", str(trace), "--arch", "fa", "--pool-blocks", "8", "--json", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["summary"]["n"] == 4 and res["summary"]["n_equal"] == 4
    assert res["summary"]["lcp_assumed_from_prompt_lengths"] is False
    assert [(r["sid"], r["k"], r["hit"]) for r in res["rows"]] == [("A", 0, 0), ("B", 0, 0),
                                                                   ("A", 1, 192), ("B", 1, 128)]
    assert "A" in capsys.readouterr().out
    # without lcp_prev the prompts are taken to extend each other: same answer here
    for r in recs:
        r.pop("lcp_prev", None)
    events, assumed = L.events_from_trace(recs)
    assert assumed and [r["hit"] for r in L.simulate(events, 8, "fa")[0]] == [0, 0, 192, 128]
