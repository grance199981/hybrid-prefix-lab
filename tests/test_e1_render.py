"""Checks hplab.sessions.render_session against the two official chat templates in
refs/, using a stand-in tokenizer (jinja2 + a regex splitter), so it runs without
transformers or any model files. The stand-in treats a run of newlines as one token,
like the Qwen tokenizers do for "\\n\\n"; the real token counts are checked by the
offline accounting run itself.

The synthetic sessions mimic the three dataset shapes:
  chat      user / assistant alternating (WildChat)
  struct    one user query, then assistant tool_calls + tool results (SWE-rebench)
  textobs   observations sent back as plain user messages (SWE-smith shard 4)
"""
import pytest

from helpers import TOOLS, StandInTokenizer, chat_session, struct_session, textobs_session
from hplab.cache_model import predict_hits
from hplab.sessions import render_session

pytest.importorskip("jinja2")


def gaps(template_file, messages, tools, thinking):
    tok = StandInTokenizer(template_file)
    prompts, ideals, flags, truncated = render_session(tok, messages, tools, thinking, 10**9)
    assert not truncated and len(prompts) >= 2
    rows = predict_hits(prompts, ideals)
    return [r["g_obs"] for r in rows[1:]], rows, flags


@pytest.mark.parametrize("template_file,thinking,expected", [
    ("qwen3.5-4b_chat_template.jinja", True, 2),
    ("qwen3.5-4b_chat_template.jinja", False, 4),
    ("qwen3-4b_chat_template.jinja", True, 0),
    ("qwen3-4b_chat_template.jinja", False, 4),
])
def test_chat_gap_matches_the_template(template_file, thinking, expected):
    g, rows, flags = gaps(template_file, chat_session(), None, thinking)
    assert g == [expected] * len(g)
    assert all(f["after_user_query"] for f in flags)
    # the previous reply is never inside the shared prefix
    assert all(r["D_R"] < r["P_prev"] + 1 for r in rows[1:])


@pytest.mark.parametrize("template_file", ["qwen3.5-4b_chat_template.jinja",
                                           "qwen3-4b_chat_template.jinja"])
def test_text_observations_behave_like_user_turns(template_file):
    g_chat, _, _ = gaps(template_file, chat_session(), None, True)
    g_obs, _, flags = gaps(template_file, textobs_session(), None, True)
    assert set(g_obs) == set(g_chat)
    assert all(f["after_user_query"] for f in flags)


@pytest.mark.parametrize("template_file,thinking,expected_gap,reply_reusable", [
    # History carries no reasoning_content (none of the datasets has it).
    # Qwen3.5, thinking on: the empty think block renders "<think>\n\n</think>"; the
    # stand-in merges "\n\n" into one token, so the prefix ends one token early.
    ("qwen3.5-4b_chat_template.jinja", True, 1, False),
    ("qwen3.5-4b_chat_template.jinja", False, 0, True),
    # Qwen3: a non-final assistant turn without reasoning_content is rendered with
    # no think block at all (template line 44), so with thinking off the four
    # generation-prompt tokens are lost inside a chain too.
    ("qwen3-4b_chat_template.jinja", True, 0, False),
    ("qwen3-4b_chat_template.jinja", False, 4, False),
])
def test_structured_chain_gap(template_file, thinking, expected_gap, reply_reusable):
    g, rows, flags = gaps(template_file, struct_session(), TOOLS, thinking)
    assert [f["after_user_query"] for f in flags] == [True] + [False] * (len(flags) - 1)
    assert g == [expected_gap] * len(g)
    assert all((r["D_I"] > r["P_prev"]) == reply_reusable for r in rows[1:])


def test_negative_control_shuffled_history_breaks_reuse():
    tok = StandInTokenizer("qwen3.5-4b_chat_template.jinja")
    a = chat_session()
    b = [dict(m, content=m["content"].replace("q0_", "zz_")) for m in a]  # first user turn differs
    pa, _, _, _ = render_session(tok, a, None, True, 10**9)
    pb, _, _, _ = render_session(tok, b, None, True, 10**9)
    rows = predict_hits([pa[2], pb[3]], [None, None])
    assert rows[1]["D_R"] < 10 and rows[1]["hy16"] == 0 and rows[1]["fa16_R"] == 0
