"""Unit tests for the pure parts of experiments/e1_prefix_hits/replay.py plus an end-to-end run against
a fake server (aiohttp, in-process) that implements a known cache policy, so
the client's bookkeeping is checked against a server whose answers are known.

The fake server tokenises the rendered prompt with the same stand-in tokenizer
as the client and reports cached_tokens under one of two policies:
  fa16     floor(D / 16) * 16, D = longest prefix shared with anything seen
  nocache  always 0
"""
import asyncio
import json
import os
import sys
import types

import numpy as np
import pytest

from experiments.e1_prefix_hits import replay as e1
from hplab.cache_model import lcp
from hplab.client import parse_metrics
from hplab.sessions import render_session
from helpers import TOOLS, StandInTokenizer, chat_session, struct_session, textobs_session

Q35 = "qwen3.5-4b_chat_template.jinja"
Q3 = "qwen3-4b_chat_template.jinja"


# ------------------------------------------------------------------ pure parts
@pytest.mark.parametrize("session,tools", [(chat_session(), None), (struct_session(), TOOLS),
                                           (textobs_session(), None)])
def test_request_points_match_render_session(session, tools):
    tok = StandInTokenizer(Q35)
    prompts, _, _, truncated = render_session(tok, session, tools, True, 10**9)
    assert not truncated
    pts = e1.request_points(session)
    assert len(pts) == len(prompts)
    assert all(session[i]["role"] == "assistant" for i in pts)


def test_request_points_skip_leading_assistant():
    msgs = [{"role": "assistant", "content": "hi"}, {"role": "system", "content": "s"},
            {"role": "assistant", "content": "x"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"}]
    assert e1.request_points(msgs) == [4]


@pytest.mark.parametrize("template", [Q35, Q3])
@pytest.mark.parametrize("thinking", [True, False])
def test_api_round_trip_renders_identically(template, thinking):
    """API format -> template format must render the same prompt as the E0 messages."""
    tok = StandInTokenizer(template)
    msgs = struct_session()
    api = e1.to_api_messages(msgs)
    assert [m["role"] for m in api] == [m["role"] for m in msgs]
    a = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                enable_thinking=thinking)
    b = tok.apply_chat_template(e1.to_template_messages(api), tools=TOOLS,
                                add_generation_prompt=True, enable_thinking=thinking)
    assert a == b
    calls = [m for m in api if m.get("tool_calls")]
    tool_msgs = [m for m in api if m["role"] == "tool"]
    assert [c["tool_calls"][0]["id"] for c in calls] == [t["tool_call_id"] for t in tool_msgs]
    assert all(isinstance(c["tool_calls"][0]["function"]["arguments"], str) for c in calls)


def test_own_assistant_message_keeps_reasoning_and_calls():
    own = e1.own_assistant_message({"content": None, "reasoning": "think",
                                    "tool_calls": [{"id": "x", "type": "function", "function": {
                                        "name": "execute_bash", "arguments": "{\"command\": \"ls\"}"}}]})
    assert own == {"role": "assistant", "content": "", "reasoning": "think",
                   "tool_calls": [{"id": "x", "type": "function", "function": {
                       "name": "execute_bash", "arguments": "{\"command\": \"ls\"}"}}]}
    tm = e1.to_template_messages([own])[0]
    assert tm["reasoning_content"] == "think"
    assert tm["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert e1.own_assistant_message({"content": "hi"}) == {"role": "assistant", "content": "hi"}


def test_judge_controls_good_and_bad():
    def rec(p, c, err=None):
        return {"prompt_tokens": p, "cached_tokens": c, "error": err}
    assert e1.judge_controls([rec(5000, 0), rec(5000, 4752), rec(5000, 0)]) == []
    assert e1.judge_controls([rec(5000, 0), rec(5000, 4992), rec(5000, 0)]) == []
    assert e1.judge_controls([rec(5000, 16), rec(5000, 4992), rec(5000, 0)])   # stale cache
    assert e1.judge_controls([rec(5000, 0), rec(5000, 0), rec(5000, 0)])        # caching off
    assert e1.judge_controls([rec(5000, 0), rec(5000, 5000), rec(5000, 0)])     # impossible
    assert e1.judge_controls([rec(5000, 0), rec(5000, 4992), rec(5000, 4992)])  # B hit
    assert e1.judge_controls([rec(5000, 0), rec(None, 0, "boom"), rec(5000, 0)])


def test_parse_counters_sums_label_sets():
    text = "\n".join([
        "# HELP vllm:prefix_cache_queries_total x",
        'vllm:prefix_cache_queries_total{engine="0",model_name="m"} 100.0',
        'vllm:prefix_cache_queries_total{engine="1",model_name="m"} 20.0',
        'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 64.0',
        'vllm:prefix_cache_hits_created{engine="0"} 1.7e9',
    ])
    assert parse_metrics(text, e1.COUNTERS, fill_missing=True) == {
        "vllm:prefix_cache_queries": 120.0, "vllm:prefix_cache_hits": 64.0}


# ----------------------------------------------------------------- fake server
aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402


class FakeServer:
    def __init__(self, template, policy):
        self.tok = StandInTokenizer(template)
        self.policy = policy
        self.seen = []
        self.queries = self.hits = 0
        self.resets = 0
        self.silent_every = 0     # > 0: every n-th streamed reply carries no delta text
        self.n_stream = 0

    def render(self, body):
        msgs = e1.to_template_messages(body["messages"])
        text = self.tok.apply_chat_template(
            msgs, tools=body.get("tools"), add_generation_prompt=True,
            enable_thinking=body["chat_template_kwargs"]["enable_thinking"])
        return np.asarray(self.tok(text)["input_ids"], dtype=np.int64)

    async def chat(self, request):
        body = await request.json()
        ids = self.render(body)
        d = max((lcp(ids, s) for s in self.seen), default=0)
        cached = min(d, len(ids) - 1) // 16 * 16 if self.policy == "fa16" else 0
        n_out = body["max_tokens"]
        out_ids = np.arange(10**6, 10**6 + n_out, dtype=np.int64)   # never matches a prompt
        self.seen.append(np.concatenate([ids, out_ids]))
        self.queries += len(ids)
        self.hits += cached
        usage = {"prompt_tokens": len(ids), "completion_tokens": n_out,
                 "prompt_tokens_details": {"cached_tokens": cached} if cached else None}
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            self.n_stream += 1
            silent = self.silent_every and self.n_stream % self.silent_every == 0
            chunks = [{"choices": [{"delta": {"role": "assistant", "content": ""}}]},
                      {"choices": [{"delta": {} if silent else {"content": "x"}}]},
                      {"choices": [{"delta": {}, "finish_reason": "length"}]},
                      {"choices": [], "usage": usage}]
            for c in chunks:
                await resp.write(("data: " + json.dumps(c) + "\n\n").encode())
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return web.json_response({
            "prompt_token_ids": ids.tolist(), "usage": usage,
            "choices": [{"finish_reason": "length", "token_ids": out_ids.tolist(),
                         "message": {"role": "assistant", "content": "own reply " * 3,
                                     "reasoning": "own thought"}}]})

    async def reset(self, request):
        self.seen, self.resets = [], self.resets + 1
        return web.json_response({"success": True})

    async def metrics(self, request):
        return web.Response(text=(
            f'vllm:prefix_cache_queries_total{{engine="0"}} {float(self.queries)}\n'
            f'vllm:prefix_cache_hits_total{{engine="0"}} {float(self.hits)}\n'))

    async def health(self, request):
        return web.Response(text="ok")

    def app(self):
        app = web.Application(client_max_size=64 * 2**20)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_post("/reset_prefix_cache", self.reset)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/health", self.health)
        return app


def run_client(monkeypatch, tmp_path, template, policy, mode, thinking="on", interleave=1,
               sessions=None, keepalive_timeout=75.0, diag=0, silent_every=0):
    sessions = sessions or [("chat-0", chat_session(5), None), ("struct-0", struct_session(5), TOOLS),
                            ("textobs-0", textobs_session(4), None)]
    monkeypatch.setitem(e1.LOADERS, "wildchat", lambda path, n, seed, tools_json=None: iter(sessions))
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda d: StandInTokenizer(template))
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    server = FakeServer(template, policy)
    server.silent_every = silent_every
    out = str(tmp_path / f"{policy}_{mode}.jsonl")

    async def go():
        runner = web.AppRunner(server.app(), keepalive_timeout=keepalive_timeout)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        a = types.SimpleNamespace(
            url=f"http://127.0.0.1:{port}", model="m", tokenizer_dir="x", chat_template=None,
            dataset="wildchat", path="x", tools_json=None, seed=1, n_sessions=3,
            thinking=thinking, mode=mode, max_len=10**6, out_cap=8, interleave=interleave,
            hybrid_block=576, diag=diag,
            label="t", out=out)
        try:
            return await e1.run(a)
        finally:
            await runner.cleanup()

    rc = asyncio.run(go())
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    summary_path = out[:-6] + ".summary.json"
    summary = json.load(open(summary_path, encoding="utf-8")) if os.path.exists(summary_path) else None
    return rc, rows, summary, server


@pytest.mark.parametrize("template", [Q35, Q3])
@pytest.mark.parametrize("thinking", ["on", "off"])
def test_R_replay_matches_known_fa16_server(monkeypatch, tmp_path, template, thinking):
    rc, rows, summary, server = run_client(monkeypatch, tmp_path, template, "fa16", "R", thinking)
    assert rc == 0 and server.resets == 1
    sess = [r for r in rows if r["sid"] != "control"]
    assert len({r["sid"] for r in sess}) == 3
    assert all(r["error"] is None and r["P"] == r["prompt_tokens"] for r in sess)
    # the server implements exactly the fa16_R rule, so every request must agree
    assert all(r["cached_tokens"] == r["fa16_R"] for r in sess)
    assert all(r["ttft_ms"] is not None and r["n_out"] <= 8 for r in sess)
    scored = summary["n_requests_scored"]
    assert summary["pred_fa16_R"]["n_equal"] == scored and summary["n_P_mismatch"] == 0
    mc = summary["metrics_check"]
    assert mc["queries_delta"] == mc["sum_prompt_tokens"] and mc["hits_delta"] == mc["sum_cached_tokens"]


def test_control_stops_the_run_when_the_server_does_not_cache(monkeypatch, tmp_path):
    rc, rows, summary, _ = run_client(monkeypatch, tmp_path, Q35, "nocache", "R")
    assert rc == 4 and summary is None
    assert [r["control"] for r in rows] == ["A", "A_again", "B"]


def test_interleaved_groups_give_the_same_per_session_account(monkeypatch, tmp_path):
    _, rows1, _, _ = run_client(monkeypatch, tmp_path, Q35, "fa16", "R", interleave=1)
    _, rows3, _, _ = run_client(monkeypatch, tmp_path, Q35, "fa16", "R", interleave=3)
    key = lambda rows: sorted((r["sid"], r["k"], r["prompt_tokens"], r["fa16_R"])  # noqa: E731
                              for r in rows if r["sid"] != "control")
    assert key(rows1) == key(rows3)


@pytest.mark.parametrize("thinking", ["on", "off"])
def test_O_replay_carries_own_output_and_accounts_from_token_ids(monkeypatch, tmp_path, thinking):
    rc, rows, summary, server = run_client(monkeypatch, tmp_path, Q35, "fa16", "O", thinking)
    assert rc == 0
    sess = [r for r in rows if r["sid"] != "control"]
    assert all(r["error"] is None for r in sess)
    # local render of the carried-back history equals what the server tokenised
    assert all(r["P_local"] == r["prompt_tokens"] == r["P"] for r in sess)
    assert summary["n_P_local_mismatch"] == 0
    # the fake server's output ids never reappear in a prompt, so nothing of the
    # reply is reusable: D_I == D_R, and the fa16 server agrees with fa16_I
    later = [r for r in sess if r["k"] > 0]
    assert later and all(r["D_I"] == r["D_R"] <= r["P_prev"] for r in later)
    assert all(r["cached_tokens"] == r["fa16_I"] for r in sess)
    assert all(r["G_prev"] == 8 for r in later)
    # the history really is the server's reply, not the dataset's
    chat_rows = [r for r in later if r["sid"] == "chat-0"]
    assert all(r["own_has_reasoning"] for r in sess) and chat_rows


def test_idle_connections_closed_by_the_server_do_not_fail_requests(monkeypatch, tmp_path):
    """While the client renders the next group of sessions (blocking, no event-loop
    turn), the server closes its idle keep-alive connections; reusing a pooled
    connection then fails with ServerDisconnectedError. Reproduced here with a short
    server keep-alive and a blocking pause."""
    import time as _time
    orig = e1.SessionRun.prepare_R

    def slow_prepare(self):
        _time.sleep(0.4)
        return orig(self)

    monkeypatch.setattr(e1.SessionRun, "prepare_R", slow_prepare)
    rc, rows, summary, _ = run_client(monkeypatch, tmp_path, Q35, "fa16", "R", interleave=1,
                                      keepalive_timeout=0.05)
    assert rc == 0
    assert [r for r in rows if r.get("error")] == []
    assert summary["n_errors"] == 0 and summary["sessions_stopped_request_error"] == 0


def test_first_chunk_time_exists_when_the_stream_carries_no_delta(monkeypatch, tmp_path):
    rc, rows, summary, server = run_client(monkeypatch, tmp_path, Q35, "fa16", "R", silent_every=3)
    sess = [r for r in rows if r["sid"] != "control"]
    silent = [r for r in sess if r["ttft_ms"] is None]
    assert rc == 0 and silent and len(silent) < len(sess)
    assert all(r["first_chunk_ms"] is not None and r["error"] is None for r in sess)
    assert all("content" not in r["delta_keys"] for r in silent)
    assert all("content" in r["delta_keys"] for r in sess if r["ttft_ms"] is not None)


def test_diag_resends_both_kinds_and_metrics_still_reconcile(monkeypatch, tmp_path):
    rc, rows, summary, server = run_client(monkeypatch, tmp_path, Q35, "fa16", "R",
                                           diag=2, silent_every=3)
    sess = [r for r in rows if r["sid"] != "control"]
    diag = [r for r in sess if r.get("diag_kind")]
    assert rc == 0
    assert sum(r["diag_kind"] == "no_delta" for r in diag) == 2
    assert sum(r["diag_kind"] == "with_delta" for r in diag) == 2
    assert all(r["diag_content_head"].startswith("own reply") for r in diag)
    assert all(r["diag_n_token_ids"] == r["n_out"] for r in diag)   # the generated ids came back
    # a diagnosed request is a no-delta one exactly when its stream had no TTFT
    assert all((r["ttft_ms"] is None) == (r["diag_kind"] == "no_delta") for r in diag)
    mc = summary["metrics_check"]
    assert mc["queries_delta"] == mc["sum_prompt_tokens"] and mc["hits_delta"] == mc["sum_cached_tokens"]
    # the re-sends do not change what the sessions themselves hit
    assert all(r["cached_tokens"] == r["fa16_R"] for r in sess)
