"""Replay multi-turn sessions against a running vLLM server and record, per request,
what the server reused next to what the offline accounting (account.py) predicts.

Two replay modes:
  R  recorded replay: the history is the dataset's messages. Each request
     generates as many tokens as the recorded reply had (capped by --out-cap,
     min_tokens = max_tokens) and the output is dropped. Streamed, so TTFT is
     measured.
  O  own-output replay: every assistant message in the history is what the
     server returned for the earlier request (content, reasoning, tool calls as
     parsed by the server); user and tool messages stay the dataset's. Not
     streamed; prompt and output token ids are asked back from the server.

Sessions are sampled and normalised by hplab.sessions, exactly as in account.py.
In R mode the prompts are rendered locally first, so every record carries the
prediction (P, D_R, fa16_R, hy<block>, hy16, ...) next to the server's
prompt_tokens / cached_tokens. In O mode the same accounting is done afterwards
from the token ids the server returns ("ideal" = prompt + generated ids).

Before the sessions, on the same code path:
  - the prefix cache is reset (needs VLLM_SERVER_DEV_MODE=1 on the server);
  - three control requests: A (must miss), A again (must hit), unrelated B
    (must miss). A failing control stops the run with exit code 4.
/metrics counters are read before and after and compared with the per-request sums.

Output: <out>.jsonl (the three control records, then one record per request)
and <out>.summary.json (measured vs predicted, counts, /metrics reconciliation).
Exit codes: 0 ok, 4 control failed, 5 cache reset failed, 6 server unreachable.

Usage:
  python experiments/e1_prefix_hits/replay.py --url http://127.0.0.1:8061 --model qwen35 \
      --tokenizer-dir <dir> [--chat-template <jinja>] \
      --dataset wildchat --path <parquet> [--tools-json <json>] \
      --seed 1 --n-sessions 20 --thinking on --mode R --out <file.jsonl>
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time

import numpy as np

from hplab.cache_model import FA_BLOCK, policy_names, predict_hits
from hplab.client import (post_full, post_stream, read_counters_async,
                          reset_prefix_cache_async, usage_fields)
from hplab.prompts import control_words
from hplab.sessions import LOADERS, render_session

NL = chr(10)
COUNTERS = ["vllm:prefix_cache_queries", "vllm:prefix_cache_hits"]


# ------------------------------------------------------------ message shaping
def request_points(messages):
    """Indices of the assistant messages that are answered by a request.
    Must stay in step with the loop in hplab.sessions.render_session."""
    pts, seen_user = [], False
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and i > 0 and seen_user:
            pts.append(i)
        if m["role"] == "user":
            seen_user = True
    return pts


def to_api_messages(messages):
    """Normalised messages -> OpenAI chat format, index-aligned.
    Tool-call arguments become JSON strings; ids are made up (the Qwen
    templates do not render them) and handed to the tool messages in order."""
    out, pending, n = [], [], 0
    for m in messages:
        msg = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "assistant" and m.get("tool_calls"):
            pending, calls = [], []
            for tc in m["tool_calls"]:
                cid = f"call_{n}"
                n += 1
                pending.append(cid)
                fn = tc["function"]
                calls.append({"id": cid, "type": "function", "function": {
                    "name": fn.get("name"),
                    "arguments": json.dumps(fn.get("arguments") or {}, ensure_ascii=False)}})
            msg["tool_calls"] = calls
        elif m["role"] == "tool":
            if pending:
                msg["tool_call_id"] = pending.pop(0)
            else:
                msg["tool_call_id"] = f"call_{n}"
                n += 1
        out.append(msg)
    return out


def to_template_messages(api_messages):
    """OpenAI chat format -> what the chat template is rendered with locally:
    arguments as dicts, `reasoning` also under `reasoning_content` (vLLM passes
    both names to the template, renderers/chat_utils/chat_message.py)."""
    out = []
    for m in api_messages:
        msg = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                if not isinstance(args, dict):
                    args = {"value": args}
                calls.append({"type": "function",
                              "function": {"name": fn.get("name"), "arguments": args}})
            msg["tool_calls"] = calls
        reasoning = m.get("reasoning") or m.get("reasoning_content")
        if reasoning:
            msg["reasoning_content"] = reasoning
        out.append(msg)
    return out


def own_assistant_message(resp_message):
    """The assistant message a client would carry back from a response."""
    msg = {"role": "assistant", "content": resp_message.get("content") or ""}
    reasoning = resp_message.get("reasoning") or resp_message.get("reasoning_content")
    if reasoning:
        msg["reasoning"] = reasoning
    calls = []
    for tc in resp_message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append({"id": tc.get("id") or f"own_{len(calls)}", "type": "function",
                      "function": {"name": fn.get("name"),
                                   "arguments": fn.get("arguments") or "{}"}})
    if calls:
        msg["tool_calls"] = calls
    return msg


def recorded_out_len(tok, message):
    """Token count of a recorded assistant message (content plus tool calls)."""
    text = message.get("content") or ""
    for tc in message.get("tool_calls") or []:
        text += NL + json.dumps(tc.get("function") or {}, ensure_ascii=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def render_prompt_len(tok, api_messages, tools, thinking):
    kw = {"tokenize": False, "enable_thinking": thinking}
    if tools:
        kw["tools"] = tools
    text = tok.apply_chat_template(to_template_messages(api_messages),
                                   add_generation_prompt=True, **kw)
    return len(tok(text, add_special_tokens=False)["input_ids"])


# ------------------------------------------------------------------ controls
def judge_controls(rows, min_block=576):
    """rows: the three control records in order (A, A again, B).
    The resent A must hit, within one hybrid block (plus 16) of its length; the
    first A and the unrelated B must not hit at all."""
    a, a2, b = rows
    problems = []
    if any(r.get("error") for r in rows):
        problems.append("request error")
        return problems
    if a["cached_tokens"] != 0:
        problems.append(f"first A hit {a['cached_tokens']} tokens after a cache reset")
    if not (a2["prompt_tokens"] - min_block - 16 <= a2["cached_tokens"] < a2["prompt_tokens"]):
        problems.append(f"resent A hit {a2['cached_tokens']} of {a2['prompt_tokens']}")
    if b["cached_tokens"] != 0:
        problems.append(f"unrelated B hit {b['cached_tokens']} tokens")
    return problems


def summarize_run(rows, policies):
    """Measured vs predicted over all requests with k > 0 (first requests excluded)."""
    rs = [r for r in rows if r["k"] > 0 and not r.get("error") and r.get("prompt_tokens")]
    out = {"n_requests_total": len(rows), "n_requests_scored": len(rs),
           "n_errors": sum(1 for r in rows if r.get("error")),
           "n_sessions": len({r["sid"] for r in rows})}
    if not rs:
        return out
    tot = sum(r["prompt_tokens"] for r in rs)
    rec = [r["prompt_tokens"] - r["cached_tokens"] for r in rs]
    out["prompt_tokens_sum"] = tot
    out["measured_hit_rate"] = sum(r["cached_tokens"] for r in rs) / tot
    out["measured_recompute_median"] = statistics.median(rec)
    out["n_P_mismatch"] = sum(1 for r in rs if r["P"] != r["prompt_tokens"])
    if any("P_local" in r for r in rs):     # O mode: local render vs the server's prompt
        out["n_P_local_mismatch"] = sum(1 for r in rs if r.get("P_local") != r["prompt_tokens"])
    for pol in policies:
        out[f"pred_{pol}"] = {
            "hit_rate": sum(r[pol] for r in rs) / sum(r["P"] for r in rs),
            "recompute_median": statistics.median(r["P"] - r[pol] for r in rs),
            "n_equal": sum(1 for r in rs if r[pol] == r["cached_tokens"]),
            "n_within16": sum(1 for r in rs if abs(r[pol] - r["cached_tokens"]) <= 16),
        }
    tt = [r["ttft_ms"] for r in rs if r.get("ttft_ms") is not None]
    if tt:
        out["ttft_ms_median"] = statistics.median(tt)
    out["e2e_ms_median"] = statistics.median(r["e2e_ms"] for r in rs)
    out["completion_tokens_sum"] = sum(r.get("completion_tokens") or 0 for r in rs)
    return out


# -------------------------------------------------------------------- session
DIAG_SEEN = {"no_delta": 0, "with_delta": 0}
EXTRA_SENT = []     # usage of the diagnostic re-sends, for the /metrics reconciliation


class SessionRun:
    """One session: its requests, sent in order, and the records they produce."""

    def __init__(self, sid, messages, tools, a, tok):
        self.sid, self.messages, self.tools, self.a, self.tok = sid, messages, tools, a, tok
        self.api = to_api_messages(messages)
        self.points = request_points(messages)
        self.thinking = a.thinking == "on"
        self.meas = []          # one dict per request sent
        self.rows = []          # final records
        self.stopped = None     # why the session ended early, if it did
        self.render_error = None
        self.n_req = 0
        # O mode state
        self.hist = None
        self.prompt_ids, self.full_ids = [], []

    def base_body(self, messages, max_tokens):
        body = {"model": self.a.model, "messages": messages, "temperature": 0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        if self.tools:
            body["tools"] = self.tools
        return body

    # ---- R mode
    def prepare_R(self):
        a = self.a
        try:
            prompts, ideals, flags, truncated = render_session(
                self.tok, self.messages, self.tools, self.thinking, a.max_len)
        except Exception as e:  # counted, not hidden
            self.render_error = f"{type(e).__name__}: {str(e)[:120]}"
            return
        self.truncated = truncated
        if len(prompts) < 2:
            return
        assert len(prompts) <= len(self.points), "request points out of step with render_session"
        if not truncated:
            assert len(prompts) == len(self.points), "request points out of step with render_session"
        self.pred = predict_hits(prompts, ideals, aligns=(a.hybrid_block, FA_BLOCK))
        for r, fl in zip(self.pred, flags):
            r.update(fl)
            r.setdefault("g_obs", None)
        self.n_out = []
        for k, p in enumerate(prompts):
            n = min(recorded_out_len(self.tok, self.messages[self.points[k]]),
                    a.out_cap, a.max_len - len(p))
            if n < 1:           # no room left to generate: the session ends here
                self.stopped = "no_output_room"
                break
            self.n_out.append(n)
        self.n_req = len(self.n_out)
        if self.n_req < 2:
            self.n_req = 0

    async def step_R(self, http, k):
        body = self.base_body(self.api[:self.points[k]], self.n_out[k])
        body.update({"min_tokens": self.n_out[k], "stream": True,
                     "stream_options": {"include_usage": True}})
        rec = {"n_out": self.n_out[k], "t_start": round(time.time(), 3)}
        try:
            r = await post_stream(http, self.a.url, body)
            rec.update(usage_fields(r["usage"]))
            rec.update({"ttft_ms": r["ttft_ms"], "first_chunk_ms": r["first_chunk_ms"],
                        "delta_keys": r["delta_keys"], "e2e_ms": r["e2e_ms"],
                        "finish_reason": r["finish_reason"], "error": None})
            await self.diagnose(http, body, rec)
        except Exception as e:  # noqa: BLE001 - recorded; the session stops
            rec["error"] = repr(e)[:300]
            self.stopped = "request_error"
            self.n_req = k + 1
        self.meas.append(rec)

    async def diagnose(self, http, body, rec):
        """--diag-no-delta N: for the first N streamed requests that produced no
        delta, and the first N that did, send the same request again without
        streaming and keep the head of what the model wrote. Diagnostic runs
        only; the extra requests are counted in the /metrics reconciliation."""
        limit = getattr(self.a, "diag", 0)
        kind = "no_delta" if rec["ttft_ms"] is None else "with_delta"
        if not limit or DIAG_SEEN[kind] >= limit:
            return
        DIAG_SEEN[kind] += 1
        again = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
        again["return_token_ids"] = True     # to see what the parser removed
        obj = await post_full(http, self.a.url, again)
        msg = obj["choices"][0]["message"]
        content = msg.get("content") or ""
        extra = usage_fields(obj.get("usage") or {})
        EXTRA_SENT.append(extra)
        ids = obj["choices"][0].get("token_ids")
        decode = getattr(self.tok, "decode", None)
        if ids is not None:
            rec["diag_n_token_ids"] = len(ids)
            if decode is not None:
                rec["diag_raw_text"] = decode(ids, skip_special_tokens=False)[:120]
        rec.update({"diag_kind": kind, "diag_content_head": content[:60],
                    "diag_content_len": len(content),
                    "diag_tool_calls": len(msg.get("tool_calls") or []),
                    "diag_reasoning_len": len(msg.get("reasoning")
                                              or msg.get("reasoning_content") or ""),
                    "diag_cached_tokens": extra["cached_tokens"]})

    def finish_R(self):
        for k, rec in enumerate(self.meas):
            row = dict(self.pred[k])
            row.update(rec)
            row["sid"] = self.sid
            self.rows.append(row)

    # ---- O mode
    def prepare_O(self):
        self.truncated = False
        if len(self.points) < 2:
            return
        self.hist = list(self.api[:self.points[0]])
        self.n_req = len(self.points)

    async def step_O(self, http, k):
        a = self.a
        try:
            p_local = render_prompt_len(self.tok, self.hist, self.tools, self.thinking)
        except Exception as e:
            self.render_error = f"{type(e).__name__}: {str(e)[:120]}"
            self.n_req = k
            return
        room = a.max_len - p_local
        if room < 1:
            self.truncated = True
            self.n_req = k
            return
        n_out = min(a.out_cap, room)
        body = self.base_body(self.hist, n_out)
        body["return_token_ids"] = True
        rec = {"n_out": n_out, "P_local": p_local, "t_start": round(time.time(), 3),
               "after_user_query": self.messages[self.points[k] - 1]["role"] == "user"}
        try:
            obj = await post_full(http, a.url, body)
            choice = obj["choices"][0]
            p_ids = obj.get("prompt_token_ids") or choice.get("prompt_token_ids")
            o_ids = choice.get("token_ids")
            if p_ids is None or o_ids is None:
                raise RuntimeError("server returned no token ids")
            own = own_assistant_message(choice["message"])
            rec.update(usage_fields(obj.get("usage") or {}))
            rec.update({"ttft_ms": None, "e2e_ms": obj["_e2e_ms"],
                        "finish_reason": choice.get("finish_reason"), "error": None,
                        "own_tool_calls": len(own.get("tool_calls") or []),
                        "own_has_reasoning": bool(own.get("reasoning")),
                        "recorded_tool_calls": len(self.api[self.points[k]].get("tool_calls") or [])})
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)[:300]
            self.stopped = "request_error"
            self.meas.append(rec)
            self.n_req = k + 1
            return
        self.meas.append(rec)
        p_ids = np.asarray(p_ids, dtype=np.int64)
        self.prompt_ids.append(p_ids)
        self.full_ids.append(np.concatenate([p_ids, np.asarray(o_ids, dtype=np.int64)]))
        nxt = self.points[k + 1] if k + 1 < len(self.points) else len(self.api)
        self.hist = self.hist + [own] + self.api[self.points[k] + 1:nxt]

    def finish_O(self):
        pred = predict_hits(self.prompt_ids, self.full_ids,
                            aligns=(self.a.hybrid_block, FA_BLOCK))
        for k, rec in enumerate(self.meas):
            if k < len(pred):
                row = dict(pred[k])
                row.setdefault("g_obs", None)
                if k > 0:
                    row["G_prev"] = int(len(self.full_ids[k - 1]) - len(self.prompt_ids[k - 1]))
            else:               # the failed request has no token ids
                row = {"k": k, "P": None}
            row.update(rec)
            row["sid"] = self.sid
            self.rows.append(row)


async def run_group(http, group, mode):
    """Round-robin over the group's sessions: every session's request k is in
    flight before any session's request k + 1."""
    k = 0
    while True:
        live = [s for s in group if k < s.n_req]
        if not live:
            break
        if mode == "R":
            await asyncio.gather(*(s.step_R(http, k) for s in live))
        else:
            await asyncio.gather(*(s.step_O(http, k) for s in live))
        k += 1
    for s in group:
        if s.meas:
            s.finish_R() if mode == "R" else s.finish_O()


async def run_controls(http, a):
    rows = []
    texts = [control_words(a.seed, a.label + "-A"), control_words(a.seed, a.label + "-A"),
             control_words(a.seed, a.label + "-B")]
    for name, text in zip(("A", "A_again", "B"), texts):
        body = {"model": a.model, "messages": [{"role": "user", "content": text}],
                "temperature": 0, "max_tokens": 8, "min_tokens": 8, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": a.thinking == "on"}}
        rec = {"sid": "control", "control": name, "k": 0}
        try:
            r = await post_stream(http, a.url, body)
            rec.update(usage_fields(r["usage"]))
            rec.update({"ttft_ms": r["ttft_ms"], "e2e_ms": r["e2e_ms"], "error": None})
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)[:300]
        rows.append(rec)
    return rows


async def run(a):
    import aiohttp
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    if a.chat_template:
        with open(a.chat_template, encoding="utf-8") as f:
            tok.chat_template = f.read()
    sessions = list(LOADERS[a.dataset](a.path, a.n_sessions, a.seed, tools_json=a.tools_json))
    random.Random(a.seed).shuffle(sessions)      # the seed also fixes the send order
    print(f"{a.label}: loaded {len(sessions)} sessions", flush=True)

    timeout = aiohttp.ClientTimeout(total=3600)
    # One connection per request. Local rendering between groups blocks the event
    # loop for seconds; the server closes idle keep-alive connections meanwhile, and
    # reusing one fails with ServerDisconnectedError.
    conn = aiohttp.TCPConnector(limit=max(a.interleave, 1) + 2, force_close=True)
    t_run = time.time()
    counts = {"sessions_loaded": len(sessions), "sessions_used": 0, "sessions_truncated": 0,
              "sessions_render_error": 0, "sessions_stopped_no_output_room": 0,
              "sessions_stopped_request_error": 0}
    all_rows = []
    DIAG_SEEN.update(no_delta=0, with_delta=0)
    del EXTRA_SENT[:]
    async with aiohttp.ClientSession(timeout=timeout, connector=conn, trust_env=False) as http:
        try:
            async with http.get(a.url + "/health") as resp:
                if resp.status != 200:
                    raise RuntimeError(f"health {resp.status}")
        except Exception as e:  # noqa: BLE001
            print(f"server unreachable: {e!r}", flush=True)
            return 6
        if not await reset_prefix_cache_async(http, a.url):
            print("prefix cache reset failed", flush=True)
            return 5
        before = await read_counters_async(http, a.url, COUNTERS)
        controls = await run_controls(http, a)
        problems = judge_controls(controls, min_block=a.hybrid_block)
        with open(a.out, "w", encoding="utf-8", newline=NL) as f:
            for r in controls:
                f.write(json.dumps(r) + NL)
            if problems:
                print("CONTROL FAILED: " + "; ".join(problems), flush=True)
                return 4
            print("controls ok: " + json.dumps(
                [(r["control"], r["prompt_tokens"], r["cached_tokens"]) for r in controls]),
                flush=True)
            for g0 in range(0, len(sessions), a.interleave):
                group = [SessionRun(sid, m, t, a, tok)
                         for sid, m, t in sessions[g0:g0 + a.interleave]]
                for s in group:
                    s.prepare_R() if a.mode == "R" else s.prepare_O()
                await run_group(http, group, a.mode)
                for s in group:
                    counts["sessions_truncated"] += bool(getattr(s, "truncated", False))
                    counts["sessions_render_error"] += s.render_error is not None
                    if s.stopped:
                        counts["sessions_stopped_" + s.stopped] += 1
                    if s.rows:
                        counts["sessions_used"] += 1
                    for r in s.rows:
                        f.write(json.dumps(r) + NL)
                    all_rows.extend(s.rows)
                f.flush()
        after = await read_counters_async(http, a.url, COUNTERS)

    summary = summarize_run(all_rows, policy_names((a.hybrid_block, FA_BLOCK)))
    summary.update(counts)
    sent = [r for r in all_rows + controls + EXTRA_SENT if r.get("prompt_tokens")]
    summary["metrics_check"] = {
        "queries_delta": after[COUNTERS[0]] - before[COUNTERS[0]],
        "hits_delta": after[COUNTERS[1]] - before[COUNTERS[1]],
        "sum_prompt_tokens": sum(r["prompt_tokens"] for r in sent),
        "sum_cached_tokens": sum(r["cached_tokens"] for r in sent),
    }
    summary.update({"label": a.label, "args": vars(a), "controls": controls,
                    "wall_seconds": time.time() - t_run})
    with open(os.path.splitext(a.out)[0] + ".summary.json", "w", encoding="utf-8",
              newline=NL) as f:
        json.dump(summary, f, indent=1)
    keys = ("n_sessions", "n_requests_scored", "n_errors", "n_P_mismatch",
            "measured_hit_rate", "measured_recompute_median", "wall_seconds")
    print(a.label, json.dumps({k: summary.get(k) for k in keys}), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--chat-template", default=None,
                    help="template file the server was started with, if not the model's own")
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--path", required=True)
    ap.add_argument("--tools-json", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-sessions", type=int, required=True)
    ap.add_argument("--thinking", required=True, choices=["on", "off"])
    ap.add_argument("--mode", required=True, choices=["R", "O"])
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--out-cap", type=int, default=64,
                    help="R: cap on the recorded reply length; O: max_tokens")
    ap.add_argument("--hybrid-block", type=int, default=576,
                    help="block size the hybrid model's server reports at startup")
    ap.add_argument("--interleave", type=int, default=1,
                    help="sessions replayed round-robin together (1 = one at a time)")
    ap.add_argument("--diag-no-delta", dest="diag", type=int, default=0,
                    help="diagnostic run: re-send up to N no-delta and N with-delta "
                         "requests without streaming and record what the model wrote")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.label is None:
        a.label = os.path.splitext(os.path.basename(a.out))[0]
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
