"""Multi-turn sessions from the three datasets, and how they turn into prompts.

Datasets (all read from parquet, sampled with a fixed seed):
  wildchat  WildChat-1M, human-chatbot conversations with at least 3 turns
  smith     SWE-smith trajectories, coding-agent sessions with tool calls
  rebench   SWE-rebench OpenHands trajectories, coding-agent sessions; needs tools.json

Only the role, the text and the tool calls of each message are read. WildChat rows
also carry user metadata (hashed IP, location, request headers); those columns are
never loaded.

A session becomes one request per assistant message: the prompt is the chat
template applied to everything before that message.
"""
import json
import random

import numpy as np

from hplab.cache_model import lcp

END_OF_TURN = "<|im_end|>\n"


# ------------------------------------------------------------------- loading
def _sample_rows(path, columns, n, seed, keep=None, filter_columns=None):
    """Sample n rows from a parquet file, reading only the row groups needed."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    eligible = list(range(total))
    if keep is not None:
        meta = pf.read(columns=filter_columns).to_pylist()
        eligible = keep(meta)
    rng = random.Random(seed)
    picked = sorted(rng.sample(eligible, min(n, len(eligible))))
    bounds, start = [], 0
    for g in range(pf.metadata.num_row_groups):
        m = pf.metadata.row_group(g).num_rows
        bounds.append((start, start + m))
        start += m
    out = []
    for g, (lo, hi) in enumerate(bounds):
        idx = [i for i in picked if lo <= i < hi]
        if not idx:
            continue
        tab = pf.read_row_group(g, columns=columns)
        for i in idx:
            out.append((i, {c: tab.column(c)[i - lo].as_py() for c in columns}))
    return out, len(eligible), total


def _flatten(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _norm_tool_calls(tcs):
    out = []
    for tc in tcs or []:
        fn = dict(tc.get("function") or {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {"value": args}
        out.append({"type": "function", "function": {"name": fn.get("name"), "arguments": args}})
    return out


def _norm_messages(raw):
    msgs = []
    for m in raw:
        role = m.get("role")
        msg = {"role": role, "content": _flatten(m.get("content"))}
        if role == "assistant":
            tcs = _norm_tool_calls(m.get("tool_calls"))
            if tcs:
                msg["tool_calls"] = tcs
        msgs.append(msg)
    return msgs


def load_rebench(path, n, seed, tools_json=None):
    if not tools_json:
        raise SystemExit("rebench needs --tools-json (tools.json from the dataset repo)")
    with open(tools_json, encoding="utf-8") as f:
        tools = json.load(f)
    rows, _, _ = _sample_rows(path, ["trajectory"], n, seed)
    for i, r in rows:
        yield f"rebench-{i}", _norm_messages(r["trajectory"]), tools


def load_smith(path, n, seed, tools_json=None):
    rows, _, _ = _sample_rows(path, ["messages"], n, seed)
    for i, r in rows:
        raw = r["messages"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        yield f"smith-{i}", _norm_messages(raw), None


def load_wildchat(path, n, seed, tools_json=None, min_turns=3):
    def keep(meta):
        seen, out = set(), []
        for i, m in enumerate(meta):
            if (m["turn"] or 0) >= min_turns and m["conversation_hash"] not in seen:
                seen.add(m["conversation_hash"])
                out.append(i)
        return out
    rows, _, _ = _sample_rows(path, ["conversation"], n, seed, keep=keep,
                              filter_columns=["turn", "conversation_hash"])
    for i, r in rows:
        msgs = [{"role": m["role"], "content": m["content"] or ""} for m in r["conversation"]]
        if any(m["role"] == "user" and not m["content"].strip() for m in msgs):
            continue
        yield f"wildchat-{i}", msgs, None


LOADERS = {"rebench": load_rebench, "smith": load_smith, "wildchat": load_wildchat}


# ----------------------------------------------------------------- rendering
def render_session(tok, messages, tools, thinking, max_len):
    """Render every request of one session with the model's chat template.

    Returns (prompts, ideals, flags, truncated):
      prompts[k]  token ids of request k (one request per assistant message)
      ideals[k]   prompt k followed by the re-rendered reply k, or None when the
                  re-rendered history does not extend the prompt
      flags[k]    after_user_query, ideal_extends_prompt
      truncated   a prompt longer than max_len ended the session early
    """
    prompts, ideals, flags = [], [], []
    truncated = False
    kw = {"tokenize": False, "enable_thinking": thinking}
    if tools:
        kw["tools"] = tools
    for i, m in enumerate(messages):
        if m["role"] != "assistant" or i == 0:
            continue
        if not any(x["role"] == "user" for x in messages[:i]):
            continue
        text = tok.apply_chat_template(messages[:i], add_generation_prompt=True, **kw)
        ids = np.asarray(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
        if len(ids) > max_len:
            truncated = True
            break
        full = tok.apply_chat_template(messages[:i + 1], add_generation_prompt=False, **kw)
        if full.endswith(END_OF_TURN):
            full = full[:-len(END_OF_TURN)]
        ideal = np.asarray(tok(full, add_special_tokens=False)["input_ids"], dtype=np.int64)
        prompts.append(ids)
        ideals.append(ideal if lcp(ids, ideal) == len(ids) else None)
        flags.append({"after_user_query": messages[i - 1]["role"] == "user",
                      "ideal_extends_prompt": ideals[-1] is not None})
    return prompts, ideals, flags, truncated
