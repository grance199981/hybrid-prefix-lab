"""Helpers shared by several test files: a stand-in tokenizer, synthetic sessions in
the shapes of the three datasets, and synthetic four-pass replay records."""
import json
import os
import random
import re

import pytest

from hplab.records import BLOCKS, DATASETS, THINKS
from hplab.sessions import _norm_messages

ROOT = os.path.join(os.path.dirname(__file__), "..")

# ------------------------------------------------------------ stand-in tokenizer
SPLIT = re.compile(r"<\|[a-z_]+\|>|</?think>|</?tool_call>|</?tool_response>|\n+|\w+|[^\w\s]| +")


class StandInTokenizer:
    """Renders one of the official chat templates in refs/ with jinja2 and splits the
    text with a regex, so prompt rendering can be tested without transformers or model
    files. A run of newlines is one token, as in the Qwen tokenizers."""

    def __init__(self, template_file):
        pytest.importorskip("jinja2")
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(msg):
            raise ValueError(msg)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
        with open(os.path.join(ROOT, "refs", template_file), encoding="utf-8") as f:
            self.template = env.from_string(f.read())
        self.vocab = {}

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                            tokenize=False, enable_thinking=True):
        return self.template.render(messages=messages, tools=tools,
                                    add_generation_prompt=add_generation_prompt,
                                    enable_thinking=enable_thinking)

    def __call__(self, text, add_special_tokens=False):
        ids = [self.vocab.setdefault(t, len(self.vocab)) for t in SPLIT.findall(text)]
        return {"input_ids": ids}


# ------------------------------------------------------------ synthetic sessions
def words(prefix, n):
    return " ".join(f"{prefix}{i}" for i in range(n))


def chat_session(turns=4):
    """User and assistant alternating (the WildChat shape)."""
    msgs = []
    for t in range(turns):
        msgs.append({"role": "user", "content": words(f"q{t}_", 30)})
        msgs.append({"role": "assistant", "content": words(f"a{t}_", 40)})
    return msgs


def struct_session(steps=4):
    """One user query, then assistant tool calls and tool results (the SWE-rebench shape)."""
    raw = [{"role": "system", "content": "You are an agent."},
           {"role": "user", "content": words("task", 50)}]
    for s in range(steps):
        raw.append({"role": "assistant", "content": words(f"plan{s}_", 10) if s % 2 == 0 else None,
                    "tool_calls": [{"id": f"c{s}", "type": "function", "function": {
                        "name": "execute_bash", "arguments": json.dumps({"command": f"ls dir{s}"})}}]})
        raw.append({"role": "tool", "content": words(f"out{s}_", 60), "name": "execute_bash",
                    "tool_call_id": f"c{s}", "tool_calls": None})
    raw.append({"role": "assistant", "content": "done", "tool_calls": None})
    return _norm_messages(raw)


def textobs_session(steps=4):
    """Observations sent back as plain user messages (the SWE-smith shape)."""
    raw = [{"role": "system", "content": "You are an agent."},
           {"role": "user", "content": words("task", 50)}]
    for s in range(steps):
        raw.append({"role": "assistant", "content": words(f"act{s}_", 20)})
        raw.append({"role": "user", "content": "OBSERVATION:\n" + words(f"out{s}_", 60)})
    raw.append({"role": "assistant", "content": "done"})
    return _norm_messages(raw)


TOOLS = [{"type": "function", "function": {"name": "execute_bash", "description": "run",
                                           "parameters": {"type": "object", "properties": {
                                               "command": {"type": "string"}}}}}]


# ------------------------------------------------------- synthetic four-pass records
def four_pass_records(overhead=0.0, drift=(0.0, 0.0, 0.0, 0.0), noise=6.0, sessions=12, turns=6,
                      skip_block=None, saved=200, seed=0):
    """Four passes of the same requests, shaped like hplab.records.load_dir output.
    TTFT = 80 + 0.09 * recompute + noise, plus `overhead` in the two C1 passes and a
    per-pass `drift`. C1 recomputes `saved` fewer tokens on every request after the first."""
    rng = random.Random(seed)
    cells = {}
    for bi, block in enumerate(BLOCKS):
        if block == skip_block:
            continue
        unit16 = block.startswith("C1")
        for ds in DATASETS:
            for th in THINKS:
                rows = {}
                for s in range(sessions):
                    for k in range(turns):
                        P = 2000 + 700 * k + 13 * s
                        rec = P if k == 0 else (500 - saved if unit16 else 500)
                        t = 80 + 0.09 * rec + rng.gauss(0, noise) + drift[bi] + (overhead if unit16 else 0)
                        rows[(1, f"{ds}-{s}", k)] = {
                            "sid": f"{ds}-{s}", "k": k, "prompt_tokens": P,
                            "cached_tokens": P - rec, "ttft_ms": t, "first_chunk_ms": t - 1,
                            "e2e_ms": t + 150, "error": None}
                cells[(block, ds, th)] = rows
    return cells
