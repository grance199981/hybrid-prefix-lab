"""Sessions for the cache-pressure replay, and what an unlimited cache would hit.

A session is {"sid": str, "prompts": [token ids of request 0, request 1, ...]}.

  synth_sessions   controlled sessions: every prompt extends the previous one of
                   its session by `step` tokens; different sessions share nothing.
  real_sessions    recorded sessions (SWE-smith, WildChat), sampled, shuffled and
                   rendered the same way as the recorded replay of E1.

infinite_cache_prediction gives, per request, the hit with an unlimited cache
and one session at a time. It is the denominator of "retention" in
pressure_judge.py and the reference that lru_sim.py must reproduce when its pool
is large enough.

Imported by pressure_replay.py, plan_pools.py and the tests; not run on its own.
"""
import random

import numpy as np

from hplab.cache_model import lcp, predict_hits
from hplab.sessions import LOADERS, render_session

FA_BLOCK = 64       # full-attention block on vLLM 1479deb0
HY_BLOCK = 576      # Qwen3.5-4B block on the same commit
ARCHS = ("fa64", "hy576", "hy16")


def synth_sessions(n_sessions, turns, p0, step, seed, vocab_ids):
    """Turn k of a session is the first p0 + k * step tokens of that session's own
    random token array, so each prompt extends the previous one. The leading
    tokens spell the session index in base len(vocab): with at least n_sessions
    distinct ids in the vocabulary the first token differs between any two
    sessions, with fewer the first few do. Deterministic for a given seed."""
    vocab = sorted({int(v) for v in vocab_ids})
    assert len(vocab) >= 2, "need at least two distinct token ids"
    width = 1
    while len(vocab) ** width < n_sessions:
        width += 1
    total = p0 + (turns - 1) * step
    assert p0 > width and step > 0 and turns >= 1, (p0, step, turns)
    out = []
    for i in range(n_sessions):
        rng = random.Random(f"e5-synth-{seed}-{i}")
        arr = [vocab[rng.randrange(len(vocab))] for _ in range(total)]
        for d in range(width):
            arr[d] = vocab[i // len(vocab) ** d % len(vocab)]
        out.append({"sid": f"synth-{i}", "prompts": [arr[:p0 + k * step] for k in range(turns)]})
    return out


def fit(sessions, max_len, n_out, min_requests=2):
    """Drop requests that leave no room for n_out output tokens; keep sessions
    that still have at least `min_requests` requests (2 by default; 1 for the
    single-turn, no-reuse workload of E12)."""
    out = []
    for s in sessions:
        prompts = [p for p in s["prompts"] if len(p) + n_out <= max_len]
        if len(prompts) >= min_requests:
            out.append({"sid": s["sid"], "prompts": prompts})
    return out


def real_sessions(dataset, path, tokenizer_dir, chat_template, seed, n_sessions, thinking,
                  max_len, n_out, tools_json=None, tok=None):
    """Recorded sessions as token-id prompts. `thinking` is "on" / "off" or a bool.
    `tok` may pass in an already loaded tokenizer."""
    if tok is None:
        from transformers import AutoTokenizer      # lazy: the rest runs without it
        tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    if chat_template:
        with open(chat_template, encoding="utf-8") as f:
            tok.chat_template = f.read()
    raw = list(LOADERS[dataset](path, n_sessions, seed, tools_json=tools_json))
    random.Random(seed).shuffle(raw)                 # the same session order as the E1 replay
    sessions, n_err = [], 0
    for sid, messages, tools in raw:
        try:
            prompts, _, _, _ = render_session(tok, messages, tools, thinking in (True, "on"), max_len)
        except Exception as e:  # counted, not hidden; the message may quote the data, so only its type
            n_err += 1
            print(f"render error {sid}: {type(e).__name__}", flush=True)
            continue
        sessions.append({"sid": sid, "prompts": [[int(x) for x in p] for p in prompts]})
    out = fit(sessions, max_len, n_out)
    print(f"{dataset}: {len(raw)} sessions loaded, {n_err} render errors, {len(out)} used", flush=True)
    return out


def _arrays(session):
    return [np.asarray(p, dtype=np.int64) for p in session["prompts"]]


def lcp_table(session):
    """{k: {j: longest common prefix of prompt k and prompt j}} for j < k."""
    arrs = _arrays(session)
    return {k: {j: lcp(arrs[k], arrs[j]) for j in range(k)} for k in range(len(arrs))}


def infinite_cache_prediction(sessions, arch):
    """{(sid, k): hit} with an unlimited cache, one session at a time.

    fa64:  floor(min(D, P - 1) / 64) * 64, D = longest common prefix with any
           earlier prompt of the session.
    hy576 / hy16: hplab.cache_model.predict_hits with align 576 / 16: the largest
           prompt-end checkpoint of an earlier request that still lies inside
           the common prefix with that request."""
    assert arch in ARCHS, arch
    out = {}
    for s in sessions:
        arrs = _arrays(s)
        for r in predict_hits(arrs, [None] * len(arrs), aligns=(HY_BLOCK, 16)):
            hit = min(r["D_R"], r["P"] - 1) // FA_BLOCK * FA_BLOCK if arch == "fa64" else r[arch]
            out[(s["sid"], r["k"])] = int(hit)
    return out


def cross_session_prefix(sessions):
    """{sid: longest common prefix of its first prompt with any other session's first prompt}."""
    firsts = [np.asarray(s["prompts"][0], dtype=np.int64) for s in sessions]
    out = {}
    for i, s in enumerate(sessions):
        best = 0
        for j, other in enumerate(firsts):
            if j != i and len(other) and len(firsts[i]) and other[0] == firsts[i][0]:
                best = max(best, lcp(firsts[i], other))
        out[s["sid"]] = best
    return out
