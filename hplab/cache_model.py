"""What the prefix cache should hit, and how many prefill steps the scheduler takes,
computed from vLLM's own rules instead of measured.

Two pieces:

1. Hit prediction for one session (`predict_hits`). A full-attention model can
   reuse any block-aligned common prefix. A hybrid model (Gated DeltaNet layers
   plus full attention, e.g. Qwen3.5) can only resume where a linear-attention
   state was saved: by default one state per request, at the last block boundary
   before the end of its prompt (`checkpoint_position`).

2. Prefill step count (`count_steps`). A transcription of
   vllm/v1/core/sched/scheduler.py:_mamba_block_aligned_split and of
   vllm/v1/kv_cache_interface.py:get_mamba_prefill_checkpoint_position /
   is_mamba_prefill_checkpoint_valid at vLLM commit 1479deb0. Parameters:
     alignment = None : the backend cannot save the checkpoint inside the final chunk
                        (vLLM main with the Triton GDN kernels)
     alignment = 64   : it can, on a 64-token grid (PR #65186)
     unit = None      : default configuration, checkpoints on the block grid only
     unit = 16 / 64   : --prefix-match-unit
     rule = "66938"   : the validity check after upstream PR #66938
                        (`checkpoint_col >= start // block`); the default "1479deb0"
                        is `checkpoint_col > (start - 1) // block`.
   Not modelled: speculative decoding, several requests in one batch.
"""
import statistics

import numpy as np

FA_BLOCK = 16           # block of the full-attention accounting policies "fa16_*"
HYBRID_BLOCK = 528      # Qwen3.5-4B block on vLLM 0.32; vLLM 1479deb0 uses 576
BLOCK = 576             # Qwen3.5-4B block on vLLM 1479deb0
MAX_CHUNK = 2048        # SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS on 1479deb0


def lcp(a, b):
    """Length of the longest common prefix of two int arrays."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    neq = np.nonzero(np.asarray(a[:n]) != np.asarray(b[:n]))[0]
    return int(neq[0]) if len(neq) else n


def checkpoint_position(num_tokens, align):
    """Where the prompt-tail state is saved: the last multiple of `align` before
    the final prompt token (get_mamba_prefill_checkpoint_position, no speculation)."""
    return max((num_tokens - 1) // align * align, 0)


# ------------------------------------------------------------- hit prediction
def predict_hits(prompts, ideals, aligns=(HYBRID_BLOCK, FA_BLOCK)):
    """prompts[k], ideals[k]: int arrays for request k of one session, in order.
    ideals[k] is the prompt followed by the re-rendered reply, or None.

    Per request:
      P       prompt length
      D_R     longest common prefix with any earlier prompt (the model's own
              output is never carried back: "recorded replay")
      D_I     same, but earlier requests also count with their re-rendered reply
              ("ideal round trip")
      fa16_R / fa16_I   full-attention hit on 16-token blocks
      hy<align>         hybrid hit when each request leaves one state at its
                        prompt-tail checkpoint on an `align` grid
      hy_exact_end_I    hybrid hit if a state were kept at the exact end of
                        every earlier request
    """
    rows = []
    for k, pk in enumerate(prompts):
        P = len(pk)
        cap = P - 1  # the last prompt token is always recomputed
        l_r = [lcp(pk, prompts[j]) for j in range(k)]
        l_i = [lcp(pk, ideals[j]) if ideals[j] is not None else l_r[j] for j in range(k)]
        d_r = max(l_r, default=0)
        d_i = max(max(l_i, default=0), d_r)
        row = {
            "k": k, "P": P, "D_R": d_r, "D_I": d_i,
            "fa16_R": min(d_r, cap) // FA_BLOCK * FA_BLOCK,
            "fa16_I": min(d_i, cap) // FA_BLOCK * FA_BLOCK,
            "hy_exact_end_I": min(d_i, cap),
        }
        for align in aligns:
            # Request j left one state at its prompt-tail checkpoint. It is usable
            # iff every token up to that position also opens this prompt.
            ckpts = [checkpoint_position(len(prompts[j]), align) for j in range(k)]
            usable = [c for j, c in enumerate(ckpts) if c <= l_r[j]]
            row[f"hy{align}"] = min(max(usable, default=0), cap)
        if k > 0:
            p_prev = len(prompts[k - 1])
            row["P_prev"] = p_prev
            row["g_obs"] = p_prev - l_r[k - 1]
            row["residue16"] = (p_prev - 1) % FA_BLOCK
        rows.append(row)
    return rows


def policy_names(aligns):
    return ["fa16_R", "fa16_I"] + [f"hy{a}" for a in aligns] + ["hy_exact_end_I"]


POLICIES = policy_names((HYBRID_BLOCK, FA_BLOCK))


def summarize(rows, policies=POLICIES):
    """rows: all requests of all sessions; the first request of each session is excluded."""
    rs = [r for r in rows if r["k"] > 0]
    out = {"n_requests": len(rs), "n_sessions": len({r["sid"] for r in rows})}
    if not rs:
        return out
    total_p = sum(r["P"] for r in rs)
    out["prompt_tokens_median"] = statistics.median(r["P"] for r in rs)
    for pol in policies:
        rec = sorted(r["P"] - r[pol] for r in rs)
        out[pol] = {
            "hit_rate": sum(r[pol] for r in rs) / total_p,
            "recompute_median": statistics.median(rec),
            "recompute_p90": rec[min(len(rec) - 1, int(0.9 * len(rec)))],
        }
    for hy in [p for p in policies if p[2:].isdigit()]:
        for fa in ("fa16_R", "fa16_I"):
            out[f"gap_pp_{fa}_minus_{hy}"] = 100 * (out[fa]["hit_rate"] - out[hy]["hit_rate"])
            den = out[fa]["recompute_median"]
            out[f"recompute_median_ratio_{hy}_over_{fa}"] = (
                out[hy]["recompute_median"] / den if den else None)
    g = {}
    for r in rs:
        key = ("after_user" if r["after_user_query"] else "in_chain")
        g.setdefault(key, {}).setdefault(str(r["g_obs"]), 0)
        g[key][str(r["g_obs"])] += 1
    out["g_obs_histogram"] = g
    return out


# ------------------------------------------------------- prefill step count
def internal_checkpoint_ok(start, end, position, hash_block, alignment, block=BLOCK, rule="1479deb0"):
    """is_mamba_prefill_checkpoint_valid: can the kernel save the checkpoint at
    `position` inside the chunk [start, end)?"""
    if alignment is None:
        return False
    assert rule in ("1479deb0", "66938"), rule
    checkpoint_col = -(-end // block) - 2
    if rule == "1479deb0":
        column_ok = checkpoint_col > (start - 1) // block
    else:
        column_ok = checkpoint_col >= start // block
    return (
        start % hash_block == 0
        and column_ok
        and start + hash_block <= position
        and start < position < end
        and (position - start) % alignment == 0
    )


def chunk_end(start, prompt, unit=None, alignment=None, block=BLOCK, max_chunk=MAX_CHUNK, rule="1479deb0"):
    """End of the prefill chunk that starts at `start` (_mamba_block_aligned_split)."""
    if start >= prompt:
        return prompt
    partial = unit is not None and unit < block          # mamba_partial_cache_hit
    hash_block = unit if partial else block
    last_cache = checkpoint_position(prompt, block)
    end = min(prompt, start + max_chunk)
    position = checkpoint_position(prompt, hash_block if partial else block)
    internal = end >= prompt and internal_checkpoint_ok(start, end, position, hash_block, alignment, block, rule)
    if internal:
        last_cache = 0
    if end < prompt:
        aligned_end = end // block * block
        if aligned_end > start or block <= max_chunk:
            end = aligned_end
    next_block = (start // block + 1) * block
    tail = checkpoint_position(prompt, hash_block) if partial and not internal else 0
    stops = (
        next_block if start % block != 0 and not internal else 0,
        last_cache,
        tail if last_cache < tail < prompt else 0,
    )
    return min((s for s in stops if start < s < end), default=end)


def count_steps(cached, prompt, unit=None, alignment=None, block=BLOCK, max_chunk=MAX_CHUNK, rule="1479deb0"):
    """Number of prefill chunks (scheduler steps) from `cached` to `prompt` tokens."""
    n, start = 0, cached
    while start < prompt:
        end = chunk_end(start, prompt, unit, alignment, block, max_chunk, rule)
        assert end > start, (start, prompt, unit, alignment)
        n, start = n + 1, end
    return n
