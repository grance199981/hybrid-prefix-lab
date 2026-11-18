"""E10: record, inside the vLLM engine process, what the prefix cache holds when a
request is admitted and when it ends, and which cached blocks get evicted.

Logging only: every patched method calls the original unchanged and returns its
result; the extra work is read-only lookups (`find_longest_cache_hit_per_group`
goes through `BlockPool.get_cached_block`, a dict lookup that does not touch the
free queue). Any exception in the logging code is caught and counted, never raised.

Loaded by `sitecustomize.py` in this directory when E10_HOOK_DIR is set. Writes
one JSON line per event to $E10_HOOK_DIR/events_<pid>.jsonl:

  {"ev": "init", ...}     group order and types, scheduler class and limits
  {"ev": "lookup", ...}   each get_computed_blocks: prompt length, first token,
                          the hit, and each group's own longest hit
  {"ev": "end", ...}      each request end (free / pop_blocks_for_free): whether
                          the state at the prompt checkpoint c = floor((P-1)/16)*16
                          is still cached and who holds it (ref_cnt), and how far
                          the attention group reaches toward c
  {"ev": "evict", ...}    each cached block evicted while new blocks are handed out
  {"ev": "error", ...}    an exception inside the logging code (first 20 only)

and, once, scheduler_init_<pid>.pkl with the Scheduler's vllm_config and
kv_cache_config (input of cpu_replay.py).
"""
import json
import os
import pickle
import threading
import time

_DIR = os.environ.get("E10_HOOK_DIR", "")
_lock = threading.Lock()
_fh = None
_errors = 0
_state = {"in_alloc": False, "hash_block": 16, "groups": None}


def _emit(rec):
    global _fh
    rec["t"] = time.time_ns()
    line = json.dumps(rec, separators=(",", ":"))
    with _lock:
        if _fh is None:
            _fh = open(os.path.join(_DIR, f"events_{os.getpid()}.jsonl"), "a", buffering=1, encoding="utf-8")
        _fh.write(line + "\n")


def _error(where, exc):
    global _errors
    _errors += 1
    if _errors <= 20:
        try:
            _emit({"ev": "error", "where": where, "msg": repr(exc)[:300]})
        except Exception:
            pass


def _group_kinds(kv_cache_config):
    """['attn'|'mamba'|other class name] per group, in group order."""
    out = []
    for g in kv_cache_config.kv_cache_groups:
        name = type(g.kv_cache_spec).__name__
        out.append("mamba" if "Mamba" in name else "attn" if "Attention" in name else name)
    return out


def _tok0(request):
    ids = getattr(request, "prompt_token_ids", None)
    return int(ids[0]) if ids is not None and request.num_prompt_tokens > 0 else None


def _head(request, n=4):
    """The first n prompt tokens: in the synthetic sessions the leading tokens spell the
    session index, so they identify the session (they are random ids, not text)."""
    ids = getattr(request, "prompt_token_ids", None)
    return [int(ids[i]) for i in range(min(n, request.num_prompt_tokens))] if ids is not None else None


def _picklable(vllm_config):
    """A shallow copy of vllm_config without the compilation config's forward context,
    which holds the model's layers (GPU tensors, local weight-loader functions)."""
    import copy
    vc = copy.copy(vllm_config)
    cc = copy.copy(vllm_config.compilation_config)
    object.__setattr__(cc, "static_forward_context", {})
    object.__setattr__(vc, "compilation_config", cc)
    return vc


def checkpoint_position(num_prompt_tokens, hash_block):
    """The prompt checkpoint a later turn can resume from (no EAGLE, lookahead <= 1)."""
    return max((num_prompt_tokens - 1) // hash_block * hash_block, 0)


def _per_group(manager, block_hashes, max_len):
    blocks, lengths = manager.coordinator.find_longest_cache_hit_per_group(block_hashes, max_len)
    return blocks, list(lengths)


# ------------------------------------------------------------------ patches
def _patch_scheduler(mod):
    cls = mod.Scheduler
    orig = cls.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        try:
            vllm_config = kwargs.get("vllm_config", args[0] if args else None)
            kv_cache_config = kwargs.get("kv_cache_config", args[1] if len(args) > 1 else None)
            kinds = _group_kinds(kv_cache_config)
            _state["groups"] = kinds
            hb = kwargs.get("hash_block_size") or getattr(self.kv_cache_manager, "hash_block_size", None)
            if hb:
                _state["hash_block"] = int(hb)
            sc = vllm_config.scheduler_config
            _emit({"ev": "init", "scheduler": type(self).__name__, "groups": kinds,
                   "block_sizes": [g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups],
                   "num_blocks": kv_cache_config.num_blocks, "hash_block_size": _state["hash_block"],
                   "block_size": kwargs.get("block_size"),
                   "max_num_batched_tokens": sc.max_num_batched_tokens, "max_num_seqs": sc.max_num_seqs,
                   "async_scheduling": getattr(sc, "async_scheduling", None),
                   "long_prefill_token_threshold": getattr(sc, "long_prefill_token_threshold", None),
                   "policy": str(getattr(sc, "policy", None))})
            path = os.path.join(_DIR, f"scheduler_init_{os.getpid()}.pkl")
            init_kwargs = {k: v for k, v in kwargs.items()
                           if k in ("include_finished_set", "log_stats", "block_size", "hash_block_size")}
            try:
                with open(path, "wb") as f:
                    pickle.dump({"cls": type(self).__name__, "vllm_config": _picklable(vllm_config),
                                 "kv_cache_config": kv_cache_config, "kwargs": init_kwargs}, f)
            except Exception as exc:          # keep at least the KV cache layout
                _error("pickle vllm_config", exc)
                with open(path, "wb") as f:
                    pickle.dump({"cls": type(self).__name__, "vllm_config": None,
                                 "kv_cache_config": kv_cache_config, "kwargs": init_kwargs}, f)
        except Exception as exc:
            _error("scheduler init", exc)

    cls.__init__ = __init__


def _patch_manager(mod):
    cls = mod.KVCacheManager
    orig_lookup = cls.get_computed_blocks
    orig_free = cls.free
    orig_pop = cls.pop_blocks_for_free

    def get_computed_blocks(self, request, max_cache_hit_length=None):
        res = orig_lookup(self, request, max_cache_hit_length)
        try:
            limit = request.get_max_cache_hit_length(max_cache_hit_length)
            _, lengths = _per_group(self, request.block_hashes, limit)
            _emit({"ev": "lookup", "rid": request.request_id, "P": request.num_prompt_tokens,
                   "tok0": _tok0(request), "head": _head(request),
                   "hit": int(res[1]), "groups": lengths, "limit": int(limit),
                   "preempted": int(getattr(request, "num_preemptions", 0))})
        except Exception as exc:
            _error("lookup", exc)
        return res

    def _record_end(self, request, how):
        try:
            P = request.num_prompt_tokens
            c = checkpoint_position(P, _state["hash_block"])
            blocks, lengths = _per_group(self, request.block_hashes, c)
            kinds = _state["groups"] or ["?"] * len(lengths)
            state = []
            for gid, (kind, n, blks) in enumerate(zip(kinds, lengths, blocks)):
                if kind != "mamba":
                    continue
                present = n == c and c > 0
                state.append({"g": gid, "present": present,
                              "ref": int(blks[-1].ref_cnt) if present and blks else None})
            attn = [n for kind, n in zip(kinds, lengths) if kind == "attn"]
            _emit({"ev": "end", "how": how, "rid": request.request_id, "P": P, "c": c,
                   "tok0": _tok0(request), "head": _head(request),
                   "n_out": int(request.num_output_tokens), "attn_to_c": attn, "state": state})
        except Exception as exc:
            _error("end", exc)

    def free(self, request):
        _record_end(self, request, "free")
        return orig_free(self, request)

    def pop_blocks_for_free(self, request):
        _record_end(self, request, "pop")
        return orig_pop(self, request)

    cls.get_computed_blocks = get_computed_blocks
    cls.free = free
    cls.pop_blocks_for_free = pop_blocks_for_free


def _patch_pool(mod):
    from vllm.v1.core.kv_cache_utils import get_group_id

    cls = mod.BlockPool
    orig_new = cls.get_new_blocks
    orig_evict = cls._maybe_evict_cached_block

    def get_new_blocks(self, num_blocks):
        _state["in_alloc"] = True
        try:
            return orig_new(self, num_blocks)
        finally:
            _state["in_alloc"] = False

    def _maybe_evict_cached_block(self, block):
        gid = ntok = None
        try:
            if block.block_hash is not None:
                gid = int(get_group_id(block.block_hash))
                ntok = getattr(block, "block_hash_num_tokens", None)
        except Exception as exc:
            _error("evict pre", exc)
        out = orig_evict(self, block)
        if out and gid is not None:
            try:
                kinds = _state["groups"]
                _emit({"ev": "evict", "g": gid, "kind": kinds[gid] if kinds else None,
                       "ntok": ntok, "alloc": _state["in_alloc"], "block": int(block.block_id)})
            except Exception as exc:
                _error("evict", exc)
        return out

    cls.get_new_blocks = get_new_blocks
    cls._maybe_evict_cached_block = _maybe_evict_cached_block


PATCHES = {
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
    "vllm.v1.core.kv_cache_manager": _patch_manager,
    "vllm.v1.core.block_pool": _patch_pool,
}


def apply(name, module):
    """Patch an already imported target module (used by the import hook and the tests)."""
    try:
        PATCHES[name](module)
    except Exception as exc:
        _error(f"patch {name}", exc)


def install():
    """Patch the target modules as soon as they are imported."""
    import importlib.abc
    import sys

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in PATCHES:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
            loader, orig_exec = spec.loader, spec.loader.exec_module

            def exec_module(module, _orig=orig_exec, _name=name):
                _orig(module)
                apply(_name, module)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
