"""The E10 logging hook against vLLM's real KVCacheManager on the CPU (skipped without vLLM).

Builds a small hybrid manager the way PR #61282's own test does (one attention group, one
Mamba "align" group, 32 blocks, hash block 16), applies the hook, runs one request through
prefill, two decode steps and free, and checks that:
  - the lookup event carries one longest-hit length per group;
  - the end event finds the state at the prompt checkpoint and reports who holds it;
  - patching does not change what the manager returns.
On main (1479deb0) the end event should show the state already back in the free queue
(ref 0); with PR #61282 it is still held (ref >= 1). The test records which one it saw
instead of asserting either: that is a measurement, not a property of the hook.
"""
import itertools
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

HOOK = os.path.join(os.path.dirname(__file__), "..", "experiments", "e10_state_eviction", "hook")


@pytest.fixture()
def hooked(tmp_path, monkeypatch):
    monkeypatch.setenv("E10_HOOK_DIR", str(tmp_path))
    sys.path.insert(0, HOOK)
    import e10_hook
    e10_hook._DIR = str(tmp_path)
    e10_hook._fh = None
    import vllm.v1.core.block_pool as bp
    import vllm.v1.core.kv_cache_manager as km
    saved = (km.KVCacheManager.get_computed_blocks, km.KVCacheManager.free,
             km.KVCacheManager.pop_blocks_for_free, bp.BlockPool.get_new_blocks,
             bp.BlockPool._maybe_evict_cached_block)
    e10_hook.apply("vllm.v1.core.kv_cache_manager", km)
    e10_hook.apply("vllm.v1.core.block_pool", bp)
    e10_hook._state["groups"] = ["attn", "mamba"]
    yield e10_hook, tmp_path
    (km.KVCacheManager.get_computed_blocks, km.KVCacheManager.free,
     km.KVCacheManager.pop_blocks_for_free, bp.BlockPool.get_new_blocks,
     bp.BlockPool._maybe_evict_cached_block) = saved
    if e10_hook._fh:
        e10_hook._fh.close()
        e10_hook._fh = None
    sys.path.remove(HOOK)


def _manager():
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import init_none_hash
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
    init_none_hash(sha256)
    attn = KVCacheGroupSpec(["attn"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32))
    mamba = KVCacheGroupSpec(["mamba"], MambaSpec(block_size=32, shapes=((1, 1),), dtypes=(torch.float32,),
                                                  mamba_cache_mode="align"))
    cfg = KVCacheConfig(num_blocks=32, kv_cache_tensors=[], kv_cache_groups=[attn, mamba])
    return KVCacheManager(cfg, max_model_len=4096, scheduler_block_size=32, hash_block_size=16, enable_caching=True)


def _request(rid, prompt):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.request import Request
    return Request(request_id=rid, prompt_token_ids=prompt, mm_features=None,
                   sampling_params=SamplingParams(max_tokens=64), pooling_params=None,
                   block_hasher=get_request_block_hasher(16, sha256))


def test_hook_events(hooked):
    e10_hook, d = hooked
    m = _manager()
    ids = itertools.count(1000)
    req = _request("req", [next(ids) for _ in range(49)])
    computed, n, _ = m.get_computed_blocks(req)
    assert n == 0
    for i, chunk in enumerate([48, 1]):               # the checkpoint at 48 sits inside a 32-token block
        assert m.allocate_slots(req, chunk, n if i == 0 else 0, computed if i == 0 else None) is not None
        req.num_computed_tokens += chunk
        m.coordinator.new_step_starts()
        _, retained = m.take_kv_cache_block_copies()
        m.block_pool.free_blocks(retained)
    for _ in range(2):
        req.append_output_token_ids(next(ids))
        assert m.allocate_slots(req, 1) is not None
        req.num_computed_tokens += 1
        m.coordinator.new_step_starts()
        _, retained = m.take_kv_cache_block_copies()
        m.block_pool.free_blocks(retained)
    m.free(req)
    follow = _request("next", list(req.all_token_ids) + [next(ids) for _ in range(20)])
    _, hit, _ = m.get_computed_blocks(follow)

    events = [json.loads(line) for f in os.listdir(d) if f.startswith("events_")
              for line in open(os.path.join(d, f))]
    lookups = [e for e in events if e["ev"] == "lookup"]
    ends = [e for e in events if e["ev"] == "end"]
    assert [e["rid"] for e in lookups] == ["req", "next"]
    assert all(len(e["groups"]) == 2 for e in lookups)
    assert lookups[1]["hit"] == hit == 48 and lookups[1]["groups"][1] == 48
    (end,) = ends
    assert end["c"] == 48 and end["state"][0]["present"] is True
    assert not [e for e in events if e["ev"] == "error"]
    print("state ref at request end:", end["state"][0]["ref"])
