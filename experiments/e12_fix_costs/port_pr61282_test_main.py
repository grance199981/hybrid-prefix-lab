"""E12 C: port PR #61282's test file to the KV cache interface of vLLM main @ 191565a2.

Main replaced `FullAttentionSpec` / `KVCacheGroupSpec` (with the manager policy inside the spec) by
`AttentionSpec` + a `ManagerConfig` per `KVCacheGroup` (RFC #42449, PR #61452). The port changes only
how the test builds its two groups and how it reads a group's block size and spec type; the requests,
the steps and every assertion stay as the PR wrote them. Groups are built the way main's own test
helpers build them (`tests/v1/core/utils.py`: `full_layer`, `mamba_layer`, `new_kv_cache_group`).
Usage: port_pr61282_test_main.py <PR test file> <out file>
"""
import sys

SUBS = [
    ("""from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)""",
     """from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroup,
    MambaSpec,
)


def _group(name, layer):
    # As main's tests/v1/core/utils.py new_kv_cache_group(): one layer, its manager policy.
    return KVCacheGroup(
        {name: layer.spec}, layer.manager_config, layer.spec.block_size, layer.spec.dcp_sharded
    )"""),
    ("""    attn = KVCacheGroupSpec(
        ["attn"],
        FullAttentionSpec(
            block_size=HASH_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )
    mamba = KVCacheGroupSpec(
        ["mamba"],
        MambaSpec(
            block_size=mamba_block_size,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
    )""",
     """    attn = _group(
        "attn",
        FullAttentionManager.layer_config(
            AttentionSpec(
                block_size=HASH_BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            )
        ),
    )
    mamba = _group(
        "mamba",
        MambaManager.layer_config(
            MambaSpec(
                block_size=mamba_block_size,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
            ),
            mamba_cache_mode="align",
        ),
    )"""),
    ("group.kv_cache_spec.block_size for group in config.kv_cache_groups",
     "group.block_size for group in config.kv_cache_groups"),
    ("isinstance(group.kv_cache_spec, spec_type)",
     "isinstance(next(iter(group.kv_cache_specs.values())), spec_type)"),
    ("FullAttentionSpec", "AttentionSpec"),
]


def main(src, dst):
    text = open(src, encoding="utf-8").read()
    for old, new in SUBS:
        n = text.count(old)
        assert n >= 1, f"not found:\n{old[:200]}"
        text = text.replace(old, new)
    assert "KVCacheGroupSpec" not in text and "kv_cache_spec." not in text
    open(dst, "w", encoding="utf-8", newline="\n").write(text)
    print("ported", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
