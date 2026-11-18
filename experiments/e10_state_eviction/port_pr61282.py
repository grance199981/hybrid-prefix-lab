"""Resolve the one conflict of the three-way merge of PR #61282 onto 1479deb0.

Input: the directory written by fetch_sources.sh. The merge of kv_cache_coordinator.py is
clean; single_type_kv_cache_manager.py has one conflict: 1479deb0 passes
`num_hash_lookahead_tokens` to `get_mamba_prefill_checkpoint_position` at the prompt
checkpoint test, and the PR moved that call into a new method `_prompt_boundary`. The
resolution takes the PR's line and gives `_prompt_boundary` the same extra argument, so
the prompt boundary stays the one 1479deb0 computes. Writes <dir>/port/<path> and prints
the sha256 of each output file.

Usage: port_pr61282.py <fetch dir>
"""
import hashlib
import os
import py_compile
import shutil
import sys

FILES = ("vllm/v1/core/kv_cache_coordinator.py", "vllm/v1/core/single_type_kv_cache_manager.py")

CONFLICT = """<<<<<<< ours
            is_prompt_checkpoint = (
                checkpoint_position
                == get_mamba_prefill_checkpoint_position(
                    request.num_prompt_tokens,
                    self.cache_hit_alignment_tokens,
                    self.drop_eagle_checkpoint_block,
                    num_hash_lookahead_tokens=self.block_pool.num_hash_lookahead_tokens,
                )
            )
=======
            is_prompt_checkpoint = checkpoint_position == self._prompt_boundary(request)
>>>>>>> pr61282
"""
RESOLVED = """            is_prompt_checkpoint = checkpoint_position == self._prompt_boundary(request)
"""
BOUNDARY = """        return get_mamba_prefill_checkpoint_position(
            request.num_prompt_tokens,
            self.cache_hit_alignment_tokens,
            self.drop_eagle_checkpoint_block,
        )
"""
BOUNDARY_PORTED = """        return get_mamba_prefill_checkpoint_position(
            request.num_prompt_tokens,
            self.cache_hit_alignment_tokens,
            self.drop_eagle_checkpoint_block,
            num_hash_lookahead_tokens=self.block_pool.num_hash_lookahead_tokens,
        )
"""


def main(root):
    for rel in FILES:
        src = os.path.join(root, "merged", rel)
        dst = os.path.join(root, "port", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, encoding="utf-8") as f:
            text = f.read()
        if rel.endswith("single_type_kv_cache_manager.py"):
            assert text.count(CONFLICT) == 1, "the conflict is not the expected one"
            text = text.replace(CONFLICT, RESOLVED)
            assert text.count(BOUNDARY) == 1, "_prompt_boundary is not the expected one"
            text = text.replace(BOUNDARY, BOUNDARY_PORTED)
        assert "<<<<<<<" not in text and ">>>>>>>" not in text, rel
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        py_compile.compile(dst, doraise=True, cfile=dst + ".pyc_check")
        os.remove(dst + ".pyc_check")
        print(hashlib.sha256(text.encode("utf-8")).hexdigest(), rel)
    shutil.rmtree(os.path.join(root, "port", "__pycache__"), ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1])
