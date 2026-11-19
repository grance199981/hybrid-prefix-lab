"""E12 C: resolve the three-way merge of PR #61282 onto vLLM main @ 191565a2.

Input: the directory written by experiments/e10_state_eviction/fetch_sources.sh with
OURS=191565a2cd0c42a1e4fd4d063f53d0ce383b364f. Four conflicts:

  single_type_kv_cache_manager.py (1): the same as on 1479deb0; resolved as in
      experiments/e10_state_eviction/port_pr61282.py (take the PR's `_prompt_boundary`, give it
      main's `num_hash_lookahead_tokens`).
  kv_cache_coordinator.py (3): main changed `free(request_id)` and `pop_blocks_for_free(request_id)`
      to take the request and to call `manager.cache_blocks_on_free(request)` before each manager
      frees; the PR changed the loops to walk `self._free_order` (attention groups first, Mamba
      groups last) instead of the group order. Resolution: main's signatures and its
      `cache_blocks_on_free` call, inside the PR's loops; the docstring keeps both statements.

Writes <dir>/port/<path>, compiles each file and prints its sha256.
Usage: port_pr61282_main.py <fetch dir>
"""
import hashlib
import os
import py_compile
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e10_state_eviction"))
import port_pr61282 as P  # noqa: E402

COORD = "vllm/v1/core/kv_cache_coordinator.py"
SINGLE = "vllm/v1/core/single_type_kv_cache_manager.py"

C1 = '''<<<<<<< ours
    def free(self, request: Request) -> None:
        """Free the blocks for the request.
=======
    def free(self, request_id: str) -> None:
        """Free the blocks for the request, one group at a time in
        ``_free_order``.
>>>>>>> pr61282
'''
R1 = '''    def free(self, request: Request) -> None:
        """Free the blocks for the request, one group at a time in
        ``_free_order``.
'''
C2 = '''<<<<<<< ours
        for manager in self.single_type_managers:
            manager.cache_blocks_on_free(request)
            manager.free(request.request_id)
=======
        for manager in self._free_order:
            manager.free(request_id)
>>>>>>> pr61282
'''
R2 = '''        for manager in self._free_order:
            manager.cache_blocks_on_free(request)
            manager.free(request.request_id)
'''
C3 = '''<<<<<<< ours
        for manager in self.single_type_managers:
            manager.cache_blocks_on_free(request)
            blocks.extend(manager.pop_blocks_for_free(request.request_id))
=======
        for manager in reversed(self._free_order):
            blocks.extend(manager.pop_blocks_for_free(request_id))
>>>>>>> pr61282
'''
R3 = '''        for manager in reversed(self._free_order):
            manager.cache_blocks_on_free(request)
            blocks.extend(manager.pop_blocks_for_free(request.request_id))
'''


def main(root):
    for rel in (COORD, SINGLE):
        src = os.path.join(root, "merged", rel)
        dst = os.path.join(root, "port", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        text = open(src, encoding="utf-8").read()
        if rel == SINGLE:
            assert text.count(P.CONFLICT) == 1, "single-type conflict is not the expected one"
            text = text.replace(P.CONFLICT, P.RESOLVED)
            assert text.count(P.BOUNDARY) == 1, "_prompt_boundary is not the expected one"
            text = text.replace(P.BOUNDARY, P.BOUNDARY_PORTED)
        else:
            for c, r in ((C1, R1), (C2, R2), (C3, R3)):
                assert text.count(c) == 1, "coordinator conflict is not the expected one:\n" + c
                text = text.replace(c, r)
        assert "<<<<<<<" not in text and ">>>>>>>" not in text and "\n=======\n" not in text, rel
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        py_compile.compile(dst, doraise=True, cfile=dst + ".pyc_check")
        os.remove(dst + ".pyc_check")
        print(hashlib.sha256(text.encode("utf-8")).hexdigest(), rel)
    shutil.rmtree(os.path.join(root, "port", "__pycache__"), ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1])
