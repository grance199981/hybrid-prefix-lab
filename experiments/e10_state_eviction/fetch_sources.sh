#!/usr/bin/env bash
# Fetch, as plain text, the vLLM files that decide how the KV cache blocks of a hybrid model are
# allocated, freed and evicted, at three revisions: our commit (1479deb0), the base of PR #61282
# and the head of PR #61282. Nothing here is executed; the files are read, merged and diffed.
# Usage: fetch_sources.sh <out_dir>
set -u
OUT="$1"
OURS=1479deb0392f1e7d2a69399e4150202284c24509
BASE=44f0ddfd7b                                   # base of PR #61282 (gh api pulls/61282 .base.sha)
THEIRS=5ce4f6c47233bca2d1f1fe02f6fd2f6591e8aed5   # head of PR #61282, checked 2026-11-18
PR_FILES="vllm/v1/core/kv_cache_coordinator.py vllm/v1/core/single_type_kv_cache_manager.py"
READ_ONLY="vllm/v1/core/block_pool.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/kv_cache_utils.py
vllm/v1/core/sched/scheduler.py vllm/v1/kv_cache_interface.py vllm/v1/request.py"
get() {  # rev, path, dest
  mkdir -p "$(dirname "$3")"
  gh api -H "Accept: application/vnd.github.raw" "repos/vllm-project/vllm/contents/$2?ref=$1" > "$3" 2> "$3.err" \
    && rm -f "$3.err" || echo "FAILED $1 $2"
}
BASE_FULL="$(gh api "repos/vllm-project/vllm/commits/$BASE" --jq .sha)"
for f in $PR_FILES; do
  get "$BASE_FULL" "$f" "$OUT/base/$f"
  get "$OURS" "$f" "$OUT/ours/$f"
  get "$THEIRS" "$f" "$OUT/theirs/$f"
done
for f in $READ_ONLY; do get "$OURS" "$f" "$OUT/ours/$f"; done
get "$THEIRS" tests/v1/core/test_mamba_boundary_snapshot_lifetime.py "$OUT/theirs/tests/test_mamba_boundary_snapshot_lifetime.py"
echo "--- merge onto ours"
for f in $PR_FILES; do
  mkdir -p "$(dirname "$OUT/merged/$f")"
  cp "$OUT/ours/$f" "$OUT/merged/$f"
  git merge-file -L ours -L base -L pr61282 "$OUT/merged/$f" "$OUT/base/$f" "$OUT/theirs/$f"
  rc=$?
  same=no; cmp -s "$OUT/base/$f" "$OUT/ours/$f" && same=yes
  echo "conflicts=$rc base==ours:$same lines=$(wc -l < "$OUT/merged/$f") $f"
done
