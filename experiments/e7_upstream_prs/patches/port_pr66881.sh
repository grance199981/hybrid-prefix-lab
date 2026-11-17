#!/usr/bin/env bash
# Port PR #66881 onto vLLM 1479deb0, as plain text: fetch the runtime files the PR touches at three
# revisions (merge base, 1479deb0, PR head c19296d4), three-way merge them with `git merge-file`, and
# copy the files the PR adds. Nothing here is executed; the merged tree goes over a copy of a
# 1479deb0 environment. port_pr66881.sha256 lists the sha256 of the merged files that were installed
# (0 conflicts; 10 files equal to the PR head, scheduler.py keeps 2 lines that main added later).
#
# Needs the GitHub CLI (`gh api`). Output: OUT/base, OUT/ours, OUT/theirs, OUT/merged.
# Usage: bash experiments/e7_upstream_prs/patches/port_pr66881.sh <output dir>
set -u
OUT="$1"
BASE=ca3c154d2674
OURS=1479deb0392f1e7d2a69399e4150202284c24509
THEIRS=c19296d49934329b0fbb6853759690bc6d6c11d9
MOD="vllm/config/cache.py vllm/engine/arg_utils.py vllm/models/glm5next/common/attention.py
vllm/models/qwen4_exp/common/qsa_indexer.py vllm/v1/core/block_pool.py vllm/v1/core/kv_cache_manager.py
vllm/v1/core/sched/scheduler.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/kv_cache_interface.py
vllm/v1/metrics/loggers.py vllm/v1/metrics/stats.py"
NEW="vllm/v1/core/hybrid_checkpoint/__init__.py vllm/v1/core/hybrid_checkpoint/cost.py
vllm/v1/core/hybrid_checkpoint/eviction.py vllm/v1/core/hybrid_checkpoint/frontier.py
vllm/v1/core/hybrid_checkpoint/keys.py vllm/v1/core/hybrid_checkpoint/manager.py
vllm/v1/core/hybrid_checkpoint/units.py"
get() {  # rev, path, dest
  mkdir -p "$(dirname "$3")"
  gh api -H "Accept: application/vnd.github.raw" "repos/vllm-project/vllm/contents/$2?ref=$1" > "$3" 2> "$3.err" \
    && rm -f "$3.err" || echo "FAILED $1 $2"
}
for f in $MOD; do
  get "$BASE" "$f" "$OUT/base/$f"
  get "$OURS" "$f" "$OUT/ours/$f"
  get "$THEIRS" "$f" "$OUT/theirs/$f"
done
for f in $NEW; do
  get "$THEIRS" "$f" "$OUT/theirs/$f"
  gh api "repos/vllm-project/vllm/contents/$f?ref=$OURS" > /dev/null 2>&1 && echo "UNEXPECTED: $f exists at ours"
done
echo "--- merge"
for f in $MOD; do
  mkdir -p "$(dirname "$OUT/merged/$f")"
  cp "$OUT/ours/$f" "$OUT/merged/$f"
  git merge-file -L ours -L base -L pr66881 "$OUT/merged/$f" "$OUT/base/$f" "$OUT/theirs/$f"
  rc=$?
  same=no; cmp -s "$OUT/base/$f" "$OUT/ours/$f" && same=yes
  echo "conflicts=$rc base==ours:$same lines=$(wc -l < "$OUT/merged/$f") $f"
done
for f in $NEW; do
  mkdir -p "$(dirname "$OUT/merged/$f")"; cp "$OUT/theirs/$f" "$OUT/merged/$f"
  echo "new lines=$(wc -l < "$OUT/merged/$f") $f"
done
