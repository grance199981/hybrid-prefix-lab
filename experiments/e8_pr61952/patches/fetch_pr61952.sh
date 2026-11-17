#!/usr/bin/env bash
# Fetch the runtime files that PR #61952 changes, at its base commit 5ec99956 and at its head 6891ee3e,
# as plain text, and write their sha256 lists. Nothing here is executed. At the time of the
# measurement the branch was 19 commits ahead of the base and 0 behind, so the head files go on top of
# the vLLM wheel of the base commit unchanged (the PR as it is, no port). pr61952_base_5ec99956.sha256
# and pr61952_head_6891ee3e.sha256 are the lists that were checked in the two environments.
#
# Needs the GitHub CLI (`gh api`). Output: OUT/base, OUT/head, OUT/base.sha256, OUT/head.sha256.
# Usage: bash experiments/e8_pr61952/patches/fetch_pr61952.sh <output dir>
set -u
OUT="$1"
BASE=5ec99956ad4253e70a56c8c0179c5af14fef377d
HEAD=6891ee3e6e3c76628ab091229de004629495a03d
FILES="vllm/compilation/breakable_cudagraph.py
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
vllm/model_executor/layers/mamba/ops/causal_conv1d.py
vllm/models/kimi_k3/nvidia/kda_metadata.py
vllm/third_party/flash_linear_attention/ops/chunk.py
vllm/third_party/flash_linear_attention/ops/chunk_o.py
vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py
vllm/third_party/flash_linear_attention/ops/cumsum.py
vllm/third_party/flash_linear_attention/ops/solve_tril.py
vllm/third_party/flash_linear_attention/ops/wy_fast.py
vllm/v1/attention/backends/gdn_attn.py
vllm/v1/attention/backends/utils.py
vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
get() {  # rev, path, dest
  mkdir -p "$(dirname "$3")"
  gh api -H "Accept: application/vnd.github.raw" "repos/vllm-project/vllm/contents/$2?ref=$1" > "$3" 2> "$3.err" \
    && rm -f "$3.err" || echo "FAILED $1 $2"
}
for f in $FILES; do
  get "$BASE" "$f" "$OUT/base/$f"
  get "$HEAD" "$f" "$OUT/head/$f"
done
for side in base head; do
  ( cd "$OUT/$side" && for f in $FILES; do sha256sum "$f"; done ) > "$OUT/$side.sha256"
  echo "$side: $(wc -l < "$OUT/$side.sha256") files"
done
n=0; for f in $FILES; do cmp -s "$OUT/base/$f" "$OUT/head/$f" || n=$((n + 1)); done
echo "files that differ between base and head: $n"
