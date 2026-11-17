#!/usr/bin/env bash
# Fetch the runtime files that PR #65186 changes, at its merge base 2f70c32d and at its head e4ee672b,
# as plain text, and write their sha256 lists. Nothing here is executed. At the time of the R3
# measurement the branch was 2 commits ahead of the merge base and 0 behind it, so the head files go on
# top of the vLLM wheel of 2f70c32d unchanged (no port). The other files of the PR are tests and CI
# configuration. pr65186_base_2f70c32d.sha256 and pr65186_head_e4ee672b.sha256 are the lists that
# were checked in the two environments.
#
# Needs the GitHub CLI (`gh api`). Output: OUT/base, OUT/head, OUT/base.sha256, OUT/head.sha256.
# Usage: bash experiments/e7_upstream_prs/patches/fetch_pr65186_head.sh <output dir>
set -u
OUT="$1"
BASE=2f70c32d07af8063b632cb67fa160b3fd19ea90b
HEAD=e4ee672bd2b79bbad63c6cb29a98364501f4ae59
FILES="vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
vllm/third_party/flash_linear_attention/ops/chunk.py
vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py"
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
