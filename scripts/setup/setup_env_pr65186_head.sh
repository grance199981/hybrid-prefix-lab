#!/usr/bin/env bash
# Two environments for PR #65186 at the head that was merged (e4ee672b) and the main it is
# based on (E7), built from envs/vllm-5ec99956 (scripts/setup/setup_env_pr61952.sh):
#   envs/vllm-2f70c32d          copy of envs/vllm-5ec99956 with the per-commit wheel of main @ 2f70c32d
#                               (the merge base of the PR; it contains #66938)
#   envs/vllm-pr65186-e4ee672b  copy of the first one with the 3 runtime Python files of the PR head e4ee672b on top
# Every other package stays at the version envs/vllm-5ec99956 has. envs/vllm-5ec99956 and
# the other environments are only read. Every step refuses to overwrite.
#
# Usage: bash scripts/setup/setup_env_pr65186_head.sh $HP_ROOT <step> [arg], steps in this order:
#   clone-base | dry-run | install | clone-pr | overlay <dir> | check
# overlay <dir>: as in setup_env_pr61952.sh (base.sha256, head.sha256, head/), for the 3 files.
set -euo pipefail

ROOT=${1:?usage: setup_env_pr65186_head.sh <remote_root> <step> [arg]}
STEP=${2:?step}
SRC=$ROOT/envs/vllm-5ec99956
BASE=$ROOT/envs/vllm-2f70c32d
PR=$ROOT/envs/vllm-pr65186-e4ee672b
COMMIT=2f70c32d07af8063b632cb67fa160b3fd19ea90b
VERSION="0.34.0rc1.dev556+g2f70c32d0"
LOG=$ROOT/logs
SP=lib/python3.12/site-packages
FREEZE_BEFORE=$LOG/env_vllm_5ec99956_freeze_before_2f70c32d.txt
CONSTRAINTS=$LOG/vllm_2f70c32d_constraints.txt

export PIP_CACHE_DIR=$ROOT/cache/pip
export TMPDIR=$ROOT/tmp
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_INDEX_URL=${PIP_INDEX_MIRROR:-https://pypi.org/simple}
export PIP_RETRIES=10
export PIP_DEFAULT_TIMEOUT=60

clone() {  # from, to
  [ -e "$2" ] && { echo "refusing to overwrite existing env: $2" >&2; exit 2; }
  echo "[$(date -Is)] cp -a $1 $2"
  ionice -c3 nice -n 19 cp -a "$1" "$2"
  local a b n
  a="envs/$(basename "$1")"; b="envs/$(basename "$2")"
  n=$(grep -rlI "$a" "$2/bin" "$2/pyvenv.cfg" | wc -l)
  grep -rlI --null "$a" "$2/bin" "$2/pyvenv.cfg" | xargs -0 sed -i "s#$a#$b#g"
  echo "[$(date -Is)] rewrote $a -> $b in $n files; left: $(grep -rlI "$a" "$2/bin" "$2/pyvenv.cfg" | wc -l)"
  du -sh "$2" | cut -f1
  "$2/bin/python" -c "import sys; print(sys.prefix)"
}

pip_vllm() {  # extra pip arguments
  "$BASE/bin/python" -m pip install "vllm==$VERSION" --extra-index-url "https://wheels.vllm.ai/$COMMIT" \
    -c "$CONSTRAINTS" "$@"
}

summary() {  # report.json
  "$BASE/bin/python" - "$1" <<'EOF'
import json, sys
for it in json.load(open(sys.argv[1]))["install"]:
    m, d = it["metadata"], it.get("download_info", {})
    print(f'{m["name"]}=={m["version"]}  {d.get("url", "")}')
EOF
}

case "$STEP" in
  clone-base)
    clone "$SRC" "$BASE"
    ;;
  dry-run)
    # Everything except vllm itself is pinned to what envs/vllm-5ec99956 has, so pip stops with an
    # error if the new wheel needs anything else changed.
    "$SRC/bin/python" -m pip freeze > "$FREEZE_BEFORE"
    grep -E '^[A-Za-z0-9._-]+==' "$FREEZE_BEFORE" | grep -v -i -E '^vllm==' > "$CONSTRAINTS"
    echo "constraints: $(wc -l < "$CONSTRAINTS") of $(wc -l < "$FREEZE_BEFORE") freeze lines"
    grep -v -E '^[A-Za-z0-9._-]+==' "$FREEZE_BEFORE" || true
    pip_vllm --dry-run --report "$LOG/vllm_2f70c32d_dryrun_report.json" 2>&1 | tail -n 15
    echo "--- would install"; summary "$LOG/vllm_2f70c32d_dryrun_report.json"
    ;;
  install)
    pip_vllm --report "$LOG/vllm_2f70c32d_install_report.json" 2>&1 | tail -n 15
    echo "--- installed"; summary "$LOG/vllm_2f70c32d_install_report.json"
    "$BASE/bin/python" -m pip check || echo "PIP_CHECK_REPORTED_CONFLICTS"
    "$BASE/bin/python" -m pip freeze > "$LOG/env_vllm_2f70c32d_freeze.txt"
    echo "--- freeze: envs/vllm-5ec99956 vs envs/vllm-2f70c32d"
    diff "$FREEZE_BEFORE" "$LOG/env_vllm_2f70c32d_freeze.txt" || true
    echo "--- envs/vllm-5ec99956 freeze now: $("$SRC/bin/python" -m pip freeze | sha256sum | cut -c1-16) before: $(sha256sum < "$FREEZE_BEFORE" | cut -c1-16)"
    CUDA_VISIBLE_DEVICES= "$BASE/bin/python" -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
    ;;
  clone-pr)
    clone "$BASE" "$PR"
    ;;
  overlay)
    D=${3:?dir with base.sha256, head.sha256 and head/}
    echo "--- installed files against main @ 2f70c32d"
    ( cd "$PR/$SP" && sha256sum -c "$D/base.sha256" )
    while read -r _ f; do f=${f#\*}; cp "$D/head/$f" "$PR/$SP/$f"; done < "$D/head.sha256"
    echo "--- after the overlay against the PR head e4ee672b"
    ( cd "$PR/$SP" && sha256sum -c "$D/head.sha256" )
    ;;
  check)
    echo "--- files that differ between the two environments (vllm package)"
    diff -rq -x __pycache__ "$BASE/$SP/vllm" "$PR/$SP/vllm" || true
    echo "freeze: base $("$BASE/bin/python" -m pip freeze | sha256sum | cut -c1-16) pr $("$PR/bin/python" -m pip freeze | sha256sum | cut -c1-16) source $("$SRC/bin/python" -m pip freeze | sha256sum | cut -c1-16)"
    for e in "$BASE" "$PR"; do
      CUDA_VISIBLE_DEVICES= "$e/bin/python" -c "
import sys, vllm
import vllm.v1.attention.backends.gdn_attn, vllm.v1.kv_cache_interface
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn
import vllm.third_party.flash_linear_attention.ops.chunk, vllm.third_party.flash_linear_attention.ops.chunk_delta_h
print(sys.prefix, vllm.__version__, 'imports ok')"
    done
    ;;
  *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac
echo "[$(date -Is)] STEP_DONE $STEP"
