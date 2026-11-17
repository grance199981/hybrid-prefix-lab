#!/usr/bin/env bash
# Two environments for PR #61952 (E8), built from envs/vllm-main (scripts/setup/setup_env.sh):
#   envs/vllm-5ec99956  copy of envs/vllm-main with the per-commit wheel of main @ 5ec99956 (the base of PR #61952)
#   envs/vllm-pr61952   copy of the first one with the 13 Python files of the PR head 6891ee3e on top
# Every other package stays at the version envs/vllm-main has (pip stops if the wheel
# needs more than vllm, xgrammar and flashinfer-python changed). envs/vllm-main and the
# other environments are only read. Every step refuses to overwrite.
#
# Usage: bash scripts/setup/setup_env_pr61952.sh $HP_ROOT <step> [arg], steps in this order:
#   clone-base | dry-run | install | clone-pr | overlay <dir> | check
# overlay <dir>: a directory with base.sha256 (the 13 files as installed at 5ec99956),
# head.sha256 (the same files at the PR head) and head/ (the files at the PR head, by path
# below site-packages); the step checks the installed files against base.sha256, copies
# head/ over them and checks the result against head.sha256.
set -euo pipefail

ROOT=${1:?usage: setup_env_pr61952.sh <remote_root> <step> [arg]}
STEP=${2:?step}
SRC=$ROOT/envs/vllm-main
BASE=$ROOT/envs/vllm-5ec99956
PR=$ROOT/envs/vllm-pr61952
COMMIT=5ec99956ad4253e70a56c8c0179c5af14fef377d
VERSION="0.34.0rc1.dev491+g5ec99956a"
LOG=$ROOT/logs
SP=lib/python3.12/site-packages

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
    -c "$LOG/vllm_5ec99956_constraints.txt" "$@"
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
    # Everything except the three packages that are expected to move is pinned to what envs/vllm-main has,
    # so pip stops with an error if the new wheel needs anything else changed.
    "$SRC/bin/python" -m pip freeze > "$LOG/env_vllm_main_freeze_before_5ec99956.txt"
    grep -E '^[A-Za-z0-9._-]+==' "$LOG/env_vllm_main_freeze_before_5ec99956.txt" \
      | grep -v -i -E '^(vllm|xgrammar|flashinfer-python)==' > "$LOG/vllm_5ec99956_constraints.txt"
    echo "constraints: $(wc -l < "$LOG/vllm_5ec99956_constraints.txt") of $(wc -l < "$LOG/env_vllm_main_freeze_before_5ec99956.txt") freeze lines"
    grep -v -E '^[A-Za-z0-9._-]+==' "$LOG/env_vllm_main_freeze_before_5ec99956.txt" || true
    pip_vllm --dry-run --report "$LOG/vllm_5ec99956_dryrun_report.json" 2>&1 | tail -n 15
    echo "--- would install"; summary "$LOG/vllm_5ec99956_dryrun_report.json"
    ;;
  install)
    pip_vllm --report "$LOG/vllm_5ec99956_install_report.json" 2>&1 | tail -n 15
    echo "--- installed"; summary "$LOG/vllm_5ec99956_install_report.json"
    "$BASE/bin/python" -m pip check || echo "PIP_CHECK_REPORTED_CONFLICTS"
    "$BASE/bin/python" -m pip freeze > "$LOG/env_vllm_5ec99956_freeze.txt"
    echo "--- freeze: envs/vllm-main vs envs/vllm-5ec99956"
    diff "$LOG/env_vllm_main_freeze_before_5ec99956.txt" "$LOG/env_vllm_5ec99956_freeze.txt" || true
    echo "--- envs/vllm-main freeze now: $("$SRC/bin/python" -m pip freeze | sha256sum | cut -c1-16) before: $(sha256sum < "$LOG/env_vllm_main_freeze_before_5ec99956.txt" | cut -c1-16)"
    CUDA_VISIBLE_DEVICES= "$BASE/bin/python" -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
    ;;
  clone-pr)
    clone "$BASE" "$PR"
    ;;
  overlay)
    D=${3:?dir with base.sha256, head.sha256 and head/}
    echo "--- installed files against main @ 5ec99956"
    ( cd "$PR/$SP" && sha256sum -c "$D/base.sha256" )
    while read -r _ f; do f=${f#\*}; cp "$D/head/$f" "$PR/$SP/$f"; done < "$D/head.sha256"
    echo "--- after the overlay against the PR head 6891ee3e"
    ( cd "$PR/$SP" && sha256sum -c "$D/head.sha256" )
    ;;
  check)
    echo "--- files that differ between the two environments (vllm package)"
    diff -rq -x __pycache__ "$BASE/$SP/vllm" "$PR/$SP/vllm" || true
    echo "freeze: base $("$BASE/bin/python" -m pip freeze | sha256sum | cut -c1-16) pr $("$PR/bin/python" -m pip freeze | sha256sum | cut -c1-16) main $("$SRC/bin/python" -m pip freeze | sha256sum | cut -c1-16)"
    for e in "$BASE" "$PR"; do
      CUDA_VISIBLE_DEVICES= "$e/bin/python" -c "
import sys, vllm
import vllm.compilation.breakable_cudagraph, vllm.v1.attention.backends.gdn_attn
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn, vllm.third_party.flash_linear_attention.ops.chunk
print(sys.prefix, vllm.__version__, 'imports ok')"
    done
    ;;
  *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac
echo "[$(date -Is)] STEP_DONE $STEP"
