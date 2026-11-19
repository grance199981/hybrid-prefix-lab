#!/usr/bin/env bash
# Two environments for E12 C (the rebase of PR #61282 onto current main):
#   envs/vllm-191565a2           copy of envs/vllm-2f70c32d with the per-commit wheel of main @ 191565a2
#                                (the newest main commit that had a wheel on 2026-11-18)
#   envs/vllm-pr61282-rebased    copy of the first one with the two rebased files of PR #61282 on top
# envs/vllm-2f70c32d and the other environments are only read. Every step refuses to overwrite.
# Same steps as setup_env_pr65186_head.sh.
#
# Usage: bash setup_env_main_191565a2.sh <remote_root> <step> [arg]
#   clone-base | dry-run | install | clone-pr | overlay <dir with base.sha256, port.sha256, port/> | check
set -euo pipefail

ROOT=${1:?usage: setup_env_main_191565a2.sh <remote_root> <step> [arg]}
STEP=${2:?step}
SRC=$ROOT/envs/vllm-2f70c32d
BASE=$ROOT/envs/vllm-191565a2
PR=$ROOT/envs/vllm-pr61282-rebased
COMMIT=191565a2cd0c42a1e4fd4d063f53d0ce383b364f
VERSION="0.34.0rc1.dev816+g191565a2c"
LOG=$ROOT/logs
SP=lib/python3.12/site-packages
FREEZE_BEFORE=$LOG/env_vllm_2f70c32d_freeze_before_191565a2.txt
CONSTRAINTS=$LOG/vllm_191565a2_constraints.txt

export PIP_CACHE_DIR=$ROOT/cache/pip
export TMPDIR=$ROOT/tmp
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_INDEX_URL=${PIP_INDEX_MIRROR:-https://mirrors.aliyun.com/pypi/simple}
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
    # Everything except vllm itself is pinned to what envs/vllm-2f70c32d has, so pip stops with an
    # error if the new wheel needs anything else changed (then report, do not loosen silently).
    "$SRC/bin/python" -m pip freeze > "$FREEZE_BEFORE"
    # UNPIN: extra packages (regex alternatives) that the new wheel may move, e.g. "humming-kernels";
    # on 2026-11-18 vllm @ 191565a2 needed humming-kernels 0.1.20, and cuda-toolkit 13.0.3 needed
    # nvidia-cuda-cccl 13.0.85, so the run used UNPIN='humming-kernels|cuda-toolkit|nvidia-[a-z0-9-]+'.
    grep -E '^[A-Za-z0-9._-]+==' "$FREEZE_BEFORE" | grep -v -i -E "^(vllm${UNPIN:+|$UNPIN})==" > "$CONSTRAINTS"
    echo "constraints: $(wc -l < "$CONSTRAINTS") of $(wc -l < "$FREEZE_BEFORE") freeze lines"
    pip_vllm --dry-run --report "$LOG/vllm_191565a2_dryrun_report.json" 2>&1 | tail -n 15
    echo "--- would install"; summary "$LOG/vllm_191565a2_dryrun_report.json"
    ;;
  install)
    pip_vllm --report "$LOG/vllm_191565a2_install_report.json" 2>&1 | tail -n 15
    echo "--- installed"; summary "$LOG/vllm_191565a2_install_report.json"
    "$BASE/bin/python" -m pip check || echo "PIP_CHECK_REPORTED_CONFLICTS"
    "$BASE/bin/python" -m pip freeze > "$LOG/env_vllm_191565a2_freeze.txt"
    echo "--- freeze: envs/vllm-2f70c32d vs envs/vllm-191565a2"
    diff "$FREEZE_BEFORE" "$LOG/env_vllm_191565a2_freeze.txt" || true
    echo "--- envs/vllm-2f70c32d freeze now: $("$SRC/bin/python" -m pip freeze | sha256sum | cut -c1-16) before: $(sha256sum < "$FREEZE_BEFORE" | cut -c1-16)"
    CUDA_VISIBLE_DEVICES= "$BASE/bin/python" -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
    ;;
  clone-pr)
    clone "$BASE" "$PR"
    ;;
  overlay)
    D=${3:?dir with base.sha256, port.sha256 and port/}
    echo "--- installed files against main @ 191565a2"
    ( cd "$PR/$SP" && sha256sum -c "$D/base.sha256" )
    while read -r _ f; do f=${f#\*}; cp "$D/port/$f" "$PR/$SP/$f"; done < "$D/port.sha256"
    echo "--- after the overlay against the rebased PR #61282"
    ( cd "$PR/$SP" && sha256sum -c "$D/port.sha256" )
    ;;
  check)
    echo "--- files that differ between the two environments (vllm package)"
    diff -rq -x __pycache__ "$BASE/$SP/vllm" "$PR/$SP/vllm" || true
    echo "freeze: base $("$BASE/bin/python" -m pip freeze | sha256sum | cut -c1-16) pr $("$PR/bin/python" -m pip freeze | sha256sum | cut -c1-16)"
    for e in "$BASE" "$PR"; do
      CUDA_VISIBLE_DEVICES= "$e/bin/python" -c "
import sys, vllm
import vllm.v1.core.kv_cache_coordinator as c, vllm.v1.core.single_type_kv_cache_manager as m
import vllm.v1.core.sched.scheduler
print(sys.prefix, vllm.__version__, 'imports ok', 'boundary hold:', hasattr(m.MambaManager, '_retain_boundary_block'))"
    done
    ;;
  *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac
echo "[$(date -Is)] STEP_DONE $STEP"
