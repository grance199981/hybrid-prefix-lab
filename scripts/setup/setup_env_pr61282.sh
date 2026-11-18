#!/usr/bin/env bash
# The environment for E10 step 3: envs/vllm-pr61282 = a copy of envs/vllm-main with the two runtime
# files of PR #61282 (ported onto 1479deb0 by experiments/e10_state_eviction/port_pr61282.py) on top.
# Nothing is downloaded or installed; envs/vllm-main is only read. Every step refuses to overwrite.
#
# Usage: bash setup_env_pr61282.sh <remote_root> <step> [arg]
#   clone | overlay <dir with ours.sha256, port.sha256, port/> | check
set -euo pipefail

ROOT=${1:?usage: setup_env_pr61282.sh <remote_root> <step> [arg]}
STEP=${2:?step}
SRC=$ROOT/envs/vllm-main
PR=$ROOT/envs/vllm-pr61282
SP=lib/python3.12/site-packages

case "$STEP" in
  clone)
    [ -e "$PR" ] && { echo "refusing to overwrite existing env: $PR" >&2; exit 2; }
    echo "[$(date -Is)] cp -a $SRC $PR"
    ionice -c3 nice -n 19 cp -a "$SRC" "$PR"
    a="envs/vllm-main"; b="envs/vllm-pr61282"
    n=$(grep -rlI "$a" "$PR/bin" "$PR/pyvenv.cfg" | wc -l)
    grep -rlI --null "$a" "$PR/bin" "$PR/pyvenv.cfg" | xargs -0 sed -i "s#$a#$b#g"
    echo "[$(date -Is)] rewrote $a -> $b in $n files; left: $(grep -rlI "$a" "$PR/bin" "$PR/pyvenv.cfg" | wc -l)"
    du -sh "$PR" | cut -f1
    "$PR/bin/python" -c "import sys; print(sys.prefix)"
    ;;
  overlay)
    D=${3:?dir with ours.sha256, port.sha256 and port/}
    echo "--- installed files against 1479deb0"
    ( cd "$PR/$SP" && sha256sum -c "$D/ours.sha256" )
    while read -r _ f; do f=${f#\*}; cp "$D/port/$f" "$PR/$SP/$f"; done < "$D/port.sha256"
    echo "--- after the overlay against the port of PR #61282"
    ( cd "$PR/$SP" && sha256sum -c "$D/port.sha256" )
    ;;
  check)
    echo "--- files that differ between the two environments (vllm package)"
    diff -rq -x __pycache__ "$SRC/$SP/vllm" "$PR/$SP/vllm" || true
    echo "freeze: main $("$SRC/bin/python" -m pip freeze | sha256sum | cut -c1-16) pr61282 $("$PR/bin/python" -m pip freeze | sha256sum | cut -c1-16)"
    for e in "$SRC" "$PR"; do
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
