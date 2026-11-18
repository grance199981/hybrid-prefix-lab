#!/usr/bin/env bash
# E10: the E5 launcher once per arm, with the logging hook loaded into the vLLM server.
# Each arm runs in its own sub-directory of the run directory; then one set of archives
# for the launcher's fetch.
#
#   E10_ARMS          arm names, space separated (default "main")
#   E10_ENV_<arm>     the venv of that arm (default: $HP_ENV)
#   E10_HOOK          1 (default): load hook/sitecustomize.py into the server; 0: plain run
#   E10_STOP_ON_FAIL  1 (default): stop after the first arm that exits non-zero
#
# The hook reaches only the server: run_e5.sh starts the client with `python -I`, which
# ignores PYTHONPATH. Cells, seeds, workload and gate come from the E5_* variables, which
# this script does not change. run_e5.sh itself is not modified.
# Usage: run_e10.sh <run_dir>     (cwd = source root; env set by the launcher)
set -uo pipefail

RUN_DIR="$1"
ARMS="${E10_ARMS:-main}"
HOOK_SRC="$PWD/experiments/e10_state_eviction/hook"
STATUS=0
log() { echo "[$(date '+%F %T %Z')] $*"; }
test -s "$HOOK_SRC/sitecustomize.py" || { log "missing $HOOK_SRC/sitecustomize.py"; exit 2; }

for ARM in $ARMS; do
  v="E10_ENV_$ARM"
  ENV_DIR="${!v:-$HP_ENV}"
  test -x "$ENV_DIR/bin/vllm" || { log "arm $ARM: no vllm in $ENV_DIR"; STATUS=2; break; }
  mkdir -p "$RUN_DIR/$ARM/hook"
  log "E10 arm $ARM starts; env $ENV_DIR; hook ${E10_HOOK:-1}"
  if [ "${E10_HOOK:-1}" = 1 ]; then
    HP_ENV="$ENV_DIR" E10_HOOK_DIR="$RUN_DIR/$ARM/hook" PYTHONPATH="$HOOK_SRC${PYTHONPATH:+:$PYTHONPATH}" \
      bash experiments/e5_cache_pressure/run_e5.sh "$RUN_DIR/$ARM"
  else
    HP_ENV="$ENV_DIR" bash experiments/e5_cache_pressure/run_e5.sh "$RUN_DIR/$ARM"
  fi
  rc=$?
  log "E10 arm $ARM finished, status=$rc; hook files: $(ls "$RUN_DIR/$ARM/hook" | tr '\n' ' ')"
  if [ $rc -ne 0 ]; then
    STATUS=$rc
    [ "${E10_STOP_ON_FAIL:-1}" = 1 ] && { log "stopping after the first failed arm"; break; }
  fi
done

cd "$RUN_DIR" || exit 1
tar -czf e5_results.tar.gz $(ls -d */e5 2>/dev/null) 2>/dev/null
tar -czf hook_logs.tar.gz $(ls -d */hook 2>/dev/null) 2>/dev/null
tar -czf serve_logs.tar.gz $(ls */serve_*.log */serve_*_facts.txt */serve_*_startup_errors.txt */nvidia_smi_*.txt 2>/dev/null) 2>/dev/null
echo "{\"status\": $STATUS, \"finished_at\": \"$(date '+%F %T %Z')\"}" > metrics.json
log "E10 finished, status=$STATUS"
exit $STATUS
