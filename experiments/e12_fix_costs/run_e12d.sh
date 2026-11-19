#!/usr/bin/env bash
# E12 D (and C on the GPU): a list of run_e10.sh jobs, one after another, each with its own E5_* / E10_*
# settings on top of the launcher's environment. Hit-rate runs: the hook stays on (E10_HOOK=1).
#   E12D_JOBS  jobs separated by ";", each "name:VAR=value,VAR=value,..." (values without "," or ";";
#              a space inside a value is written as "+", e.g. E5_OUT_LENS=16+256)
#   E12D_STOP_ON_FAIL  1 (default): stop after the first job that exits non-zero
# Usage: run_e12d.sh <run_dir>     (cwd = source root; env set by the launcher)
set -uo pipefail
RUN_DIR="$1"
log() { echo "[$(date '+%F %T %Z')] $*"; }
STATUS=0
IFS=';' read -r -a JOBS <<< "${E12D_JOBS:?}"
for JOB in "${JOBS[@]}"; do
  JOB="$(echo "$JOB" | xargs)"
  [ -z "$JOB" ] && continue
  NAME="${JOB%%:*}"; SPEC="${JOB#*:}"
  envs=()
  IFS=',' read -r -a KV <<< "$SPEC"
  for kv in "${KV[@]}"; do envs+=("${kv//+/ }"); done
  log "job $NAME: ${envs[*]}"
  env "${envs[@]}" bash experiments/e10_state_eviction/run_e10.sh "$RUN_DIR/$NAME"
  rc=$?
  log "job $NAME finished, status=$rc"
  if [ $rc -ne 0 ]; then
    STATUS=$rc
    [ "${E12D_STOP_ON_FAIL:-1}" = 1 ] && { log "stopping after the first failed job"; break; }
  fi
done
cd "$RUN_DIR" || exit 1
tar -czf e5_results.tar.gz $(ls -d */*/e5 2>/dev/null) 2>/dev/null
tar -czf hook_logs.tar.gz $(ls -d */*/hook 2>/dev/null) 2>/dev/null
tar -czf serve_logs.tar.gz $(ls */*/serve_*.log */*/serve_*_facts.txt */*/nvidia_smi_*.txt 2>/dev/null) 2>/dev/null
echo "{\"status\": $STATUS, \"finished_at\": \"$(date '+%F %T %Z')\"}" > metrics.json
log "E12 D finished, status=$STATUS"
exit $STATUS
