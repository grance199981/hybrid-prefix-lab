#!/usr/bin/env bash
# E12 A: the cost of PR #61282, main and the port interleaved A-B-B-A on one idle GPU.
# Each segment runs three workloads through run_e10.sh (one arm, logging hook OFF: the hook writes
# events from the scheduler and would add host time to what is measured here):
#   S  synthetic 12-turn sessions, 16 at a time, output 16 and 256, pools E12A_S_CELLS
#   U  single-turn synthetic prompts (no prefix reuse), 16 at a time, output 256, pools E12A_U_CELLS
#   L  recorded SWE-smith sessions, 8 workers, output 16, pools E12A_L_CELLS
# Segment order: E12A_ORDER (default "main pr61282 pr61282 main"). Environments: E10_ENV_<arm>.
# The gate (E5_GATE_ANY_USER etc.) comes from the environment and applies before every cell.
# Usage: run_e12a.sh <run_dir>     (cwd = source root; env set by the launcher)
set -uo pipefail
RUN_DIR="$1"
ORDER="${E12A_ORDER:-main pr61282 pr61282 main}"
WORKLOADS="${E12A_WORKLOADS:-S U L}"
log() { echo "[$(date '+%F %T %Z')] $*"; }
STATUS=0
i=0
for ARM in $ORDER; do
  i=$((i + 1))
  for W in $WORKLOADS; do
    D="$RUN_DIR/seg${i}_${W}"
    mkdir -p "$D"
    case "$W" in
      S) envs=(E5_STAGE=S E5_WORKLOAD=synth E5_TURNS=12 E5_P0=2000 E5_STEP=250 E5_MIN_TURNS=2
               E5_INFLIGHT=16 E5_LIVE=16 E5_N_SESSIONS=16 E5_OUT_LENS="16 256" E5_CELLS="${E12A_S_CELLS:?}") ;;
      U) envs=(E5_STAGE=U E5_WORKLOAD=synth E5_TURNS=1 E5_P0="${E12A_U_P0:-3375}" E5_STEP=250 E5_MIN_TURNS=1
               E5_INFLIGHT=16 E5_LIVE=16 E5_N_SESSIONS="${E12A_U_N:-192}" E5_OUT_LENS=256 E5_CELLS="${E12A_U_CELLS:?}") ;;
      L) envs=(E5_STAGE=L E5_WORKLOAD=smith E5_THINKING=on E5_INFLIGHT=8 E5_LIVE=8 E5_N_SESSIONS=24
               E5_OUT_LENS=16 E5_CELLS="${E12A_L_CELLS:?}") ;;
      *) log "unknown workload $W"; exit 2 ;;
    esac
    log "segment $i arm $ARM workload $W -> $D"
    env "${envs[@]}" E5_MODE=closed E10_ARMS="$ARM" E10_HOOK=0 bash experiments/e10_state_eviction/run_e10.sh "$D"
    rc=$?
    log "segment $i arm $ARM workload $W finished, status=$rc"
    if [ $rc -ne 0 ]; then STATUS=$rc; log "stopping: a segment failed, the A-B-B-A order is broken"; break 2; fi
  done
done
cd "$RUN_DIR" || exit 1
tar -czf e5_results.tar.gz $(ls -d seg*/*/e5 2>/dev/null) 2>/dev/null
tar -czf serve_logs.tar.gz $(ls seg*/*/serve_*.log seg*/*/serve_*_facts.txt seg*/*/nvidia_smi_*.txt 2>/dev/null) 2>/dev/null
echo "{\"status\": $STATUS, \"finished_at\": \"$(date '+%F %T %Z')\"}" > metrics.json
log "E12 A finished, status=$STATUS"
exit $STATUS
