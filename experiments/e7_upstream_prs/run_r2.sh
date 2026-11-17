#!/usr/bin/env bash
# Round R2 of E7: the cache-pressure launcher (experiments/e5_cache_pressure/run_e5.sh) once per
# arm of the ported PR #66881, each arm in its own sub-directory of the run directory, then one set
# of archives over all arms.
#
#   R2_ARMS            arm names, space separated (default "P0 P1 P2")
#   R2_EXTRA_<arm>     extra `vllm serve` arguments of that arm (empty = none), e.g.
#                        R2_EXTRA_P1="--mamba-frontier-checkpoint"
#                        R2_EXTRA_P2="--mamba-frontier-checkpoint --prefix-cache-eviction gds
#                                     --prefill-cost-coefficients 5.06e-5 1.16e-3"
#   R2_STOP_ON_FAIL    1 (default): stop after the first arm that exits non-zero
#
# Everything else (HP_*, cells, seeds, workload, gate) is read by run_e5.sh from its E5_* and GATE_*
# variables, which this script does not change. HP_ENV must be the environment with the ported PR.
# Outputs: RUN_DIR/<arm>/ as written by run_e5.sh, RUN_DIR/e5_results.tar.gz, serve_logs.tar.gz,
# metrics.json ({"status": ...}).
# Usage: bash experiments/e7_upstream_prs/run_r2.sh <run_dir>
set -uo pipefail

mkdir -p "${1:?usage: run_r2.sh <run_dir>}" && RUN_DIR="$(cd "$1" && pwd)" || exit 2
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$REPO/scripts/serve_lib.sh"
ARMS="${R2_ARMS:-P0 P1 P2}"
STATUS=0

for ARM in $ARMS; do
  v="R2_EXTRA_$ARM"
  EXTRA="${!v:-}"
  mkdir -p "$RUN_DIR/$ARM"
  log "R2 arm $ARM starts; extra serve arguments: ${EXTRA:-none}"
  E5_SERVE_EXTRA="$EXTRA" bash "$REPO/experiments/e5_cache_pressure/run_e5.sh" "$RUN_DIR/$ARM"
  rc=$?
  log "R2 arm $ARM finished, status=$rc"
  if [ $rc -ne 0 ]; then
    STATUS=$rc
    [ "${R2_STOP_ON_FAIL:-1}" = 1 ] && { log "stopping after the first failed arm"; break; }
  fi
done

cd "$RUN_DIR" || exit 1
tar -czf e5_results.tar.gz $(ls -d P*/e5 2>/dev/null) 2>/dev/null
tar -czf serve_logs.tar.gz $(ls P*/serve_*.log P*/serve_*_facts.txt P*/serve_*_startup_errors.txt P*/nvidia_smi_*.txt 2>/dev/null) 2>/dev/null
echo "{\"status\": $STATUS, \"finished_at\": \"$(date '+%F %T %Z')\"}" > metrics.json
log "R2 finished, status=$STATUS"
exit $STATUS
