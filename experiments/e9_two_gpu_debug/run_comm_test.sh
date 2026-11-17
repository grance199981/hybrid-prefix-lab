#!/usr/bin/env bash
# Step 1: the communication test of two GPUs without vLLM (comm_test.py), four groups in order:
#   K  control   NCCL_P2P_DISABLE=1, 20 s, one wrong element sent at steps 2 and 7; the checker must find both
#   A            NCCL_P2P_DISABLE=1
#   B            no NCCL_P2P_* variable (what the library chooses by itself)
#   C            NCCL_P2P_LEVEL=SYS (P2P forced across NUMA nodes)
# All groups set NCCL_DEBUG=INFO, so the rank logs show which transport NCCL used.
#
# Environment: HP_ROOT, HP_ENV (virtualenv with torch), E9_GPU_A (rank 0), E9_GPU_B (rank 1),
#   E9_SECONDS (240), E9_GROUPS ("K A B C"), GIVE_UP_HOURS (1), E9_GATE_SLEEP (60 s between gate checks),
#   E9_GATE_ANY_USER (1: both GPUs free of every user's processes; 0: only of ours),
#   E9_GATE_MAX_UTIL (5 %), E9_GATE_MAX_MEM (2048 MiB).
# Stop rule: the first group that does not end as expected, or a GPU that does not look healthy after a
# group, or a GPU that is not free before a group, ends the run. Nothing is retried.
# Outputs: RUN_DIR/<group>/ (comm_test.py), nvidia-smi snapshots, gpus.csv, summary.json, e9_results.tar.gz.
# Exit codes: 0 ok, 3 gate never opened, 4 a GPU got busy, 5 a GPU looks unhealthy, else the code of comm_test.py.
# Usage: experiments/e9_two_gpu_debug/run_comm_test.sh RUN_DIR
#        then: python experiments/e9_two_gpu_debug/summarize.py comm RUN_DIR
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$1"
RUN_DIR="$(cd "$1" && pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
. scripts/serve_lib.sh

ENVDIR="${HP_ENV:?}"
PY="$ENVDIR/bin/python"
GPU_A="${E9_GPU_A:?}"                 # rank 0
GPU_B="${E9_GPU_B:?}"                 # rank 1
SECS="${E9_SECONDS:-240}"
GROUPS_TO_RUN="${E9_GROUPS:-K A B C}"
GIVE_UP_HOURS="${GIVE_UP_HOURS:-1}"
GATE_SLEEP="${E9_GATE_SLEEP:-60}"
export TMPDIR="${HP_ROOT:?}/tmp" PYTHONDONTWRITEBYTECODE=1
STATUS=0
ANY_USER="${E9_GATE_ANY_USER:-1}"
MAX_UTIL="${E9_GATE_MAX_UTIL:-5}"
MAX_MEM="${E9_GATE_MAX_MEM:-2048}"

free_once() {   # both GPUs pass the gate (both are checked and logged)
  local u ok=0
  for u in "$GPU_A" "$GPU_B"; do
    gpu_idle_once "$u" "$ANY_USER" "$MAX_UTIL" "$MAX_MEM" || ok=1
  done
  return $ok
}

healthy() {     # after a group: both GPUs answer, nothing asks for a reset, our processes are gone
  local tag="$1" n=0 a b x
  while [ $n -lt 15 ]; do
    read -r a x < <(gpu_procs "$GPU_A"); read -r b x < <(gpu_procs "$GPU_B")
    [ $((a + b)) -eq 0 ] && break
    sleep 2; n=$((n + 1))
  done
  nvidia-smi > "$RUN_DIR/nvidia_smi_after_$tag.txt" 2>&1 || { log "nvidia-smi failed after $tag"; return 1; }
  nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader \
    > "$RUN_DIR/nvidia_smi_query_after_$tag.txt" 2>&1
  grep -q "$GPU_A" "$RUN_DIR/nvidia_smi_query_after_$tag.txt" && grep -q "$GPU_B" "$RUN_DIR/nvidia_smi_query_after_$tag.txt" \
    || { log "a GPU is missing from nvidia-smi after $tag"; return 1; }
  # the process table is left out: names of other users' processes are not ours to judge
  if sed '/Processes:/,$d' "$RUN_DIR/nvidia_smi_after_$tag.txt" | grep -i -E "reset|ERR!|Unknown Error|lost" > /dev/null; then
    log "nvidia-smi reports a problem after $tag"; return 1
  fi
  log "after $tag: own processes left = $((a + b)), processes of any user: $x on $GPU_B"
  return 0
}

SAMPLER=""
cleanup() { [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null; SAMPLER=""; }
trap cleanup EXIT
trap 'log "terminated"; cleanup; exit 143' TERM INT

log "comm test waiting for both GPUs ($GPU_A rank 0, $GPU_B rank 1); groups: $GROUPS_TO_RUN; $SECS s each"
log "gate: any-user rule = $ANY_USER, utilization <= $MAX_UTIL %, memory used <= $MAX_MEM MiB"
DEADLINE=$(( $(date +%s) + GIVE_UP_HOURS * 3600 ))
while :; do
  if free_once && sleep "$GATE_SLEEP" && free_once && sleep "$GATE_SLEEP" && free_once; then break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then log "gate never opened within $GIVE_UP_HOURS h; giving up"; exit 3; fi
  sleep 240
done
log "gate open"
nvidia-smi > "$RUN_DIR/nvidia_smi_before.txt" 2>&1
"$PY" -c 'import torch; print("torch", torch.__version__, "nccl", torch.cuda.nccl.version())' \
  > "$RUN_DIR/versions.txt" 2>&1
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader -l 2 > "$RUN_DIR/gpus.csv" 2>/dev/null &
SAMPLER=$!

for G in $GROUPS_TO_RUN; do
  if ! free_once; then log "a GPU is not free before group $G; stopping"; STATUS=4; break; fi
  EXTRA=(); ARGS=(--seconds "$SECS")
  case "$G" in
    K) EXTRA=(NCCL_P2P_DISABLE=1); ARGS=(--seconds 20 --inject-steps 2,7 --expect injected) ;;
    A) EXTRA=(NCCL_P2P_DISABLE=1) ;;
    B) EXTRA=() ;;
    C) EXTRA=(NCCL_P2P_LEVEL=SYS) ;;
    *) log "unknown group $G"; STATUS=2; break ;;
  esac
  log "group $G starts: ${EXTRA[*]:-no NCCL_P2P variable}"
  env -u NCCL_P2P_DISABLE -u NCCL_P2P_LEVEL NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES="$GPU_A,$GPU_B" "${EXTRA[@]}" \
    "$PY" experiments/e9_two_gpu_debug/comm_test.py --out "$RUN_DIR/$G" --tag "$G" --backend nccl "${ARGS[@]}"
  RC=$?
  log "group $G finished, exit=$RC"
  if ! healthy "$G"; then STATUS=5; break; fi
  if [ $RC -ne 0 ]; then log "group $G did not end as expected; the later groups are not started"; STATUS=$RC; break; fi
  sleep 10
done

cleanup
cd "$RUN_DIR" || exit 1
tar -czf e9_results.tar.gz $(ls -d [KABC] nvidia_smi_*.txt versions.txt gpus.csv 2>/dev/null) 2>/dev/null
"$PY" -c '
import glob, json
print(json.dumps({p.split("/")[0]: json.load(open(p)) for p in sorted(glob.glob("[KABC]/summary.json"))}, indent=1))
' > summary.json
echo "{\"status\": $STATUS, \"finished_at\": \"$(date '+%F %T %Z')\"}" > metrics.json
log "comm test finished, status=$STATUS"
exit $STATUS
