#!/usr/bin/env bash
# Steps 7 to 9: the reproducer without vLLM (graph_repro.py) in several modes, one after the other.
#   control      no profiler, 2 groups, rank 1 reports a wrong checksum at step 5; the checks must find it
#   plain        no profiler
#   nsysnograph  nsys launch --trace=cuda,nvtx, collection running
#   nsys         nsys launch --trace=cuda,nvtx --cuda-graph-trace=node, collection running (the setting that
#                faults in vLLM with Nsight Systems 2025.1.3)
#   nsysnoevent  the nsys mode plus --cuda-event-trace=false
#   nsysnew      the nsys mode with another Nsight Systems binary, E9_NSYS_NEW
# A mode name may carry a suffix after "@" (nsys@1, nsys@2) to run the same mode more than once.
# The two ranks see both GPUs (rank 0 uses the first, rank 1 the second), as vLLM's workers do.
# A mode that fails does not stop the run; it is not retried. Before every mode both GPUs are checked.
#
# Environment: HP_ROOT, HP_ENV (virtualenv with torch and its NCCL), E9_GPU_A, E9_GPU_B,
#   E9_NSYS (nsys), E9_NSYS_NEW, E9_MODES ("control plain nsysnograph nsys"), E9_REPRO_GROUPS (60),
#   E9_REPRO_STEPS (128), E9_REPRO_ARGS (extra arguments for every mode, e.g. "--async-output 1"),
#   GIVE_UP_HOURS (1), E9_GATE_ANY_USER (1), E9_GATE_MAX_UTIL (5), E9_GATE_MAX_MEM (2048).
# Outputs: RUN_DIR/<mode>/ (graph_repro.py, trace.nsys-rep, nsys.log, parent.log), summary.json.
# Exit codes: 0 every mode ended as expected, 1 some mode did not, 3 gate never opened, 4 a GPU got busy,
#             5 a GPU looks unhealthy.
# Usage: experiments/e9_two_gpu_debug/run_graph_repro.sh RUN_DIR
#        then: python experiments/e9_two_gpu_debug/summarize.py repro RUN_DIR/summary.json
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$1"
RUN_DIR="$(cd "$1" && pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
. scripts/serve_lib.sh

PY="${HP_ENV:?}/bin/python"
GPU_A="${E9_GPU_A:?}"
GPU_B="${E9_GPU_B:?}"
NSYS="${E9_NSYS:-nsys}"
NSYS_NEW="${E9_NSYS_NEW:-}"           # a second Nsight Systems binary, used by the mode nsysnew
MODES="${E9_MODES:-control plain nsysnograph nsys}"
GROUPS_N="${E9_REPRO_GROUPS:-60}"
EXTRA_ARGS="${E9_REPRO_ARGS:-}"
STEPS_N="${E9_REPRO_STEPS:-128}"
GIVE_UP_HOURS="${GIVE_UP_HOURS:-1}"
ANY_USER="${E9_GATE_ANY_USER:-1}"
MAX_UTIL="${E9_GATE_MAX_UTIL:-5}"
MAX_MEM="${E9_GATE_MAX_MEM:-2048}"
export TMPDIR="${HP_ROOT:?}/tmp" PYTHONDONTWRITEBYTECODE=1
STATUS=0

free_once() {   # both GPUs pass the gate (both are checked and logged)
  local u ok=0
  for u in "$GPU_A" "$GPU_B"; do
    gpu_idle_once "$u" "$ANY_USER" "$MAX_UTIL" "$MAX_MEM" || ok=1
  done
  return $ok
}

SP=""
stop_run() {      # by PID only: the launched process and everything below it
  [ -n "$SP" ] || return 0
  local p pids="$SP $(descendants "$SP" | tr '\n' ' ')"
  for p in $pids; do kill "$p" 2>/dev/null; done
  sleep 5
  for p in $pids; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; done
  wait "$SP" 2>/dev/null
  SP=""
}
trap 'stop_run' EXIT
trap 'log "terminated"; stop_run; exit 143' TERM INT

log "reproducer waiting for both GPUs ($GPU_A rank 0, $GPU_B rank 1); modes: $MODES; $GROUPS_N groups of $STEPS_N steps"
DEADLINE=$(( $(date +%s) + GIVE_UP_HOURS * 3600 ))
while :; do
  if free_once && sleep 20 && free_once && sleep 20 && free_once; then break; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then log "gate never opened within $GIVE_UP_HOURS h; giving up"; exit 3; fi
  sleep 240
done
log "gate open"
"$NSYS" --version 2>&1 | head -n 1
[ -n "$NSYS_NEW" ] && "$NSYS_NEW" --version 2>&1 | head -n 1
nvidia-smi > "$RUN_DIR/nvidia_smi_before.txt" 2>&1

for MODE in $MODES; do
  if ! free_once; then log "a GPU is not free before $MODE; stopping"; STATUS=4; break; fi
  OUT="$RUN_DIR/$MODE"; mkdir -p "$OUT"
  PRE=(); COLLECT=0; ARGS=(--groups "$GROUPS_N" --steps "$STEPS_N")
  NS="$NSYS"                          # the profiler binary of this mode
  SESS="repro$(date +%s)"
  case "${MODE%%@*}" in
    control)     ARGS=(--groups 2 --steps "$STEPS_N" --inject-step 5 --expect injected) ;;
    plain)       ;;
    nsysnograph) PRE=("$NSYS" launch --session-new="$SESS" --trace=cuda,nvtx); COLLECT=1 ;;
    nsys)        PRE=("$NSYS" launch --session-new="$SESS" --trace=cuda,nvtx --cuda-graph-trace=node); COLLECT=1 ;;
    nsysnoevent) PRE=("$NSYS" launch --session-new="$SESS" --trace=cuda,nvtx --cuda-graph-trace=node --cuda-event-trace=false); COLLECT=1 ;;
    nsysnew)     [ -x "$NSYS_NEW" ] || { log "E9_NSYS_NEW is not an executable: $NSYS_NEW"; STATUS=1; continue; }
                 NS="$NSYS_NEW"; PRE=("$NS" launch --session-new="$SESS" --trace=cuda,nvtx --cuda-graph-trace=node); COLLECT=1 ;;
    *) log "unknown mode $MODE"; STATUS=1; continue ;;
  esac
  log "$MODE command: CUDA_VISIBLE_DEVICES=$GPU_A,$GPU_B NCCL_DEBUG=INFO ${PRE[*]} $PY experiments/e9_two_gpu_debug/graph_repro.py --out $OUT ${ARGS[*]} $EXTRA_ARGS"
  env CUDA_VISIBLE_DEVICES="$GPU_A,$GPU_B" NCCL_DEBUG=INFO "${PRE[@]}" \
    "$PY" experiments/e9_two_gpu_debug/graph_repro.py --out "$OUT" "${ARGS[@]}" $EXTRA_ARGS > "$OUT/parent.log" 2>&1 &
  SP=$!
  log "$MODE started pid $SP"
  n=0
  while [ ! -e "$OUT/ready0" ] || [ ! -e "$OUT/ready1" ]; do   # both ranks warmed up, graph captured
    kill -0 "$SP" 2>/dev/null || break
    n=$((n + 1)); [ $n -gt 180 ] && break
    sleep 2
  done
  if [ -e "$OUT/ready0" ] && [ -e "$OUT/ready1" ]; then
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$OUT/nvidia_smi.txt"
    if [ "$COLLECT" = 1 ]; then
      "$NS" start --session="$SESS" --sample=none --cpuctxsw=none -o "$OUT/trace" --force-overwrite=true >> "$OUT/nsys.log" 2>&1
      log "$MODE nsys start rc=$?"
    fi
    touch "$OUT/go"
    log "$MODE measured part started"
  else
    log "$MODE the ranks did not get ready"
  fi
  n=0
  while kill -0 "$SP" 2>/dev/null; do           # the parent ends its ranks itself when one stalls
    n=$((n + 1)); [ $n -gt 900 ] && { log "$MODE still running after 30 min; stopping it"; break; }
    sleep 2
  done
  kill -0 "$SP" 2>/dev/null && stop_run
  wait "$SP" 2>/dev/null; RC=$?; SP=""
  if [ "$COLLECT" = 1 ]; then
    "$NS" stop --session="$SESS" >> "$OUT/nsys.log" 2>&1
    log "$MODE nsys stop rc=$? report: $(ls -la "$OUT/trace.nsys-rep" 2>/dev/null | awk '{print $5}') bytes"
    "$NS" shutdown --session="$SESS" --kill none >> "$OUT/nsys.log" 2>&1
  fi
  log "$MODE finished, exit=$RC | $(tail -n 1 "$OUT/parent.log" | cut -c1-300)"
  [ $RC -eq 0 ] || STATUS=1
  n=0
  while [ $n -lt 15 ]; do
    read -r a x < <(gpu_procs "$GPU_A"); read -r b x < <(gpu_procs "$GPU_B")
    [ $((a + b)) -eq 0 ] && break
    sleep 2; n=$((n + 1))
  done
  nvidia-smi > "$RUN_DIR/nvidia_smi_after_$MODE.txt" 2>&1 || { log "nvidia-smi failed after $MODE"; STATUS=5; break; }
  if sed '/Processes:/,$d' "$RUN_DIR/nvidia_smi_after_$MODE.txt" | grep -i -E "reset|ERR!|Unknown Error|lost" > /dev/null; then
    log "nvidia-smi reports a problem after $MODE"; STATUS=5; break
  fi
  log "after $MODE: own processes left = $((a + b))"
  sleep 10
done

cd "$RUN_DIR" || exit 1
tar -czf e9_repro_results.tar.gz $(ls */summary.json */rank*.json */rank*.log */parent.log */progress*.txt */nsys.log */nvidia_smi.txt nvidia_smi_*.txt 2>/dev/null) 2>/dev/null
"$PY" -c '
import glob, json
print(json.dumps({p.split("/")[0]: json.load(open(p)) for p in sorted(glob.glob("*/summary.json"))}, indent=1))
' > summary.json
echo "{\"status\": $STATUS, \"finished_at\": \"$(date '+%F %T %Z')\"}" > metrics.json
log "reproducer finished, status=$STATUS"
exit $STATUS
