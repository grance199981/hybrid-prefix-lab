#!/usr/bin/env bash
# One server at a time, a few controlled requests each, optionally under Nsight Systems.
# Used by E4 (profiling Q1, Q2, Q2b, Q3), the cold-prompt probes of E7 and E8, and the
# vLLM steps of E9. For each server config and each mode: wait until the GPU(s) are
# free, start `vllm serve`, optionally run a correctness check, send warm-up requests,
# start the profiler collection (nsys modes), send the measured requests with
# experiments/e4_profiling/probe_client.py, stop the collection, stop the server by PID,
# then export the trace to SQLite and analyse it on the CPU.
#
# Usage: scripts/run_probe.sh RUN_DIR
# Outputs go to RUN_DIR/probe/ (<cfg>_<mode>.jsonl request records, <cfg>_<mode>_warmup.jsonl,
# .nsys-rep / .sqlite traces, .analysis.json, client / nsys / analysis logs, <cfg>_<mode>_check.json)
# and RUN_DIR/ (serve_<cfg>_<mode>.log, nvidia-smi snapshots, placement_<tag>.txt);
# probe_results.tar.gz and serve_logs.tar.gz pack the small ones; metrics.json holds the status.
#
# Required environment
#   HP_ROOT          experiment root on the GPU server (caches and tmp go below it)
#   HP_ENV           virtualenv with vLLM (bin/python, bin/vllm)
#   GATE_GPU_UUID    the GPU to use (rank 0 of two-GPU configs, unless *_T2ns)
#   HP_QWEN35_4B_DIR, HP_QWEN35_9B_DIR, HP_QWEN3_DIR   model directories
#   PROBE_CONFIGS    server configs, see below          PROBE_LENGTHS  prompt lengths
# Optional (default)
#   GATE_GPU2_UUID   second GPU, for *_T2* configs
#   PROBE_MODES ("plain nsys")   PROBE_REPS (5)   PROBE_WARMUP (1)   PROBE_OUT_LEN (4: output tokens)
#   PROBE_CLIENT (experiments/e4_profiling/probe_client.py)   PROBE_CLIENT_ARGS (""), e.g.
#                    "--pattern groups --batches 64 --decode-tokens 128"
#   PROBE_ANALYZE (experiments/e4_profiling/analyze_q1.py)    analysis of an nsys / nsysnew trace
#   PROBE_NSYS (nsys)  the Nsight Systems binary;   PROBE_NSYS_NEW  a second binary, mode nsysnew
#   PROBE_NVML (0)     1: sample NVML during the plain runs (nvml_sampler.py)
#   PROBE_SCALING (0)  1: run q2_scaling.py first, before any server; PROBE_SCALING_ARGS ("")
#   PROBE_MAX_LEN (32768)  PROBE_KV_BYTES_4B (24 GiB)  PROBE_KV_BYTES_9B (16 GiB)  PROBE_PORT (8061)
#   PROBE_SERVE_EXTRA  extra `vllm serve` arguments for every config, split on spaces
#   PROBE_CHECK      a script run against every server before any measurement (e.g.
#                    experiments/e7_upstream_prs/check_cache_correctness.py), called with
#                    --url --model --tokenizer-dir --json and PROBE_CHECK_ARGS; non-zero exit:
#                    no measurements on that server
#   PROBE_STARTUP_TRIES (360)  health checks, 5 s apart, before a server counts as failed
#   PROBE_STOP_ON_FAIL (0)     1: never start another server after a failure
#   GIVE_UP_HOURS (2)          how long to wait for the gate
#
# Server configs (all bf16, prefix caching on, --seed 0):
#   h4_C0  Qwen3.5-4B defaults             h4_C1  Qwen3.5-4B --prefix-match-unit 16
#   h4_C64 Qwen3.5-4B --prefix-match-unit 64
#   h9_C0  Qwen3.5-9B defaults             h9_C1  Qwen3.5-9B --prefix-match-unit 16
#   fa_F0  Qwen3-4B defaults               fa_F16 Qwen3-4B --prefix-match-unit 16
#   h4_T2, fa_T2     --tensor-parallel-size 2 on GATE_GPU_UUID and GATE_GPU2_UUID
#   h4_T2n, fa_T2n   the same with --disable-custom-all-reduce (all-reduce through NCCL)
#   fa_T2ns          fa_T2n with the two GPUs in the other order (rank 0 on GATE_GPU2_UUID)
#
# Modes (what each changes against plain):
#   plain        no profiler, default worker start method, no NVTX scopes
#   nsys         VLLM_NVTX_SCOPES_FOR_PROFILING=1, VLLM_WORKER_MULTIPROC_METHOD=spawn (so the
#                profiler follows the workers), `nsys launch --trace=cuda,nvtx
#                --cuda-graph-trace=node`; collection started after the warm-up, stopped
#                after the measured requests
#   envonly      the two environment variables only, no profiler
#   nsyscuda     nsys with --trace=cuda only; spawn, no NVTX scopes
#   nsysnograph  nsys without --cuda-graph-trace=node
#   nsysidle     nsys, but the collection is never started
#   nsysnoevent  nsys plus --cuda-event-trace=false
#   nsysnew      nsys with the binary PROBE_NSYS_NEW
#   nsysc0, nsysc1  nsys with the whole server bound (taskset) to the CPUs of NUMA node 0 or 1
# Warning: on two GPUs, Nsight Systems 2025.1.3 tracing CUDA graph nodes during a collection
# caused CUDA faults (unspecified launch failure); 2026.6.1.251 did not (experiments/e9_two_gpu_debug).
#
# Gate (scripts/serve_lib.sh): the first GPU must have none of our processes (none of anybody's
# with PROBE_GATE_ANY_USER=1), utilisation <= PROBE_GATE_MAX_UTIL (5) % and memory used <=
# PROBE_GATE_MAX_MEM (2048) MiB, three checks a minute apart, and again before every server.
# A *_T2* config also needs the second GPU free of every user's processes, at most 5 % and
# 2048 MiB; with PROBE_GATE2_SHARED=1 the second GPU only needs none of our processes and the
# limits of the first. Raise the limits only for runs whose result is not a latency.
#
# Exit codes: 0 ok, 1 some run failed, 2 unknown config or mode, 3 gate never opened,
#             4 a GPU got busy, 6 port in use.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$1"
RUN_DIR="$(cd "$1" && pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
. scripts/serve_lib.sh

ROOT="${HP_ROOT:?}"
ENVDIR="${HP_ENV:?}"
PY="$ENVDIR/bin/python"
VLLM="$ENVDIR/bin/vllm"
GPU_UUID="${GATE_GPU_UUID:?}"
GPU2_UUID="${GATE_GPU2_UUID:-}"     # second GPU, only for *_T2* configs
Q35_4B="${HP_QWEN35_4B_DIR:?}"
Q35_9B="${HP_QWEN35_9B_DIR:?}"
Q3="${HP_QWEN3_DIR:?}"
NSYS="${PROBE_NSYS:-nsys}"
GIVE_UP_HOURS="${GIVE_UP_HOURS:-2}"

CONFIGS="${PROBE_CONFIGS:?}"
MODES="${PROBE_MODES:-plain nsys}"
LENGTHS="${PROBE_LENGTHS:?}"
REPS="${PROBE_REPS:-5}"
WARMUP="${PROBE_WARMUP:-1}"
OUT_TOKENS="${PROBE_OUT_LEN:-4}"
CLIENT_PY="${PROBE_CLIENT:-experiments/e4_profiling/probe_client.py}"
ANALYZE_PY="${PROBE_ANALYZE:-experiments/e4_profiling/analyze_q1.py}"
CLIENT_ARGS="${PROBE_CLIENT_ARGS:-}"
NVML="${PROBE_NVML:-0}"
SCALING="${PROBE_SCALING:-0}"
MAX_LEN="${PROBE_MAX_LEN:-32768}"
KV_4B="${PROBE_KV_BYTES_4B:-25769803776}"
KV_9B="${PROBE_KV_BYTES_9B:-17179869184}"
PORT="${PROBE_PORT:-8061}"
URL="http://127.0.0.1:$PORT"

export_serving_env "$ROOT"
OUT="$RUN_DIR/probe"
mkdir -p "$OUT"

gate_gpu1() {
  gpu_idle_once "$GPU_UUID" "${PROBE_GATE_ANY_USER:-0}" "${PROBE_GATE_MAX_UTIL:-5}" "${PROBE_GATE_MAX_MEM:-2048}"
}
gate_gpu2() {
  [ -n "$GPU2_UUID" ] || { log "GATE_GPU2_UUID is not set"; return 1; }
  if [ "${PROBE_GATE2_SHARED:-0}" = 1 ]; then
    gpu_idle_once "$GPU2_UUID" 0 "${PROBE_GATE_MAX_UTIL:-5}" "${PROBE_GATE_MAX_MEM:-2048}"
  else
    gpu_idle_once "$GPU2_UUID" 1 5 2048      # shared GPU: nothing may start next to somebody else's job
  fi
}
NEED_GPU2=0
case " $CONFIGS " in *_T2\ *|*_T2n\ *|*_T2ns\ *) NEED_GPU2=1 ;; esac
gate_once() { gate_gpu1 && { [ "$NEED_GPU2" -eq 0 ] || gate_gpu2; }; }

SP=""        # the process started here (the server, or `nsys launch` around it)
API=""       # the process that listens on the port
# Not serve_lib's stop_tree: the API server is stopped first and given 5 s before the
# process around it (nsys launch) is stopped, so the server shuts down its workers itself.
stop_server() {   # by PID only
  local p pids=""
  # tensor-parallel workers are grandchildren of the API server: collect the whole tree first
  [ -n "$API" ] && pids="$API $(descendants "$API" | tr '\n' ' ')"
  [ -n "$SP" ] && pids="$pids $SP $(descendants "$SP" | tr '\n' ' ')"
  [ -n "$API" ] && kill "$API" 2>/dev/null
  [ -n "$SP" ] && [ "$SP" != "$API" ] && { sleep 5; kill "$SP" 2>/dev/null; }
  # A hung worker can keep the server from exiting, so never wait without a limit:
  # up to 60 s for the tree to go, then TERM, then KILL.
  local n=0 alive
  while [ $n -lt 30 ]; do
    alive=0
    for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && break
    sleep 2; n=$((n + 1))
  done
  for p in $pids; do
    if kill -0 "$p" 2>/dev/null; then log "process $p outlived the server; kill $p"; kill "$p" 2>/dev/null; fi
  done
  sleep 5
  for p in $pids; do
    if kill -0 "$p" 2>/dev/null; then log "process $p ignores TERM; kill -9 $p"; kill -9 "$p" 2>/dev/null; fi
  done
  [ -n "$SP" ] && wait "$SP" 2>/dev/null
  SP=""; API=""
}
trap 'stop_server' EXIT
trap 'log "terminated"; stop_server; exit 143' TERM INT

log "probe waiting for the gate GPU $GPU_UUID; configs: $CONFIGS; modes: $MODES; lengths: $LENGTHS; reps: $REPS"
wait_for_gate gate_once "$GIVE_UP_HOURS"
log "gate open; starting"
"$NSYS" --version 2>&1 | head -n 1
nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader > "$RUN_DIR/nvidia_smi_before.txt"

STATUS=0
if [ "$SCALING" = 1 ]; then   # stand-alone sweep on the idle GPU; nothing else is running yet
  log "scaling sweep starting"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" experiments/e4_profiling/q2_scaling.py \
    --out "$OUT/scaling.json" ${PROBE_SCALING_ARGS:-} > "$OUT/scaling.log" 2>&1
  rc=$?
  log "scaling sweep rc=$rc $(tail -n 1 "$OUT/scaling.log" | cut -c1-200)"
  [ $rc -eq 0 ] || STATUS=1
  sleep 10
fi
for CFG in $CONFIGS; do
  case "$CFG" in
    h4_C0)  MODEL="$Q35_4B"; NAME=qwen35; PARSER=qwen3_xml; KV="$KV_4B"; EXTRA=(--language-model-only) ;;
    h4_C1)  MODEL="$Q35_4B"; NAME=qwen35; PARSER=qwen3_xml; KV="$KV_4B"; EXTRA=(--language-model-only --prefix-match-unit 16) ;;
    # 64 tokens: the grid on which PR #65186 saves Triton prefill checkpoints
    h4_C64) MODEL="$Q35_4B"; NAME=qwen35; PARSER=qwen3_xml; KV="$KV_4B"; EXTRA=(--language-model-only --prefix-match-unit 64) ;;
    h9_C0)  MODEL="$Q35_9B"; NAME=qwen35; PARSER=qwen3_xml; KV="$KV_9B"; EXTRA=(--language-model-only) ;;
    h9_C1)  MODEL="$Q35_9B"; NAME=qwen35; PARSER=qwen3_xml; KV="$KV_9B"; EXTRA=(--language-model-only --prefix-match-unit 16) ;;
    fa_F0)  MODEL="$Q3"; NAME=qwen3; PARSER=hermes; KV="$KV_4B"; EXTRA=() ;;
    fa_F16) MODEL="$Q3"; NAME=qwen3; PARSER=hermes; KV="$KV_4B"; EXTRA=(--prefix-match-unit 16) ;;
    h4_T2)  MODEL="$Q35_4B"; NAME=qwen35; PARSER=qwen3_xml; KV="$KV_4B"; EXTRA=(--language-model-only --tensor-parallel-size 2) ;;
    fa_T2)  MODEL="$Q3"; NAME=qwen3; PARSER=hermes; KV="$KV_4B"; EXTRA=(--tensor-parallel-size 2) ;;
    # the same two with vLLM's own all-reduce kernel switched off, so NCCL does the all-reduce
    h4_T2n) MODEL="$Q35_4B"; NAME=qwen35; PARSER=qwen3_xml; KV="$KV_4B"; EXTRA=(--language-model-only --tensor-parallel-size 2 --disable-custom-all-reduce) ;;
    fa_T2n) MODEL="$Q3"; NAME=qwen3; PARSER=hermes; KV="$KV_4B"; EXTRA=(--tensor-parallel-size 2 --disable-custom-all-reduce) ;;
    # fa_T2n with the two GPUs in the other order (rank 0 on GATE_GPU2_UUID, rank 1 on the gate GPU)
    fa_T2ns) MODEL="$Q3"; NAME=qwen3; PARSER=hermes; KV="$KV_4B"; EXTRA=(--tensor-parallel-size 2 --disable-custom-all-reduce) ;;
    *) log "unknown config $CFG"; exit 2 ;;
  esac
  DEVICES="$GPU_UUID"; TWO=0
  case "$CFG" in *_T2|*_T2n) DEVICES="$GPU_UUID,$GPU2_UUID"; TWO=1 ;; *_T2ns) DEVICES="$GPU2_UUID,$GPU_UUID"; TWO=1 ;; esac
  for MODE in $MODES; do
    TAG="${CFG}_${MODE}"
    idle=0
    for _ in 1 2 3; do
      if gate_gpu1 && { [ "$TWO" -eq 0 ] || gate_gpu2; }; then idle=1; break; fi
      sleep 60
    done
    if [ $idle -ne 1 ]; then
      log "a GPU this config needs is busy before $TAG; stopping so as not to crowd it"
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
      STATUS=4; break 2
    fi
    if [ "$(ss -ltn 2>/dev/null | grep -c ":$PORT ")" -gt 0 ]; then log "port $PORT in use"; STATUS=6; break 2; fi

    LOG="$RUN_DIR/serve_$TAG.log"
    CMD=("$VLLM" serve "$MODEL" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT"
         --dtype bfloat16 --max-model-len "$MAX_LEN" --kv-cache-memory-bytes "$KV"
         --enable-prefix-caching --enable-prompt-tokens-details --generation-config vllm --seed 0
         --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser "$PARSER" "${EXTRA[@]}")
    [ -n "${PROBE_SERVE_EXTRA:-}" ] && CMD+=($PROBE_SERVE_EXTRA)
    SESS="probe$(date +%s)"
    USE_NSYS=0; NSYS_COLLECT=0; NSYS_OPTS=(); MODE_ENV=(); PREFIX=()
    NSB="$NSYS"                       # the profiler binary of this mode
    BOTH_ENV=(VLLM_NVTX_SCOPES_FOR_PROFILING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn)
    case "$MODE" in
      plain)       ;;
      nsys)        USE_NSYS=1; NSYS_COLLECT=1; NSYS_OPTS=(--trace=cuda,nvtx --cuda-graph-trace=node); MODE_ENV=("${BOTH_ENV[@]}") ;;
      envonly)     MODE_ENV=("${BOTH_ENV[@]}") ;;
      nsyscuda)    USE_NSYS=1; NSYS_COLLECT=1; NSYS_OPTS=(--trace=cuda); MODE_ENV=(VLLM_WORKER_MULTIPROC_METHOD=spawn) ;;
      nsysnograph) USE_NSYS=1; NSYS_COLLECT=1; NSYS_OPTS=(--trace=cuda,nvtx); MODE_ENV=("${BOTH_ENV[@]}") ;;
      nsysidle)    USE_NSYS=1; NSYS_COLLECT=0; NSYS_OPTS=(--trace=cuda,nvtx --cuda-graph-trace=node); MODE_ENV=("${BOTH_ENV[@]}") ;;
      nsysnoevent) USE_NSYS=1; NSYS_COLLECT=1; NSYS_OPTS=(--trace=cuda,nvtx --cuda-graph-trace=node --cuda-event-trace=false); MODE_ENV=("${BOTH_ENV[@]}") ;;
      nsysnew)     USE_NSYS=1; NSYS_COLLECT=1; NSYS_OPTS=(--trace=cuda,nvtx --cuda-graph-trace=node); MODE_ENV=("${BOTH_ENV[@]}")
                   NSB="${PROBE_NSYS_NEW:?}" ;;
      # numactl may be missing; taskset binds the CPUs and memory follows first touch
      nsysc0|nsysc1) USE_NSYS=1; NSYS_COLLECT=1; NSYS_OPTS=(--trace=cuda,nvtx --cuda-graph-trace=node); MODE_ENV=("${BOTH_ENV[@]}")
                   PREFIX=(taskset -c "$(cat "/sys/devices/system/node/node${MODE#nsysc}/cpulist")") ;;
      *) log "unknown mode $MODE"; exit 2 ;;
    esac
    if [ "$USE_NSYS" = 1 ]; then
      log "$TAG command: CUDA_VISIBLE_DEVICES=$DEVICES ${MODE_ENV[*]} ${PREFIX[*]} $NSB launch --session-new=$SESS ${NSYS_OPTS[*]} ${CMD[*]}"
      env CUDA_VISIBLE_DEVICES="$DEVICES" "${MODE_ENV[@]}" "${PREFIX[@]}" \
        "$NSB" launch --session-new="$SESS" "${NSYS_OPTS[@]}" \
        "${CMD[@]}" > "$LOG" 2>&1 &
    else
      log "$TAG command: CUDA_VISIBLE_DEVICES=$DEVICES ${MODE_ENV[*]} ${CMD[*]}"
      env CUDA_VISIBLE_DEVICES="$DEVICES" "${MODE_ENV[@]}" "${CMD[@]}" > "$LOG" 2>&1 &
    fi
    SP=$!
    log "$TAG started pid $SP, log $LOG"
    if ! wait_healthy "$URL" "$SP" "${PROBE_STARTUP_TRIES:-360}"; then
      log "$TAG server failed to come up"; grep -E "Error|error" "$LOG" | tail -n 8 | cut -c1-300
      stop_server; STATUS=1
      if [ "${PROBE_STOP_ON_FAIL:-0}" = 1 ]; then log "stopping after the first failure"; break 2; fi
      continue
    fi
    API="$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -n 1 | cut -d= -f2)"
    log "$TAG healthy. listening pid ${API:-?}, its descendants: $([ -n "$API" ] && descendants "$API" | tr '\n' ' ')"
    grep -i -E "block size|GDN prefill kernel|GPU KV cache size|prefix_match_unit|CUDA graph|cudagraph|nccl|tensor_parallel|world_size" "$LOG" \
      | cut -c1-300 | head -n 30 > "$RUN_DIR/serve_${TAG}_facts.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$RUN_DIR/nvidia_smi_$TAG.txt"
    if [ -n "${PROBE_CHECK:-}" ]; then
      "$PY" "$PROBE_CHECK" --url "$URL" --model "$NAME" --tokenizer-dir "$MODEL" --json "$OUT/${TAG}_check.json" ${PROBE_CHECK_ARGS:-} > "$OUT/check_$TAG.log" 2>&1
      crc=$?
      log "$TAG check rc=$crc $(tail -n 1 "$OUT/check_$TAG.log" | cut -c1-200)"
      if [ $crc -ne 0 ]; then
        log "$TAG check failed: no measurements on this server"; stop_server; STATUS=1
        if [ "${PROBE_STOP_ON_FAIL:-0}" = 1 ]; then log "stopping after the first failure"; break 2; fi
        continue
      fi
    fi

    CLIENT=("$PY" "$CLIENT_PY" --url "$URL" --model "$NAME"
            --tokenizer-dir "$MODEL" --lengths $LENGTHS --out-tokens "$OUT_TOKENS" $CLIENT_ARGS)
    # warm-up requests (kernel compilation, allocator) are sent before collection starts
    "${CLIENT[@]}" --reps 0 --warmup "$WARMUP" --label "${TAG}_warm" --out "$OUT/${TAG}_warmup.jsonl" >> "$OUT/client_$TAG.log" 2>&1
    # Where the server's processes are after the warm-up: allowed CPUs, the CPU each one is on now,
    # and the pages of its memory per NUMA node (from numa_maps). Read-only; one line per process.
    {
      for p in $API $([ -n "$API" ] && descendants "$API"); do
        [ -r "/proc/$p/status" ] || continue
        echo "pid $p | $(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-50) | cpus_allowed $(awk '/Cpus_allowed_list/{print $2}' "/proc/$p/status") | on_cpu $(ps -o psr= -p "$p" | tr -d ' ') | pages $(awk '{for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/){split($i,a,"=");s[a[1]]+=a[2]}} END{for(k in s) printf "%s=%d ", k, s[k]}' "/proc/$p/numa_maps" 2>/dev/null)"
      done
    } > "$RUN_DIR/placement_$TAG.txt" 2>/dev/null
    if [ "$NSYS_COLLECT" = 1 ]; then
      # sampling and context-switch tracing are `start` options since Nsight Systems 2021.5
      "$NSB" start --session="$SESS" --sample=none --cpuctxsw=none -o "$OUT/$TAG" --force-overwrite=true >> "$OUT/nsys_$TAG.log" 2>&1
      log "$TAG nsys start rc=$?"
    fi
    NV=""
    if [ "$NVML" = 1 ] && [ "$MODE" = plain ]; then
      "$PY" experiments/e4_profiling/nvml_sampler.py --uuid "$GPU_UUID" --out "$OUT/${TAG}_nvml.csv" \
        >> "$OUT/client_$TAG.log" 2>&1 &
      NV=$!
    fi
    "${CLIENT[@]}" --reps "$REPS" --label "$TAG" --out "$OUT/$TAG.jsonl" >> "$OUT/client_$TAG.log" 2>&1
    rc=$?
    if [ -n "$NV" ]; then kill "$NV" 2>/dev/null; wait "$NV" 2>/dev/null; fi
    log "$TAG client rc=$rc $(tail -n 1 "$OUT/client_$TAG.log" | cut -c1-200)"
    [ $rc -eq 0 ] || STATUS=1
    if [ "$NSYS_COLLECT" = 1 ]; then
      "$NSB" stop --session="$SESS" >> "$OUT/nsys_$TAG.log" 2>&1
      log "$TAG nsys stop rc=$? report: $(ls -la "$OUT/$TAG.nsys-rep" 2>/dev/null | awk '{print $5}') bytes"
    fi
    kill -0 "$SP" 2>/dev/null || { log "$TAG server died"; STATUS=1; }
    log "$TAG done. oom_or_dead=$(grep -c -E 'OutOfMemoryError|EngineDeadError' "$LOG")"
    stop_server
    if [ "$TWO" -eq 1 ]; then   # nothing of this run may stay on the shared second GPU
      left="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$GPU2_UUID")"
      log "$TAG after stop: processes on the second GPU = $left"
    fi
    [ "$USE_NSYS" = 1 ] && "$NSB" shutdown --session="$SESS" --kill none >> "$OUT/nsys_$TAG.log" 2>&1
    sleep 10
    if [ "$STATUS" -ne 0 ] && [ "${PROBE_STOP_ON_FAIL:-0}" = 1 ]; then log "stopping after the first failure"; break 2; fi

    if { [ "$MODE" = nsys ] || [ "$MODE" = nsysnew ]; } && [ -s "$OUT/$TAG.nsys-rep" ]; then   # CPU only, server is down
      "$NSB" export --type sqlite --force-overwrite=true -o "$OUT/$TAG.sqlite" "$OUT/$TAG.nsys-rep" >> "$OUT/nsys_$TAG.log" 2>&1
      CUDA_VISIBLE_DEVICES= "$PY" "$ANALYZE_PY" --sqlite "$OUT/$TAG.sqlite" \
        --requests "$OUT/$TAG.jsonl" --json "$OUT/$TAG.analysis.json" >> "$OUT/analyze_$TAG.log" 2>&1
      log "$TAG analysis rc=$? $(tail -n 1 "$OUT/analyze_$TAG.log" | cut -c1-200)"
    fi
  done
done

nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader > "$RUN_DIR/nvidia_smi_after.txt"
tar -czf "$RUN_DIR/probe_results.tar.gz" -C "$OUT" $(cd "$OUT" && ls *.jsonl *.analysis.json *_check.json *.log *.csv scaling.json 2>/dev/null) 2>/dev/null
tar -czf "$RUN_DIR/serve_logs.tar.gz" -C "$RUN_DIR" $(cd "$RUN_DIR" && ls serve_*.log serve_*_facts.txt nvidia_smi_*.txt placement_*.txt 2>/dev/null) 2>/dev/null
echo "{\"status\": $STATUS, \"finished_at\": \"$(date '+%F %T %Z')\"}" > "$RUN_DIR/metrics.json"
log "probe finished, status=$STATUS"
exit $STATUS
