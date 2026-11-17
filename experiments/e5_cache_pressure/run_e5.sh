#!/usr/bin/env bash
# Launcher of the cache-pressure experiment (E5) on a GPU server.
# For each cell CFG@KVMIB of E5_CELLS: start one vLLM server on the gate GPU with
# --kv-cache-memory-bytes KVMIB MiB, run pressure_replay.py for every seed and
# output length, stop the server by PID.
#
# Server configs:
#   F0  Qwen3-4B, defaults (64-token blocks)
#   C0  Qwen3.5-4B, defaults (576-token blocks)
#   C1  Qwen3.5-4B, --prefix-match-unit 16
#
# E5_EXPECT_REFUSE lists cells whose server is expected to refuse to start (the
# pool cannot hold one request of --max-model-len). For such a cell a start-up
# that ends with vLLM's KV cache complaint is logged as ok and the run goes on;
# a server that comes up anyway, or dies for another reason, is a failure.
#
# Gate: nothing starts unless the gate GPU is idle (no compute process of this
# user, utilization <= GATE_MAX_UTIL %, memory used <= GATE_MAX_MEM MiB; with
# GATE_ANY_USER=1 no compute process of any user), checked three times 60 s
# apart at the beginning and again before every server start. Raise
# GATE_MAX_UTIL only for runs whose result is a hit count, not a latency.
#
# After the first failure no other server is started (E5_STOP_ON_FAIL=0 turns
# that off). A failed control request or cache reset always ends the run.
#
# Inputs (environment):
#   HP_ROOT, HP_ENV (venv with vllm), HP_QWEN35_DIR, HP_QWEN3_DIR, GATE_GPU_UUID
#   E5_CELLS "F0@11241 C1@3456 ...", E5_STAGE (S, I, L, W or K; goes into the tag),
#   E5_WORKLOAD (synth, smith or wildchat), E5_MODE (closed or window),
#   E5_INFLIGHT, E5_N_SESSIONS, E5_OUT_LENS "16 256"; synth also E5_TURNS, E5_P0, E5_STEP
#   optional: E5_SEEDS (1 2 3), E5_LIVE, E5_MAX_LEN (16384), E5_THINKING (off),
#   E5_PORT (8062), E5_MAX_NUM_SEQS, E5_KV_METRICS=1, E5_SERVE_EXTRA, E5_CPUSET,
#   E5_EXPECT_REFUSE, E5_STOP_ON_FAIL (1), E5_STARTUP_TRIES (360),
#   GATE_ANY_USER (0), GATE_MAX_UTIL (5), GATE_MAX_MEM (2048), GIVE_UP_HOURS (2)
# Outputs: RUN_DIR/e5/<tag>.jsonl, .summary.json, .metrics.jsonl with
#   tag = <cfg>_<stage>_<workload>_kv<MiB>_out<n>_seed<seed>; server logs and
#   archives in RUN_DIR; RUN_DIR/metrics.json with the exit status.
#
# Exit codes: 0 ok, 1 some run failed, 2 bad arguments, 3 gate never opened,
#             4 GPU got busy, 6 port in use, 7 a control request or reset failed.
# Usage: bash experiments/e5_cache_pressure/run_e5.sh <run_dir>
set -uo pipefail

mkdir -p "${1:?usage: run_e5.sh <run_dir>}" && RUN_DIR="$(cd "$1" && pwd)" || exit 2
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 2
. "$REPO/scripts/serve_lib.sh"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

ROOT="${HP_ROOT:?}"
ENVDIR="${HP_ENV:?}"
PY="$ENVDIR/bin/python"
VLLM="$ENVDIR/bin/vllm"
GPU_UUID="${GATE_GPU_UUID:?}"
QWEN35_DIR="${HP_QWEN35_DIR:?}"
QWEN3_DIR="${HP_QWEN3_DIR:?}"
GIVE_UP_HOURS="${GIVE_UP_HOURS:-2}"
GATE_ANY_USER="${GATE_ANY_USER:-0}"
GATE_MAX_UTIL="${GATE_MAX_UTIL:-5}"
GATE_MAX_MEM="${GATE_MAX_MEM:-2048}"

CELLS="${E5_CELLS:?}"                  # e.g. "F0@11776 C1@4096"
EXPECT_REFUSE="${E5_EXPECT_REFUSE:-}"  # cells expected to refuse to start
STAGE="${E5_STAGE:?}"                  # S, I, L, W or K: goes into the tag
WORKLOAD="${E5_WORKLOAD:?}"            # synth, smith or wildchat
SEEDS="${E5_SEEDS:-1 2 3}"
MODE="${E5_MODE:?}"                    # closed or window
INFLIGHT="${E5_INFLIGHT:?}"
LIVE="${E5_LIVE:-$INFLIGHT}"
N_SESSIONS="${E5_N_SESSIONS:?}"
OUT_LENS="${E5_OUT_LENS:?}"            # output lengths, space separated
MAX_LEN="${E5_MAX_LEN:-16384}"
THINKING="${E5_THINKING:-off}"         # recorded workloads only
PORT="${E5_PORT:-8062}"
URL="http://127.0.0.1:$PORT"
STOP_ON_FAIL="${E5_STOP_ON_FAIL:-1}"

case "$WORKLOAD" in
  synth)    WL_ARGS=(--turns "${E5_TURNS:?}" --p0 "${E5_P0:?}" --step "${E5_STEP:?}") ;;
  smith)    WL_ARGS=(--path "$ROOT/data/swe-smith-trajectories/data/tool-00004-of-00008.parquet" --thinking "$THINKING") ;;
  wildchat) WL_ARGS=(--path "$ROOT/data/wildchat-1m/data/train-00008-of-00014.parquet" --thinking "$THINKING") ;;
  *) echo "unknown workload $WORKLOAD"; exit 2 ;;
esac
for CELL in $CELLS $EXPECT_REFUSE; do
  case "$CELL" in
    F0@*|C0@*|C1@*) ;;
    *) echo "bad cell $CELL (want CFG@KVMIB with CFG in F0 C0 C1)"; exit 2 ;;
  esac
  case "${CELL#*@}" in ''|*[!0-9]*) echo "bad pool size in $CELL (MiB, digits only)"; exit 2 ;; esac
done
SERVER_OPTS=()
[ "${E5_KV_METRICS:-0}" = 1 ] && SERVER_OPTS=(--kv-cache-metrics --kv-cache-metrics-sample 1.0)
# The hybrid model needs one state block per possible sequence: with the default --max-num-seqs 256 a pool
# under 256 blocks (about 4.6 GiB for Qwen3.5-4B) refuses to start. E5_MAX_NUM_SEQS lowers it for every cell.
[ -n "${E5_MAX_NUM_SEQS:-}" ] && SERVER_OPTS+=(--max-num-seqs "$E5_MAX_NUM_SEQS")
# E5_SERVE_EXTRA: extra `vllm serve` arguments for every cell, split on spaces (e.g. the flags of PR #66881).
# shellcheck disable=SC2206
[ -n "${E5_SERVE_EXTRA:-}" ] && SERVER_OPTS+=($E5_SERVE_EXTRA)
PIN=()                                  # optional CPU pinning of the server and the client
[ -n "${E5_CPUSET:-}" ] && PIN=(taskset -c "$E5_CPUSET")

export_serving_env "$ROOT"
OUT="$RUN_DIR/e5"
mkdir -p "$OUT"

gate_check() { gpu_idle_once "$GPU_UUID" "$GATE_ANY_USER" "$GATE_MAX_UTIL" "$GATE_MAX_MEM"; }

SP=""        # the process started here
API=""       # the process that listens on the port
stop_server() {   # by PID only
  stop_tree "$API" "$SP"
  SP=""; API=""
}
trap 'stop_server' EXIT
trap 'log "terminated"; stop_server; exit 143' TERM INT

log "E5 waiting for the gate GPU $GPU_UUID; stage $STAGE, workload $WORKLOAD, cells: $CELLS; expected to refuse: ${EXPECT_REFUSE:-none}; seeds: $SEEDS; output lengths: $OUT_LENS; mode $MODE, in flight $INFLIGHT, live $LIVE"
wait_for_gate gate_check "$GIVE_UP_HOURS"
log "gate open; starting E5"
nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu,power.limit --format=csv,noheader > "$RUN_DIR/nvidia_smi_before.txt"

STATUS=0
for CELL in $CELLS; do
  CFG="${CELL%@*}"; KVMIB="${CELL#*@}"; ID="${CFG}_kv${KVMIB}"
  case "$CFG" in
    F0) MODEL="$QWEN3_DIR"; NAME=qwen3; PARSER=hermes; ARCH=fa64; EXTRA=() ;;
    C0) MODEL="$QWEN35_DIR"; NAME=qwen35; PARSER=qwen3_xml; ARCH=hy576; EXTRA=(--language-model-only) ;;
    C1) MODEL="$QWEN35_DIR"; NAME=qwen35; PARSER=qwen3_xml; ARCH=hy16; EXTRA=(--language-model-only --prefix-match-unit 16) ;;
  esac
  REFUSE=0
  case " $EXPECT_REFUSE " in *" $CELL "*) REFUSE=1 ;; esac

  idle=0
  for _ in 1 2 3; do if gate_check; then idle=1; break; fi; sleep 60; done
  if [ $idle -ne 1 ]; then
    log "gate GPU is busy before $ID; stopping so as not to crowd it"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
    STATUS=4; break
  fi
  if [ "$(ss -ltn 2>/dev/null | grep -c ":$PORT ")" -gt 0 ]; then log "port $PORT in use"; STATUS=6; break; fi

  LOG="$RUN_DIR/serve_$ID.log"
  CMD=(${PIN[@]+"${PIN[@]}"} "$VLLM" serve "$MODEL" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT"
       --dtype bfloat16 --max-model-len "$MAX_LEN" --kv-cache-memory-bytes "$((KVMIB * 1048576))"
       --enable-prefix-caching --enable-prompt-tokens-details --generation-config vllm --seed 0
       --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser "$PARSER"
       ${EXTRA[@]+"${EXTRA[@]}"} ${SERVER_OPTS[@]+"${SERVER_OPTS[@]}"})
  log "$ID command: CUDA_VISIBLE_DEVICES=$GPU_UUID ${CMD[*]}"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "${CMD[@]}" > "$LOG" 2>&1 &
  SP=$!
  log "$ID started pid $SP, log $LOG"
  ok=0
  wait_healthy "$URL" "$SP" "${E5_STARTUP_TRIES:-360}" && ok=1     # 5 s per try
  if [ $ok -ne 1 ]; then
    grep -E "Error|error" "$LOG" | tail -n 8 | cut -c1-400 > "$RUN_DIR/serve_${ID}_startup_errors.txt"
    if [ $REFUSE -eq 1 ] && ! kill -0 "$SP" 2>/dev/null && grep -q -i "KV cache" "$RUN_DIR/serve_${ID}_startup_errors.txt"; then
      log "$ID refused to start, as expected: $(grep -i "KV cache" "$RUN_DIR/serve_${ID}_startup_errors.txt" | tail -n 1 | cut -c1-300)"
      stop_server; sleep 10; continue
    fi
    log "$ID server failed to come up"; cat "$RUN_DIR/serve_${ID}_startup_errors.txt"
    stop_server; STATUS=1
    if [ "$STOP_ON_FAIL" = 1 ]; then log "stopping after the first failure"; break; fi
    continue
  fi
  API="$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -n 1 | cut -d= -f2)"
  log "$ID healthy. listening pid ${API:-?}, its descendants: $([ -n "$API" ] && descendants "$API" | tr '\n' ' ')"
  grep -i -E "block size|GDN prefill kernel|GPU KV cache size|prefix_match_unit|num_gpu_blocks|Maximum concurrency" "$LOG" \
    | cut -c1-300 | head -n 30 > "$RUN_DIR/serve_${ID}_facts.txt"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$RUN_DIR/nvidia_smi_$ID.txt"
  if [ $REFUSE -eq 1 ]; then
    log "$ID came up although it was expected to refuse; no requests sent"
    stop_server; STATUS=1; sleep 10
    if [ "$STOP_ON_FAIL" = 1 ]; then log "stopping after the first failure"; break; fi
    continue
  fi

  for SEED in $SEEDS; do
    for OUTLEN in $OUT_LENS; do
      TAG="${CFG}_${STAGE}_${WORKLOAD}_kv${KVMIB}_out${OUTLEN}_seed${SEED}"
      ${PIN[@]+"${PIN[@]}"} "$PY" experiments/e5_cache_pressure/pressure_replay.py --url "$URL" --model "$NAME" \
        --tokenizer-dir "$MODEL" --workload "$WORKLOAD" "${WL_ARGS[@]}" \
        --seed "$SEED" --n-sessions "$N_SESSIONS" --max-len "$MAX_LEN" --n-out "$OUTLEN" \
        --mode "$MODE" --inflight "$INFLIGHT" --live "$LIVE" --arch "$ARCH" --label "$TAG" \
        --out "$OUT/$TAG.jsonl" --metrics-out "$OUT/$TAG.metrics.jsonl" >> "$OUT/client_$ID.log" 2>&1
      rc=$?
      log "$TAG rc=$rc $(tail -n 1 "$OUT/client_$ID.log" | cut -c1-300)"
      if [ $rc -eq 4 ] || [ $rc -eq 5 ]; then
        log "control or cache reset failed in $TAG: ending the whole run"
        stop_server; STATUS=7; break 3
      fi
      [ $rc -eq 0 ] || STATUS=1
      kill -0 "$SP" 2>/dev/null || { log "$ID server died during $TAG"; STATUS=1; break 2; }
      if [ "$STATUS" -ne 0 ] && [ "$STOP_ON_FAIL" = 1 ]; then break 2; fi
    done
  done
  log "$ID done. oom_or_dead=$(grep -c -E 'OutOfMemoryError|EngineDeadError' "$LOG")"
  stop_server
  sleep 10
  if [ "$STATUS" -ne 0 ] && [ "$STOP_ON_FAIL" = 1 ]; then log "stopping after the first failure"; break; fi
done

nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu,power.limit --format=csv,noheader > "$RUN_DIR/nvidia_smi_after.txt"
tar -czf "$RUN_DIR/e5_results.tar.gz" -C "$OUT" $(cd "$OUT" && ls *.jsonl *.summary.json *.log 2>/dev/null) 2>/dev/null
tar -czf "$RUN_DIR/serve_logs.tar.gz" -C "$RUN_DIR" $(cd "$RUN_DIR" && ls serve_*.log serve_*_facts.txt serve_*_startup_errors.txt nvidia_smi_*.txt 2>/dev/null) 2>/dev/null
echo "{\"status\": $STATUS, \"finished_at\": \"$(date '+%F %T %Z')\"}" > "$RUN_DIR/metrics.json"
log "E5 finished, status=$STATUS"
exit $STATUS
