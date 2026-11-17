#!/usr/bin/env bash
# Multi-turn replay launcher (E1, E2, E3, E6, and the four-pass runs of E7 / E8).
#
# For each server config in REPLAY_CONFIGS: wait until the GPU is idle, start one
# vLLM server, run experiments/e1_prefix_hits/replay.py against it for every
# (seed, dataset, thinking, mode, interleave), then stop the server by PID and
# move on. Every client run starts with a prefix-cache reset and three control
# requests; a failed control or cache reset ends the whole run (exit 7).
#
# Usage:  scripts/run_replay.sh RUN_DIR          (any cwd; the repo root is found
#                                                  from this script's path)
#
# Server configs. A trailing lower-case letter marks a repeat of the same config
# (C0a, C0b, ...): same command line, separate logs and output files.
#   F0  Qwen3-4B, defaults                       F1  Qwen3-4B with the Qwen3.5 chat template
#   C0  Qwen3.5, defaults                        C1  --prefix-match-unit $REPLAY_UNIT
#   C2  --prefix-cache-retention-interval 576    C3  --prefix-match-unit 16 + C2
#   C4  --prefix-match-unit 16 --prefix-cache-retention-interval None
# Every server: --dtype bfloat16 --max-model-len $REPLAY_MAX_LEN
#   --kv-cache-memory-bytes $REPLAY_KV_BYTES --enable-prefix-caching
#   --enable-prompt-tokens-details --generation-config vllm --seed 0
#   --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser <hermes|qwen3_xml>
#   (the Qwen3.5 configs also --language-model-only), then $REPLAY_SERVE_EXTRA.
#
# Required variables
#   HP_ROOT          experiment root on the GPU server: data/, models/, cache/, tmp/ below it
#   HP_ENV           Python environment with vLLM (bin/python, bin/vllm)
#   HP_QWEN35_DIR    hybrid model directory (Qwen3.5-4B, -9B or -27B-FP8)
#   HP_QWEN3_DIR     full-attention model directory (Qwen3-4B)
#   GATE_GPU_UUID    the GPU to run on (CUDA_VISIBLE_DEVICES=<uuid>)
#   REPLAY_CONFIGS   e.g. "F0 C1 C4 C0 F1 C3 C2" (E1) or "C0a C1a C1b C0b" (four-pass)
#   REPLAY_N_CHAT, REPLAY_N_SMITH, REPLAY_N_REBENCH   sessions per seed in R replay
#   REPLAY_OUT_CAP   R replay: cap on the recorded reply length (min_tokens = max_tokens)
#   REPLAY_O_OUT_CAP O replay: max_tokens
#   REPLAY_KV_BYTES  --kv-cache-memory-bytes (fixed, so every config gets the same pool)
# Optional variables (default)
#   REPLAY_SEEDS ("1 2 3")  REPLAY_DATASETS ("wildchat smith rebench")
#   REPLAY_THINKS ("on off")  REPLAY_MODES ("R O")  REPLAY_INTERLEAVES ("1")
#   REPLAY_MULTI_CONFIGS ("F0 C0 C3")  configs that also run interleave > 1 (R mode only)
#   REPLAY_O_CONFIGS ($REPLAY_CONFIGS) configs that run O replay
#   REPLAY_O_N_CHAT / _SMITH / _REBENCH (the R numbers)  sessions per seed in O replay
#   REPLAY_CONTROL_ONLY_CONFIGS ("")  configs that only start and send the three controls
#   REPLAY_DIAG_CONFIGS ("")  configs that end with the no-delta diagnostic run
#   REPLAY_DIAG_N (40)  REPLAY_HYBRID_BLOCK (576: block size the hybrid server reports)
#   REPLAY_UNIT (16)  --prefix-match-unit of config C1
#   REPLAY_SERVE_EXTRA ("")  extra `vllm serve` arguments for every config, split on spaces
#   REPLAY_MAX_LEN (32768)  REPLAY_PORT (8061)
#   REPLAY_ACCOUNT (0)  1: afterwards redo the offline accounting with block 576 (CPU only)
#   REPLAY_ACCOUNT_SEEDS ("1 2 3")
#   REPLAY_GATE_SELFTEST (0)  1: one gate check, then exit
#   GATE_ANY_USER (0)  1: the GPU must also be free of every other user's processes
#   GATE_MAX_UTIL (5)  GATE_MAX_MEM (2048)  gate thresholds, % and MiB
#   NOT_BEFORE_EPOCH (0)  GIVE_UP_HOURS (12)
#
# Gate: nothing starts until the GPU is idle on three checks 60 s apart (see
# scripts/serve_lib.sh). The same check is repeated (up to three times, a minute
# apart) before every server start; if the GPU is still busy then, the script
# stops instead of crowding in.
#
# Output in RUN_DIR: e1/<cfg>_<dataset>_think<on|off>_<R|O>_s<S>_seed<seed>.jsonl and
# .summary.json (replay.py), e1/client_<cfg>.log, diag/ (diagnostic run),
# serve_<cfg>.log, serve_<cfg>_facts.txt, nvidia_smi_*.txt, summary.json (all
# run summaries merged), e1_requests.tar.gz, diag_requests.tar.gz,
# serve_logs.tar.gz, metrics.json ({"status": ...}); with REPLAY_ACCOUNT=1 also
# e0b/ and e0b_requests.tar.gz.
#
# Exit codes: 0 ok, 1 some run failed, 3 gate never opened, 4 GPU got busy,
#             6 port in use, 7 a control request or cache reset failed (stop rule).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$REPO/scripts/serve_lib.sh"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

RUN_DIR="$1"
ROOT="${HP_ROOT:?}"
ENVDIR="${HP_ENV:?}"
PY="$ENVDIR/bin/python"
VLLM="$ENVDIR/bin/vllm"
GPU_UUID="${GATE_GPU_UUID:?}"
QWEN35_DIR="${HP_QWEN35_DIR:?}"
QWEN3_DIR="${HP_QWEN3_DIR:?}"
NOT_BEFORE_EPOCH="${NOT_BEFORE_EPOCH:-0}"
GIVE_UP_HOURS="${GIVE_UP_HOURS:-12}"
GATE_ANY_USER="${GATE_ANY_USER:-0}"
GATE_MAX_UTIL="${GATE_MAX_UTIL:-5}"
GATE_MAX_MEM="${GATE_MAX_MEM:-2048}"

CONFIGS="${REPLAY_CONFIGS:?}"
SEEDS="${REPLAY_SEEDS:-1 2 3}"
DATASETS="${REPLAY_DATASETS:-wildchat smith rebench}"
THINKS="${REPLAY_THINKS:-on off}"
MODES="${REPLAY_MODES:-R O}"
INTERLEAVES="${REPLAY_INTERLEAVES:-1}"
MULTI_CONFIGS="${REPLAY_MULTI_CONFIGS:-F0 C0 C3}"
O_CONFIGS="${REPLAY_O_CONFIGS:-$CONFIGS}"
CONTROL_ONLY="${REPLAY_CONTROL_ONLY_CONFIGS:-}"
DIAG_CONFIGS="${REPLAY_DIAG_CONFIGS:-}"
DIAG_N="${REPLAY_DIAG_N:-40}"
HYBRID_BLOCK="${REPLAY_HYBRID_BLOCK:-576}"
UNIT="${REPLAY_UNIT:-16}"
SERVE_EXTRA="${REPLAY_SERVE_EXTRA:-}"
ACCOUNT="${REPLAY_ACCOUNT:-0}"
ACCOUNT_SEEDS="${REPLAY_ACCOUNT_SEEDS:-1 2 3}"
N_CHAT="${REPLAY_N_CHAT:?}"; N_SMITH="${REPLAY_N_SMITH:?}"; N_REBENCH="${REPLAY_N_REBENCH:?}"
O_N_CHAT="${REPLAY_O_N_CHAT:-$N_CHAT}"; O_N_SMITH="${REPLAY_O_N_SMITH:-$N_SMITH}"; O_N_REBENCH="${REPLAY_O_N_REBENCH:-$N_REBENCH}"
OUT_CAP="${REPLAY_OUT_CAP:?}"
O_OUT_CAP="${REPLAY_O_OUT_CAP:?}"
MAX_LEN="${REPLAY_MAX_LEN:-32768}"
KV_BYTES="${REPLAY_KV_BYTES:?}"
PORT="${REPLAY_PORT:-8061}"
URL="http://127.0.0.1:$PORT"
REPLAY="$REPO/experiments/e1_prefix_hits/replay.py"

export_serving_env "$ROOT"
OUT="$RUN_DIR/e1"
mkdir -p "$OUT"

gate_check() { gpu_idle_once "$GPU_UUID" "$GATE_ANY_USER" "$GATE_MAX_UTIL" "$GATE_MAX_MEM"; }

if [ "${REPLAY_GATE_SELFTEST:-0}" = "1" ]; then
  if gate_check; then log "selftest: gate would OPEN"; else log "selftest: gate stays CLOSED"; fi
  exit 0
fi

SP=""
stop_server() {   # by PID only: the server and everything below it
  [ -n "$SP" ] || return 0
  stop_tree "$SP"
  SP=""
}
trap 'stop_server' EXIT
trap 'log "terminated"; stop_server; exit 143' TERM INT

log "replay waiting for the gate GPU $GPU_UUID; configs: $CONFIGS; seeds: $SEEDS"
while [ "$(date +%s)" -lt "$NOT_BEFORE_EPOCH" ]; do sleep 60; done
wait_for_gate gate_check "$GIVE_UP_HOURS"
log "gate open; starting the replay"
date '+%F %T %Z' > "$RUN_DIR/e1_started_at"
nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader > "$RUN_DIR/nvidia_smi_before.txt"

dataset_args() {  # name -> DS_PATH, DS_TOOLS, DS_N, DS_ON
  case "$1" in
    wildchat) DS_PATH="$ROOT/data/wildchat-1m/data/train-00008-of-00014.parquet"; DS_TOOLS=""; DS_N=$N_CHAT; DS_ON=$O_N_CHAT ;;
    smith) DS_PATH="$ROOT/data/swe-smith-trajectories/data/tool-00004-of-00008.parquet"; DS_TOOLS=""; DS_N=$N_SMITH; DS_ON=$O_N_SMITH ;;
    rebench) DS_PATH="$ROOT/data/swe-rebench-openhands/trajectories.parquet"; DS_TOOLS="$ROOT/data/swe-rebench-openhands/tools.json"; DS_N=$N_REBENCH; DS_ON=$O_N_REBENCH ;;
    *) log "unknown dataset $1"; exit 2 ;;
  esac
}

STATUS=0
for CFG in $CONFIGS; do
  TEMPLATE=""
  case "${CFG%[a-z]}" in
    F0) MODEL="$QWEN3_DIR"; NAME=qwen3; PARSER=hermes; EXTRA=() ;;
    F1) MODEL="$QWEN3_DIR"; NAME=qwen3; PARSER=qwen3_xml; TEMPLATE="$REPO/refs/qwen3.5-4b_chat_template.jinja"
        EXTRA=(--chat-template "$TEMPLATE") ;;
    C0) MODEL="$QWEN35_DIR"; NAME=qwen35; PARSER=qwen3_xml; EXTRA=(--language-model-only) ;;
    C1) MODEL="$QWEN35_DIR"; NAME=qwen35; PARSER=qwen3_xml; EXTRA=(--language-model-only --prefix-match-unit "$UNIT") ;;
    C2) MODEL="$QWEN35_DIR"; NAME=qwen35; PARSER=qwen3_xml; EXTRA=(--language-model-only --prefix-cache-retention-interval 576) ;;
    C3) MODEL="$QWEN35_DIR"; NAME=qwen35; PARSER=qwen3_xml; EXTRA=(--language-model-only --prefix-match-unit 16 --prefix-cache-retention-interval 576) ;;
    C4) MODEL="$QWEN35_DIR"; NAME=qwen35; PARSER=qwen3_xml; EXTRA=(--language-model-only --prefix-match-unit 16 --prefix-cache-retention-interval None) ;;
    *) log "unknown config $CFG"; exit 2 ;;
  esac

  idle=0
  for _ in 1 2 3; do if gate_check; then idle=1; break; fi; sleep 60; done
  if [ $idle -ne 1 ]; then
    log "gate GPU is busy before $CFG; stopping so as not to crowd it"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
    STATUS=4; break
  fi
  if [ "$(ss -ltn 2>/dev/null | grep -c ":$PORT ")" -gt 0 ]; then log "port $PORT in use"; STATUS=6; break; fi

  LOG="$RUN_DIR/serve_$CFG.log"
  CMD=("$VLLM" serve "$MODEL" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT"
       --dtype bfloat16 --max-model-len "$MAX_LEN" --kv-cache-memory-bytes "$KV_BYTES"
       --enable-prefix-caching --enable-prompt-tokens-details --generation-config vllm --seed 0
       --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser "$PARSER" "${EXTRA[@]}")
  # shellcheck disable=SC2206  # split on spaces on purpose
  [ -n "$SERVE_EXTRA" ] && CMD+=($SERVE_EXTRA)
  log "$CFG server command: CUDA_VISIBLE_DEVICES=$GPU_UUID ${CMD[*]}"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "${CMD[@]}" > "$LOG" 2>&1 &
  SP=$!
  log "$CFG server pid $SP, log $LOG"
  if ! wait_healthy "$URL" "$SP" 360; then
    log "$CFG server failed to come up"; grep -E "Error|error" "$LOG" | tail -n 8
    stop_server; STATUS=1; continue
  fi
  log "$CFG healthy. children: $(pgrep -P "$SP" | tr '\n' ' ')"
  grep -i -E "block_size|block size|mamba|prefix_match_unit|hash_block_size|retention|Maximum concurrency|GPU KV cache size|num_gpu_blocks|Chat template|kv_cache_memory" "$LOG" \
    | cut -c1-400 | head -n 40 > "$RUN_DIR/serve_${CFG}_facts.txt"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$RUN_DIR/nvidia_smi_$CFG.txt"

  case " $CONTROL_ONLY " in *" $CFG "*)
    # Start-up check: reset the cache and send the three control requests, no sessions.
    dataset_args wildchat
    TAG="${CFG}_controls_only"
    "$PY" "$REPLAY" --url "$URL" --model "$NAME" --tokenizer-dir "$MODEL" ${TEMPLATE:+--chat-template "$TEMPLATE"} --dataset wildchat --path "$DS_PATH" --seed 0 --n-sessions 0 --thinking on --mode R --max-len "$MAX_LEN" --hybrid-block "$HYBRID_BLOCK" --out-cap "$OUT_CAP" --label "$TAG" --out "$OUT/$TAG.jsonl" >> "$OUT/client_$CFG.log" 2>&1
    rc=$?
    log "$TAG rc=$rc $(grep -E 'controls ok|CONTROL FAILED|reset failed|unreachable' "$OUT/client_$CFG.log" | tail -n 1 | cut -c1-200)"
    [ $rc -eq 0 ] || STATUS=7
    stop_server; sleep 10; continue ;;
  esac

  for SEED in $SEEDS; do
    for DS in $DATASETS; do
      dataset_args "$DS"
      for THINK in $THINKS; do
        for MODE in $MODES; do
          for S in $INTERLEAVES; do
            if [ "$S" != 1 ]; then
              [ "$MODE" = R ] || continue
              case " $MULTI_CONFIGS " in *" $CFG "*) ;; *) continue ;; esac
            fi
            if [ "$MODE" = O ]; then
              case " $O_CONFIGS " in *" $CFG "*) ;; *) continue ;; esac
            fi
            if [ "$MODE" = O ]; then N=$DS_ON; CAP=$O_OUT_CAP; else N=$DS_N; CAP=$OUT_CAP; fi
            [ "$N" -gt 0 ] || continue
            TAG="${CFG}_${DS}_think${THINK}_${MODE}_s${S}_seed${SEED}"
            "$PY" "$REPLAY" --url "$URL" --model "$NAME" \
              --tokenizer-dir "$MODEL" ${TEMPLATE:+--chat-template "$TEMPLATE"} \
              --dataset "$DS" --path "$DS_PATH" ${DS_TOOLS:+--tools-json "$DS_TOOLS"} \
              --seed "$SEED" --n-sessions "$N" --thinking "$THINK" --mode "$MODE" \
              --max-len "$MAX_LEN" --hybrid-block "$HYBRID_BLOCK" --out-cap "$CAP" --interleave "$S" --label "$TAG" \
              --out "$OUT/$TAG.jsonl" >> "$OUT/client_$CFG.log" 2>&1
            rc=$?
            log "$TAG rc=$rc $(tail -n 1 "$OUT/client_$CFG.log" | cut -c1-300)"
            if [ $rc -eq 4 ] || [ $rc -eq 5 ]; then
              log "control or cache reset failed in $TAG: stop rule, ending the whole run"
              stop_server; STATUS=7; break 6
            fi
            [ $rc -eq 0 ] || STATUS=1
            kill -0 "$SP" 2>/dev/null || { log "$CFG server died during $TAG"; STATUS=1; break 5; }
          done
        done
      done
    done
  done
  # Diagnostic run (not part of any comparison): SWE-smith, thinking off; requests
  # whose stream carried no delta are sent again without streaming to see what the
  # model wrote. Output goes to $RUN_DIR/diag so the judges never read it.
  case " $DIAG_CONFIGS " in *" $CFG "*)
    if kill -0 "$SP" 2>/dev/null && [ "$STATUS" -ne 7 ]; then
      mkdir -p "$RUN_DIR/diag"
      dataset_args smith
      TAG="${CFG}_diag_smith_thinkoff_seed${SEEDS%% *}"
      "$PY" "$REPLAY" --url "$URL" --model "$NAME" \
        --tokenizer-dir "$MODEL" ${TEMPLATE:+--chat-template "$TEMPLATE"} \
        --dataset smith --path "$DS_PATH" --seed "${SEEDS%% *}" --n-sessions "$DS_N" \
        --thinking off --mode R --max-len "$MAX_LEN" --hybrid-block "$HYBRID_BLOCK" --out-cap "$OUT_CAP" \
        --diag-no-delta "$DIAG_N" --label "$TAG" --out "$RUN_DIR/diag/$TAG.jsonl" \
        >> "$RUN_DIR/diag/client_$CFG.log" 2>&1
      log "$TAG rc=$? (diagnostic)"
    fi ;;
  esac
  log "$CFG done. oom_or_dead=$(grep -c -E 'OutOfMemoryError|EngineDeadError' "$LOG")"
  stop_server
  sleep 10
done

nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader > "$RUN_DIR/nvidia_smi_after.txt"

# Offline accounting again with the block size this vLLM build reports (576). CPU only;
# runs after every server is stopped so it cannot disturb the TTFT measurements.
# Same sessions and sample sizes as the original accounting run (300 / 100 / 100).
if [ "$ACCOUNT" = 1 ] && [ "$STATUS" -ne 7 ]; then
  ACC_OUT="$RUN_DIR/e0b"; mkdir -p "$ACC_OUT"
  ACC_MODELS="qwen35=$QWEN35_DIR,qwen3=$QWEN3_DIR"
  account() {  # name n
    local seed rc=0
    dataset_args "$1"
    for seed in $ACCOUNT_SEEDS; do
      CUDA_VISIBLE_DEVICES= "$PY" "$REPO/experiments/e1_prefix_hits/account.py" --dataset "$1" --path "$DS_PATH" ${DS_TOOLS:+--tools-json "$DS_TOOLS"} --models "$ACC_MODELS" --seed "$seed" --n-sessions "$2" --max-len "$MAX_LEN" --hybrid-block 576 --out "$ACC_OUT" >> "$ACC_OUT/$1.log" 2>&1 || rc=1
    done
    return $rc
  }
  log "accounting (block 576) starting"
  account wildchat 300 & B1=$!
  account smith 100 & B2=$!
  account rebench 100 & B3=$!
  ACC_FAIL=0
  wait $B1 || ACC_FAIL=1; wait $B2 || ACC_FAIL=1; wait $B3 || ACC_FAIL=1
  tar -czf "$RUN_DIR/e0b_requests.tar.gz" -C "$ACC_OUT" . 2>/dev/null
  log "accounting finished, failed=$ACC_FAIL"
  [ "$ACC_FAIL" -eq 0 ] || STATUS=1
fi
"$PY" - "$OUT" "$RUN_DIR/summary.json" <<'PYEOF'
import glob, json, os, sys
out, dest = sys.argv[1], sys.argv[2]
items = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(out, "*.summary.json")))]
json.dump({"n_summaries": len(items), "summaries": items}, open(dest, "w", encoding="utf-8"), indent=1)
print("merged", len(items), "summaries")
PYEOF
tar -czf "$RUN_DIR/e1_requests.tar.gz" -C "$OUT" . 2>/dev/null
mkdir -p "$RUN_DIR/diag"
tar -czf "$RUN_DIR/diag_requests.tar.gz" -C "$RUN_DIR/diag" . 2>/dev/null
tar -czf "$RUN_DIR/serve_logs.tar.gz" -C "$RUN_DIR" $(cd "$RUN_DIR" && ls serve_*.log serve_*_facts.txt nvidia_smi_*.txt 2>/dev/null) 2>/dev/null
echo "{\"status\": $STATUS, \"finished_at\": \"$(date '+%F %T %Z')\"}" > "$RUN_DIR/metrics.json"
log "replay finished, status=$STATUS"
exit $STATUS
