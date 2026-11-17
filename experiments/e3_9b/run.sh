#!/usr/bin/env bash
# E3 on the GPU server: the four-pass design of E2 (C0a C1a C1b C0b, seeds 4-6,
# 30 / 10 / 6 sessions, R replay one session at a time, output cap 16) on Qwen3.5-9B,
# with a 16 GiB KV cache (the 9B weights take about 18 GiB) and no diagnostic run.
# About 3 hours on one L40.
#
# Usage: HP_ROOT=... HP_ENV=... HP_QWEN35_DIR=<Qwen3.5-9B> HP_QWEN3_DIR=<Qwen3-4B> \
#        GATE_GPU_UUID=<gpu-uuid> experiments/e3_9b/run.sh RUN_DIR
# Then judge.py and locate_overhead.py of experiments/e2_ttft_cost on RUN_DIR/e1, and
# experiments/e3_9b/summarize.py on the two results directories.
# Any REPLAY_* variable set beforehand overrides the value below (see scripts/run_replay.sh).
set -u
: "${REPLAY_CONFIGS:=C0a C1a C1b C0b}"
: "${REPLAY_SEEDS:=4 5 6}"
: "${REPLAY_DATASETS:=wildchat smith rebench}"
: "${REPLAY_THINKS:=on off}"
: "${REPLAY_MODES:=R}"
: "${REPLAY_INTERLEAVES:=1}"
: "${REPLAY_MULTI_CONFIGS:=}" "${REPLAY_O_CONFIGS:=}"
: "${REPLAY_N_CHAT:=30}" "${REPLAY_N_SMITH:=10}" "${REPLAY_N_REBENCH:=6}"
: "${REPLAY_O_N_CHAT:=0}" "${REPLAY_O_N_SMITH:=0}" "${REPLAY_O_N_REBENCH:=0}"
: "${REPLAY_OUT_CAP:=16}" "${REPLAY_O_OUT_CAP:=512}"
: "${REPLAY_MAX_LEN:=32768}"
: "${REPLAY_KV_BYTES:=17179869184}"      # 16 GiB
: "${REPLAY_HYBRID_BLOCK:=576}"
: "${REPLAY_DIAG_CONFIGS:=}"
export REPLAY_CONFIGS REPLAY_SEEDS REPLAY_DATASETS REPLAY_THINKS REPLAY_MODES REPLAY_INTERLEAVES \
  REPLAY_MULTI_CONFIGS REPLAY_O_CONFIGS REPLAY_N_CHAT REPLAY_N_SMITH REPLAY_N_REBENCH \
  REPLAY_O_N_CHAT REPLAY_O_N_SMITH REPLAY_O_N_REBENCH REPLAY_OUT_CAP REPLAY_O_OUT_CAP \
  REPLAY_MAX_LEN REPLAY_KV_BYTES REPLAY_HYBRID_BLOCK REPLAY_DIAG_CONFIGS
exec "$(dirname "${BASH_SOURCE[0]}")/../../scripts/run_replay.sh" "$1"
