#!/usr/bin/env bash
# E1 on the GPU server: seven server configs (F0 C1 C4 C0 F1 C3 C2), seeds 1-3, three
# datasets, thinking on and off; R replay one session at a time (all configs) and
# 8 sessions round-robin (F0 C0 C3); O replay (F0 F1 C1 C4); afterwards the offline
# accounting with block 576 (CPU). About 25 hours on one L40.
#
# Usage: HP_ROOT=... HP_ENV=... HP_QWEN35_DIR=<Qwen3.5-4B> HP_QWEN3_DIR=<Qwen3-4B> \
#        GATE_GPU_UUID=<gpu-uuid> experiments/e1_prefix_hits/run.sh RUN_DIR
# Any REPLAY_* variable set beforehand overrides the value below (see scripts/run_replay.sh).
set -u
: "${REPLAY_CONFIGS:=F0 C1 C4 C0 F1 C3 C2}"
: "${REPLAY_SEEDS:=1 2 3}"
: "${REPLAY_DATASETS:=wildchat smith rebench}"
: "${REPLAY_THINKS:=on off}"
: "${REPLAY_MODES:=R O}"
: "${REPLAY_INTERLEAVES:=1 8}"
: "${REPLAY_MULTI_CONFIGS:=F0 C0 C3}"
: "${REPLAY_O_CONFIGS:=F0 F1 C1 C4}"
: "${REPLAY_N_CHAT:=100}" "${REPLAY_N_SMITH:=30}" "${REPLAY_N_REBENCH:=20}"
: "${REPLAY_O_N_CHAT:=10}" "${REPLAY_O_N_SMITH:=6}" "${REPLAY_O_N_REBENCH:=6}"
: "${REPLAY_OUT_CAP:=16}" "${REPLAY_O_OUT_CAP:=512}"
: "${REPLAY_MAX_LEN:=32768}"
: "${REPLAY_KV_BYTES:=25769803776}"      # 24 GiB
: "${REPLAY_ACCOUNT:=1}"
export REPLAY_CONFIGS REPLAY_SEEDS REPLAY_DATASETS REPLAY_THINKS REPLAY_MODES REPLAY_INTERLEAVES \
  REPLAY_MULTI_CONFIGS REPLAY_O_CONFIGS REPLAY_N_CHAT REPLAY_N_SMITH REPLAY_N_REBENCH \
  REPLAY_O_N_CHAT REPLAY_O_N_SMITH REPLAY_O_N_REBENCH REPLAY_OUT_CAP REPLAY_O_OUT_CAP \
  REPLAY_MAX_LEN REPLAY_KV_BYTES REPLAY_ACCOUNT
exec "$(dirname "${BASH_SOURCE[0]}")/../../scripts/run_replay.sh" "$1"
