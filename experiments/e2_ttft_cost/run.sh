#!/usr/bin/env bash
# E2 on the GPU server: four passes C0a C1a C1b C0b (C0 = defaults, C1 =
# --prefix-match-unit 16) of the same requests on Qwen3.5-4B, seeds 4-6, R replay one
# session at a time, 30 / 10 / 6 sessions per seed, output cap 16, KV cache 24 GiB.
# After C0b the no-delta diagnostic run (SWE-smith, thinking off, seed 4) goes to
# RUN_DIR/diag. About 2 hours on one L40.
#
# Usage: HP_ROOT=... HP_ENV=... HP_QWEN35_DIR=<Qwen3.5-4B> HP_QWEN3_DIR=<Qwen3-4B> \
#        GATE_GPU_UUID=<gpu-uuid> experiments/e2_ttft_cost/run.sh RUN_DIR
# Then:  python experiments/e2_ttft_cost/judge.py RUN_DIR/e1 --json judge.json
#        python experiments/e2_ttft_cost/locate_overhead.py RUN_DIR/e1 --block 576 --json locate.json
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
: "${REPLAY_KV_BYTES:=25769803776}"      # 24 GiB
: "${REPLAY_DIAG_CONFIGS:=C0b}" "${REPLAY_DIAG_N:=40}"
export REPLAY_CONFIGS REPLAY_SEEDS REPLAY_DATASETS REPLAY_THINKS REPLAY_MODES REPLAY_INTERLEAVES \
  REPLAY_MULTI_CONFIGS REPLAY_O_CONFIGS REPLAY_N_CHAT REPLAY_N_SMITH REPLAY_N_REBENCH \
  REPLAY_O_N_CHAT REPLAY_O_N_SMITH REPLAY_O_N_REBENCH REPLAY_OUT_CAP REPLAY_O_OUT_CAP \
  REPLAY_MAX_LEN REPLAY_KV_BYTES REPLAY_DIAG_CONFIGS REPLAY_DIAG_N
exec "$(dirname "${BASH_SOURCE[0]}")/../../scripts/run_replay.sh" "$1"
