#!/usr/bin/env bash
# E6 on the GPU server: the four-pass design of E2 (C0a C1a C1b C0b, seeds 4-6,
# R replay one session at a time, output cap 16) on Qwen3.5-27B-FP8, with half the
# sessions (15 / 5 / 3 per seed), a 10 GiB KV cache, block size 832 and
# --max-num-seqs 64. About 3.5 hours on one L40.
#
# --max-num-seqs 64: with the default 256 the server needs 256 linear-attention state
# blocks; an 8 GiB cache held only 157 (vLLM refuses to start), and a 13.5 GiB cache
# did not fit next to the 27.9 GiB of FP8 weights on a 48 GB GPU. Requests are sent
# one at a time, so the limit does not change what is measured; both configs run with it.
#
# Usage: HP_ROOT=... HP_ENV=... HP_QWEN35_DIR=<Qwen3.5-27B-FP8> HP_QWEN3_DIR=<Qwen3-4B> \
#        GATE_GPU_UUID=<gpu-uuid> [taskset -c <cores of the GPU's NUMA node>] \
#        experiments/e6_27b/run.sh RUN_DIR
# Then: python experiments/e2_ttft_cost/judge.py RUN_DIR/e1 --json judge.json
#       python experiments/e2_ttft_cost/locate_overhead.py RUN_DIR/e1 --block 832 --json locate.json
# Any REPLAY_* variable set beforehand overrides the value below (see scripts/run_replay.sh).
set -u
: "${REPLAY_CONFIGS:=C0a C1a C1b C0b}"
: "${REPLAY_SEEDS:=4 5 6}"
: "${REPLAY_DATASETS:=wildchat smith rebench}"
: "${REPLAY_THINKS:=on off}"
: "${REPLAY_MODES:=R}"
: "${REPLAY_INTERLEAVES:=1}"
: "${REPLAY_MULTI_CONFIGS:=}" "${REPLAY_O_CONFIGS:=}"
: "${REPLAY_N_CHAT:=15}" "${REPLAY_N_SMITH:=5}" "${REPLAY_N_REBENCH:=3}"
: "${REPLAY_O_N_CHAT:=0}" "${REPLAY_O_N_SMITH:=0}" "${REPLAY_O_N_REBENCH:=0}"
: "${REPLAY_OUT_CAP:=16}" "${REPLAY_O_OUT_CAP:=512}"
: "${REPLAY_MAX_LEN:=32768}"
: "${REPLAY_KV_BYTES:=10737418240}"      # 10 GiB
: "${REPLAY_HYBRID_BLOCK:=832}"
: "${REPLAY_SERVE_EXTRA:=--max-num-seqs 64}"
: "${REPLAY_PORT:=8063}"
: "${REPLAY_DIAG_CONFIGS:=}"
export REPLAY_CONFIGS REPLAY_SEEDS REPLAY_DATASETS REPLAY_THINKS REPLAY_MODES REPLAY_INTERLEAVES \
  REPLAY_MULTI_CONFIGS REPLAY_O_CONFIGS REPLAY_N_CHAT REPLAY_N_SMITH REPLAY_N_REBENCH \
  REPLAY_O_N_CHAT REPLAY_O_N_SMITH REPLAY_O_N_REBENCH REPLAY_OUT_CAP REPLAY_O_OUT_CAP \
  REPLAY_MAX_LEN REPLAY_KV_BYTES REPLAY_HYBRID_BLOCK REPLAY_SERVE_EXTRA REPLAY_PORT REPLAY_DIAG_CONFIGS
exec "$(dirname "${BASH_SOURCE[0]}")/../../scripts/run_replay.sh" "$1"
