#!/usr/bin/env bash
# The four passes of two vLLM builds, interleaved and back to back in one run, so that a slow drift
# over the hours of the run cancels between the builds:
#   segment 1  base  C0a C1a
#   segment 2  PR    C0a C1a
#   segment 3  PR    C1b C0b   (with the diagnostic run after C0b)
#   segment 4  base  C1b C0b   (with the diagnostic run after C0b)
# Each segment is one call of scripts/run_replay.sh in its own sub-directory. Used for the E8
# follow-up F (REPLAY_SERVE_EXTRA="--max-num-seqs 8") and for round R3 of E7 (REPLAY_UNIT=64).
#
# BASE_ENV / PR_ENV are the two environments (each with bin/python and bin/vllm). Everything else
# (HP_*, REPLAY_*, GATE_*) is read by run_replay.sh from the environment and is the same for all
# four segments; this script sets only HP_ENV, REPLAY_CONFIGS and REPLAY_DIAG_CONFIGS.
# A segment that fails stops the run; the later segments are not started.
# Outputs: RUN_DIR/seg1_base ... seg4_base as written by run_replay.sh (e1/, diag/, serve logs),
# RUN_DIR/e1_requests.tar.gz, diag_requests.tar.gz, serve_logs.tar.gz, metrics.json ({"status": ...}).
# Judge each build by merging seg*_base/e1 into one directory and seg*_pr/e1 into another.
# Usage: bash experiments/e8_pr61952/run_followup_f.sh <run_dir>
set -uo pipefail

mkdir -p "${1:?usage: run_followup_f.sh <run_dir>}" && RUN_DIR="$(cd "$1" && pwd)" || exit 2
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$REPO/scripts/serve_lib.sh"
BASE="${BASE_ENV:?}"
PR="${PR_ENV:?}"
STATUS=0

seg() {  # name, env dir, configs, diagnostic configs
  mkdir -p "$RUN_DIR/$1"
  log "segment $1 starts: env $2, configs: $3"
  HP_ENV="$2" REPLAY_CONFIGS="$3" REPLAY_DIAG_CONFIGS="$4" bash "$REPO/scripts/run_replay.sh" "$RUN_DIR/$1"
  local rc=$?
  log "segment $1 finished, status=$rc"
  return $rc
}

seg seg1_base "$BASE" "C0a C1a" "" \
  && seg seg2_pr "$PR" "C0a C1a" "" \
  && seg seg3_pr "$PR" "C1b C0b" "C0b" \
  && seg seg4_base "$BASE" "C1b C0b" "C0b"
STATUS=$?
[ $STATUS -ne 0 ] && log "a segment failed; the later segments were not started"

cd "$RUN_DIR" || exit 1
tar -czf e1_requests.tar.gz $(ls -d seg*/e1 2>/dev/null) 2>/dev/null
tar -czf diag_requests.tar.gz $(ls -d seg*/diag 2>/dev/null) 2>/dev/null
tar -czf serve_logs.tar.gz $(ls seg*/serve_*.log seg*/serve_*_facts.txt seg*/nvidia_smi_*.txt 2>/dev/null) 2>/dev/null
echo "{\"status\": $STATUS, \"finished_at\": \"$(date '+%F %T %Z')\"}" > metrics.json
echo "{}" > summary.json
log "run finished, status=$STATUS"
exit $STATUS
