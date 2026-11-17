#!/usr/bin/env bash
# Offline accounting (E0) on the GPU server: account.py for the three datasets in
# parallel, seeds 1-3, both tokenizers, block 528 (as in the original run; the
# replay launcher's REPLAY_ACCOUNT=1 redoes it with 576). CPU only.
#
# Optional start gate: with GATE_GPU_UUID set, wait until NOT_BEFORE_EPOCH and until
# that GPU has none of this user's processes and at most 5 % utilisation on three
# checks 60 s apart (scripts/serve_lib.sh), so the CPU load does not disturb a GPU
# measurement of one's own that is still running.
#
# Usage: HP_ROOT=... HP_QWEN35_DIR=<Qwen3.5-4B> HP_QWEN3_DIR=<Qwen3-4B or its tokenizer> \
#        [HP_ENV=<env> | HP_PYTHON=<python>] experiments/e1_prefix_hits/run_account.sh RUN_DIR
# Output: RUN_DIR/e0/<dataset>_<qwen35|qwen3>_think<on|off>_seed<seed>.jsonl and
#         .summary.json, RUN_DIR/summary.json (merged), e0_requests.tar.gz, metrics.json.
# Exit codes: 0 ok, 1 an accounting run failed, 3 gate never opened.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$REPO/scripts/serve_lib.sh"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false   # one core per process

RUN_DIR="$1"
ROOT="${HP_ROOT:?}"
PY="${HP_PYTHON:-${HP_ENV:?set HP_ENV or HP_PYTHON}/bin/python}"
NOT_BEFORE_EPOCH="${NOT_BEFORE_EPOCH:-0}"
GIVE_UP_HOURS="${GIVE_UP_HOURS:-48}"
SEEDS="${ACCOUNT_SEEDS:-1 2 3}"
N_AGENT="${ACCOUNT_N_AGENT:-100}"
N_CHAT="${ACCOUNT_N_CHAT:-300}"
MAX_LEN="${ACCOUNT_MAX_LEN:-32768}"
MODELS="qwen35=${HP_QWEN35_DIR:?},qwen3=${HP_QWEN3_DIR:?}"

OUT="$RUN_DIR/e0"
mkdir -p "$OUT"

if [ -n "${GATE_GPU_UUID:-}" ]; then
  gate_check() { gpu_idle_once "$GATE_GPU_UUID" 0 5 1000000000; }   # utilisation only
  while [ "$(date +%s)" -lt "$NOT_BEFORE_EPOCH" ]; do sleep 60; done
  wait_for_gate gate_check "$GIVE_UP_HOURS"
fi
log "accounting starting"
date '+%F %T %Z' > "$RUN_DIR/e0_started_at"

run_dataset() {  # name parquet n_sessions [tools_json]
  local name="$1" path="$2" n="$3" tools="${4:-}" seed rc=0
  for seed in $SEEDS; do
    "$PY" "$REPO/experiments/e1_prefix_hits/account.py" --dataset "$name" --path "$path" \
      ${tools:+--tools-json "$tools"} --models "$MODELS" --seed "$seed" \
      --n-sessions "$n" --max-len "$MAX_LEN" --out "$OUT" \
      >> "$OUT/$name.log" 2>&1 || rc=1
  done
  return $rc
}

run_dataset wildchat "$ROOT/data/wildchat-1m/data/train-00008-of-00014.parquet" "$N_CHAT" & P1=$!
run_dataset smith "$ROOT/data/swe-smith-trajectories/data/tool-00004-of-00008.parquet" "$N_AGENT" & P2=$!
run_dataset rebench "$ROOT/data/swe-rebench-openhands/trajectories.parquet" "$N_AGENT" \
  "$ROOT/data/swe-rebench-openhands/tools.json" & P3=$!
FAIL=0
wait $P1 || FAIL=1
wait $P2 || FAIL=1
wait $P3 || FAIL=1

"$PY" - "$OUT" "$RUN_DIR/summary.json" <<'PYEOF'
import glob, json, os, sys
out, dest = sys.argv[1], sys.argv[2]
items = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(out, "*.summary.json")))]
json.dump({"n_summaries": len(items), "summaries": items}, open(dest, "w", encoding="utf-8"), indent=1)
print("merged", len(items), "summaries")
PYEOF
tar -czf "$RUN_DIR/e0_requests.tar.gz" -C "$OUT" . 2>/dev/null
echo "{\"failed\": $FAIL, \"finished_at\": \"$(date '+%F %T %Z')\"}" > "$RUN_DIR/metrics.json"
log "accounting finished, failed=$FAIL"
exit $FAIL
