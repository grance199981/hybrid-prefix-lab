#!/usr/bin/env bash
# E10 step 1: every CPU replay listed in README.md, each in its own directory
# (the hook's events are per process and the analysis windows them by time), then the analysis.
#   run_cpu_replay.sh <python> <scheduler_init.pkl> <out_dir> [parallel jobs] [cpu list]
# CPU only (CUDA_VISIBLE_DEVICES is emptied). Reads nothing but the pickle; writes only under <out_dir>.
set -uo pipefail
PY=$1; PKL=$2; OUT=$3; JOBS=${4:-8}; CPUS=${5:-96-127}
SRC=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cp "$PKL" "$OUT/scheduler_init.pkl"
jobs_list() {
  for b in 192 202 211 288 336 384 432 480; do for o in 16 256; do for s in 1 2 3; do echo "$b $o $s 1"; done; done; done
  for b in 192 288; do for o in 16 256; do for s in 1 2 3; do for g in 0 3; do echo "$b $o $s $g"; done; done; done; done
}
run_one() {  # blocks out seed gap
  local d="$OUT/kv$1b_out$2_seed$3_gap$4"
  mkdir -p "$d"
  CUDA_VISIBLE_DEVICES= taskset -c "$CPUS" "$PY" "$SRC/cpu_replay.py" --init "$OUT/scheduler_init.pkl" \
    --blocks "$1" --n-out "$2" --seed "$3" --gap "$4" --out "$d" > "$d/replay.log" 2>&1
  echo "$1 $2 $3 $4 rc=$? $(tail -n 1 "$d/replay.log" | cut -c1-200)"
}
export -f run_one
export OUT PY SRC CPUS
echo "[$(date '+%F %T %Z')] start, $(jobs_list | wc -l) replays, $JOBS at a time on CPUs $CPUS"
jobs_list | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _ | sort -n > "$OUT/replays.txt"
echo "[$(date '+%F %T %Z')] replays done: $(grep -c 'rc=0' "$OUT/replays.txt") ok of $(wc -l < "$OUT/replays.txt")"
"$PY" "$SRC/analyze_e10.py" $(ls -d "$OUT"/kv*_gap1) --json "$OUT/analysis_gap1.json" > "$OUT/analysis_gap1.txt"
"$PY" "$SRC/analyze_e10.py" $(ls -d "$OUT"/kv*_gap0) --json "$OUT/analysis_gap0.json" > "$OUT/analysis_gap0.txt"
"$PY" "$SRC/analyze_e10.py" $(ls -d "$OUT"/kv*_gap3) --json "$OUT/analysis_gap3.json" > "$OUT/analysis_gap3.txt"
echo "[$(date '+%F %T %Z')] analysis done"
