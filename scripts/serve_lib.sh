# Shell helpers shared by the experiment launchers. Source it:  . scripts/serve_lib.sh
#
# The launchers run on a shared GPU server, so they follow three rules:
#   - nothing starts until the GPU is idle on three checks a minute apart (gate);
#   - processes are stopped by PID only, never by matching a name;
#   - a launcher never stops another user's process.

log() { echo "[$(date '+%F %T %Z')] $*"; }

# gpu_procs UUID  ->  prints "<own> <any>": compute processes on that GPU,
# of this user and of anybody.
gpu_procs() {
  local want="$1" mine=0 any=0 uuid pid owner
  while IFS=, read -r uuid pid; do
    uuid="${uuid// /}"; pid="${pid// /}"
    [ "$uuid" = "$want" ] || continue
    any=$((any + 1))
    owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
    [ "$owner" = "$(id -un)" ] && mine=$((mine + 1))
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
  echo "$mine $any"
}

# gpu_idle_once UUID [ANY_USER [MAX_UTIL [MAX_MEM_MIB]]]
# True when the GPU has none of our processes (and none of anybody's if ANY_USER=1),
# its utilisation is at most MAX_UTIL % and at most MAX_MEM_MIB MiB are in use.
gpu_idle_once() {
  local u="$1" any_user="${2:-0}" max_util="${3:-5}" max_mem="${4:-2048}" mine any util mem
  read -r mine any < <(gpu_procs "$u")
  util="$(nvidia-smi -i "$u" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  mem="$(nvidia-smi -i "$u" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  log "gate check $u: own processes = $mine, processes of any user = $any, utilization = ${util:-?} %, memory used = ${mem:-?} MiB"
  [ "$mine" -eq 0 ] && { [ "$any_user" != 1 ] || [ "$any" -eq 0 ]; } \
    && [ -n "$util" ] && [ "$util" -le "$max_util" ] && [ -n "$mem" ] && [ "$mem" -le "$max_mem" ]
}

# wait_for_gate CHECK_CMD GIVE_UP_HOURS
# Runs CHECK_CMD three times, a minute apart, until all three pass. Exits 3 when
# that never happens within GIVE_UP_HOURS.
wait_for_gate() {
  local check="$1" hours="${2:-12}" deadline
  deadline=$(( $(date +%s) + hours * 3600 ))
  while true; do
    if $check && sleep 60 && $check && sleep 60 && $check; then return 0; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then log "gate never opened within $hours h; giving up"; exit 3; fi
    sleep 300
  done
}

# descendants PID  ->  all descendants of a process, depth first.
descendants() {
  local c
  for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; descendants "$c"; done
}

# stop_tree PID...  ->  stop these processes and everything below them, by PID:
# TERM, wait up to 60 s, TERM again, then KILL what is left.
stop_tree() {
  local p pids="" n=0 alive
  for p in "$@"; do [ -n "$p" ] && pids="$pids $p $(descendants "$p" | tr '\n' ' ')"; done
  [ -n "${pids// /}" ] || return 0
  for p in $pids; do kill "$p" 2>/dev/null; done
  while [ $n -lt 30 ]; do
    alive=0
    for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && break
    sleep 2; n=$((n + 1))
  done
  for p in $pids; do
    if kill -0 "$p" 2>/dev/null; then log "process $p outlived its parent; kill $p"; kill "$p" 2>/dev/null; fi
  done
  sleep 5
  for p in $pids; do
    if kill -0 "$p" 2>/dev/null; then log "process $p ignores TERM; kill -9 $p"; kill -9 "$p" 2>/dev/null; fi
  done
  for p in "$@"; do [ -n "$p" ] && wait "$p" 2>/dev/null; done
  return 0
}

# wait_healthy URL PID TRIES  ->  poll URL/health every 5 s while PID lives.
wait_healthy() {
  local url="$1" pid="$2" tries="${3:-360}" i
  for i in $(seq 1 "$tries"); do
    curl -sf "$url/health" > /dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

# Environment every vLLM server and client of these experiments runs with.
# ROOT is the experiment root on the server (caches go below it).
export_serving_env() {
  local root="$1"
  export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$root/cache/hf"
  export VLLM_CACHE_ROOT="$root/cache/vllm" TRITON_CACHE_DIR="$root/cache/triton"
  export XDG_CACHE_HOME="$root/cache/xdg" FLASHINFER_WORKSPACE_BASE="$root/cache"
  export TMPDIR="$root/tmp" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export VLLM_SERVER_DEV_MODE=1      # for POST /reset_prefix_cache between runs
  # FlashInfer's sampler is JIT-compiled with the system nvcc, which was too old on
  # the server used here. All requests are greedy, so the sampler does not change
  # any output.
  export VLLM_USE_FLASHINFER_SAMPLER=0
  mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
}
