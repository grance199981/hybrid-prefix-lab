"""E10 step 1: replay the E5 S-tier workload through vLLM's own Scheduler and
KVCacheManager on the CPU, with fake model outputs.

The scheduler is built from the vllm_config and kv_cache_config that the logging hook
pickled inside a real server (scheduler_init_<pid>.pkl); only the number of blocks is
replaced. The engine loop follows EngineCore.step (synchronous) or
EngineCore.step_with_batch_queue (when max_concurrent_batches > 1) at commit 1479deb0,
without a model: a request that has all its prompt tokens scheduled after a step gets
one sampled token. The logging hook is applied in-process, so the events have the same
format as on the GPU and analyze_e10.py reads both.

Workload (E5 S tier): n_sessions sessions run at once, one per worker; turn k of a
session is the first p0 + k*step tokens of the session's own random array; a worker adds
its next turn `gap` engine steps after the previous turn finished. Each request decodes
exactly n_out tokens.

Writes <out>/<label>.jsonl (one row per request, the pressure_replay.py fields that the
analysis uses) and <out>/hook/events_<pid>.jsonl. Needs vLLM (run it with the venv's
python, CUDA_VISIBLE_DEVICES empty).

Usage: cpu_replay.py --init scheduler_init.pkl --blocks 192 --n-out 16 --seed 1 \
         --out DIR [--gap 1] [--n-sessions 16 --turns 12 --p0 2000 --step 250]
"""
import argparse
import collections
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def synth_prompts(n_sessions, turns, p0, step, seed):
    """Same structure as pressure_sessions.synth_sessions: each prompt extends the previous one,
    and the first token tells the sessions apart (the session index is the rank of the
    first token). Token values are arbitrary ids; only their equality matters here."""
    import random
    total = p0 + (turns - 1) * step
    out = []
    for i in range(n_sessions):
        rng = random.Random(f"e10-cpu-{seed}-{i}")
        arr = [1000 + rng.randrange(4096) for _ in range(total)]
        arr[0] = 10 + i
        out.append([arr[:p0 + k * step] for k in range(turns)])
    return out


def checkpoint(P, unit=16):
    return max((P - 1) // unit * unit, 0)


def build(init_path, blocks):
    with open(init_path, "rb") as f:
        d = pickle.load(f)
    vllm_config, kv_cache_config = d["vllm_config"], d["kv_cache_config"]
    assert vllm_config is not None, "the pickle holds no vllm_config"
    kv_cache_config.num_blocks = blocks
    vllm_config.cache_config.num_gpu_blocks = blocks

    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.structured_output import StructuredOutputManager

    kw = d["kwargs"]
    hash_fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
    init_none_hash(hash_fn)
    hasher = get_request_block_hasher(kw["hash_block_size"], hash_fn, vllm_config.num_hash_lookahead_tokens)
    cls = vllm_config.scheduler_config.get_scheduler_cls()
    assert cls.__name__ == d["cls"], (cls.__name__, d["cls"])
    sched = cls(vllm_config=vllm_config, kv_cache_config=kv_cache_config,
                structured_output_manager=StructuredOutputManager(vllm_config),
                include_finished_set=kw.get("include_finished_set", False), log_stats=False,
                block_size=kw["block_size"], hash_block_size=kw["hash_block_size"])
    return sched, hasher, vllm_config


class Engine:
    """EngineCore.step / step_with_batch_queue without a model executor."""

    def __init__(self, sched, queue_size):
        self.sched = sched
        self.queue = collections.deque() if queue_size > 1 else None
        self.queue_size = queue_size
        self.steps = 0

    def _fake_output(self, out):
        from vllm.v1.outputs import ModelRunnerOutput
        req_ids = list(out.num_scheduled_tokens)
        sampled = []
        for rid in req_ids:
            r = self.sched.requests.get(rid)
            sampled.append([7] if r is not None and r.num_computed_tokens >= r.num_prompt_tokens else [])
        return ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=sampled)

    def step(self):
        s = self.sched
        if self.queue is None:
            if not s.has_requests():
                return False
            out = s.schedule(False)
            s.update_from_output(out, self._fake_output(out))
            self.steps += 1
            return True
        q = self.queue
        if s.has_requests():
            out = s.schedule(False)
            executed = out.total_num_scheduled_tokens > 0
            q.appendleft((out, self._fake_output(out)))
            if len(q) < self.queue_size and (executed or s.has_requests()):
                self.steps += 1
                return True
        elif not q:
            return False
        out, mro = q.pop()
        s.update_from_output(out, mro)
        self.steps += 1
        return True


def run(a):
    sys.path.insert(0, os.path.join(HERE, "hook"))
    os.makedirs(os.path.join(a.out, "hook"), exist_ok=True)
    os.environ["E10_HOOK_DIR"] = os.path.join(a.out, "hook")
    import e10_hook
    import vllm.v1.core.block_pool as bp
    import vllm.v1.core.kv_cache_manager as km
    import vllm.v1.core.sched.scheduler as sm
    for name, mod in (("vllm.v1.core.sched.scheduler", sm), ("vllm.v1.core.kv_cache_manager", km),
                      ("vllm.v1.core.block_pool", bp)):
        e10_hook.apply(name, mod)

    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    sched, hasher, vllm_config = build(a.init, a.blocks)
    engine = Engine(sched, getattr(vllm_config, "max_concurrent_batches", 1))
    sessions = synth_prompts(a.n_sessions, a.turns, a.p0, a.step, a.seed)
    label = f"cpu_kv{a.blocks}b_out{a.n_out}_seed{a.seed}_gap{a.gap}"
    rows, live, due = [], {}, {}           # live: worker -> (k, Request, row); due: worker -> step to add next
    nxt = {w: 0 for w in range(a.n_sessions)}

    def add(w):
        k = nxt[w]
        P = len(sessions[w][k])
        sp = SamplingParams(max_tokens=a.n_out, min_tokens=a.n_out, ignore_eos=True)
        req = Request(request_id=f"s{w}-k{k}", prompt_token_ids=sessions[w][k], sampling_params=sp,
                      pooling_params=None, block_hasher=hasher)
        sched.add_request(req)
        row = {"label": label, "sid": f"synth-{w}", "k": k, "P": P, "n_out": a.n_out,
               "pred_inf": checkpoint(len(sessions[w][k - 1])) if k else 0,
               "t_send_ns": time.time_ns(), "step_send": engine.steps}
        rows.append(row)
        live[w] = (req, row)
        nxt[w] = k + 1

    for w in range(a.n_sessions):
        add(w)
    max_steps = a.max_steps
    peak = 0      # most blocks referenced at once (pool size minus free queue, cached-free blocks count as free)
    while live or due:
        for w in [w for w, t in due.items() if engine.steps >= t]:
            del due[w]
            add(w)
        if not engine.step() and not due:
            break
        pool = sched.kv_cache_manager.block_pool
        peak = max(peak, pool.num_gpu_blocks - pool.get_num_free_blocks())
        for w, (req, row) in list(live.items()):
            if req.is_finished():
                row["t_end_ns"] = time.time_ns()
                row["step_end"] = engine.steps
                row["preemptions"] = int(req.num_preemptions)
                del live[w]
                if nxt[w] < a.turns:
                    due[w] = engine.steps + a.gap
        if engine.steps > max_steps:
            raise SystemExit(f"no progress after {max_steps} steps; live {sorted(live)}")
    path = os.path.join(a.out, f"{label}.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"label": label, "requests": len(rows), "steps": engine.steps,
                      "scheduler": type(sched).__name__, "queue": engine.queue_size, "peak_in_use": peak}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--n-out", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gap", type=int, default=1, help="engine steps between a turn's end and the next turn")
    ap.add_argument("--n-sessions", type=int, default=16)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--p0", type=int, default=2000)
    ap.add_argument("--step", type=int, default=250)
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--out", required=True)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
