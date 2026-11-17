"""Concurrent replay client: send multi-turn sessions with several requests in
flight and record, per request, what the server reused next to what an
unlimited cache would have given (pred_inf).

Every request is a streamed /v1/completions call with a token-id prompt, so the
server sees exactly the prompt the accounting was done on. max_tokens =
min_tokens = --n-out, temperature 0.

Modes:
  closed  --inflight C workers; each takes the next session from a queue
          shuffled by the seed and sends its turns back to back.
  window  a fixed schedule (build_schedule): --live N sessions are active, the
          next request is the next turn of one of them picked at random (seeded);
          a finished session is replaced by the next one in the queue. The
          schedule is walked in order with at most C requests in flight, and
          never two turns of one session at once. `seq` is the position in the
          schedule. Requests dispatched at the same moment can reach the server
          in another order (thread timing); t_send_ns is recorded per request.

Before the sessions, on the same code path: /health, a prefix-cache reset
(needs VLLM_SERVER_DEV_MODE=1), three control requests (A must miss, A again
must hit, unrelated B must miss), and another reset. /metrics is sampled every
--metrics-interval seconds; the counters are read before the controls and after
the run and compared with the per-request sums in <out stem>.summary.json.

Outputs: <out>.jsonl (one record per request, controls first),
<out stem>.summary.json, <out stem>.metrics.jsonl (the /metrics samples).

Exit codes: 0 ok, 1 request errors, 4 control failed, 5 cache reset failed,
6 server unreachable.

Usage:
  python experiments/e5_cache_pressure/pressure_replay.py --url http://127.0.0.1:8062 \
      --model qwen3 --tokenizer-dir DIR --workload synth --seed 1 --n-sessions 16 \
      --turns 12 --p0 2000 --step 250 --n-out 16 --mode closed --inflight 16 \
      --arch fa64 --out F0_S_synth_kv11241_out16_seed1.jsonl
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.parse

from hplab.client import completion_body, http_get, read_metrics, reset_prefix_cache, stream_completion
from hplab.prompts import token_ids
from pressure_sessions import (ARCHS, cross_session_prefix, fit, infinite_cache_prediction,
                               lcp_table, real_sessions, synth_sessions)

NL = chr(10)
GAUGES = ["vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting"]
COUNTERS = ["vllm:num_preemptions", "vllm:prefix_cache_queries", "vllm:prefix_cache_hits",
            "vllm:kv_block_idle_before_evict_seconds_count"]
CONTROL_OUT = 8


# ------------------------------------------------------------------- schedule
def session_queue(sessions, seed):
    """The order in which sessions are started."""
    queue = list(sessions)
    random.Random(seed).shuffle(queue)
    return queue


def build_schedule(sessions, mode, live, seed):
    """window: [(sid, k)] in dispatch order; every request once, each session's
    turns in order, never more than `live` sessions started and unfinished.
    closed: None (the workers take sessions from session_queue as they go)."""
    if mode == "closed":
        return None
    rng = random.Random(seed)
    queue = list(sessions)
    rng.shuffle(queue)                    # same order as session_queue(sessions, seed)
    active, schedule = [[s, 0] for s in queue[:live]], []
    nxt = len(active)
    while active:
        i = rng.randrange(len(active))
        s, k = active[i]
        schedule.append((s["sid"], k))
        active[i][1] = k + 1
        if k + 1 == len(s["prompts"]):
            if nxt < len(queue):
                active[i] = [queue[nxt], 0]
                nxt += 1
            else:
                active.pop(i)
    return schedule


# -------------------------------------------------------------------- metrics
def read_server_metrics(host, port):
    return read_metrics(host, port, set(GAUGES + COUNTERS))


class Sampler(threading.Thread):
    """One /metrics sample per interval, as JSON lines, until stop()."""
    def __init__(self, host, port, path, interval):
        super().__init__(daemon=True)
        self.host, self.port, self.path, self.interval = host, port, path, interval
        self.done = threading.Event()

    def run(self):
        with open(self.path, "w", encoding="utf-8", newline=NL) as f:
            while True:
                last = self.done.is_set()
                rec = {"t_ns": time.time_ns()}
                try:
                    rec.update(read_server_metrics(self.host, self.port))
                except Exception as e:  # noqa: BLE001 - recorded
                    rec["error"] = repr(e)[:200]
                f.write(json.dumps(rec) + NL)
                f.flush()
                if last:
                    break
                self.done.wait(self.interval)

    def stop(self):
        self.done.set()
        self.join()


# ------------------------------------------------------------------- requests
def run_controls(host, port, a, tok):
    ids_a = token_ids(tok, f"{a.label}-control-A-{a.seed}", a.control_len)
    ids_b = token_ids(tok, f"{a.label}-control-B-{a.seed}", a.control_len)
    rows = []
    for name, ids in (("A", ids_a), ("A_again", ids_a), ("B", ids_b)):
        rec = {"label": a.label, "sid": "control", "control": name, "k": 0, "P": len(ids)}
        try:
            rec.update(stream_completion(host, port, completion_body(a.model, ids, CONTROL_OUT)))
            rec["error"] = None
        except Exception as e:  # noqa: BLE001 - recorded
            rec["error"] = repr(e)[:300]
        rows.append(rec)
    return rows


def judge_controls(rows):
    """rows: the control records in order (A, A again, B). Returns the problems."""
    if any(r.get("error") for r in rows):
        return ["request error"]
    a, a2, b = rows
    problems = []
    if a["cached_tokens"] != 0:
        problems.append(f"first A hit {a['cached_tokens']} tokens after a cache reset")
    if not 0 < a2["cached_tokens"] < a2["prompt_tokens"]:
        problems.append(f"resent A hit {a2['cached_tokens']} of {a2['prompt_tokens']}")
    if b["cached_tokens"] != 0:
        problems.append(f"unrelated B hit {b['cached_tokens']} tokens")
    return problems


class Runner:
    """Sends the session requests and writes one record per request."""
    def __init__(self, a, host, port, sessions, pred, f):
        self.a, self.host, self.port, self.pred, self.f = a, host, port, pred, f
        self.prompts = {s["sid"]: s["prompts"] for s in sessions}
        self.lcps = {s["sid"]: lcp_table(s) for s in sessions}
        self.cond = threading.Condition()
        self.inflight, self.seq, self.busy, self.rows = 0, 0, set(), []

    def _one(self, sid, k, seq, at_send, worker=None):
        ids = self.prompts[sid][k]
        rec = {"label": self.a.label, "sid": sid, "k": k, "seq": seq, "P": len(ids),
               "n_out": self.a.n_out, "inflight_at_send": at_send, "worker": worker,
               "pred_inf": self.pred[(sid, k)], "lcp_prev": self.lcps[sid][k]}
        try:
            rec.update(stream_completion(self.host, self.port,
                                         completion_body(self.a.model, ids, self.a.n_out)))
            rec["error"] = None
        except Exception as e:  # noqa: BLE001 - recorded
            rec.update({"t_send_ns": None, "t_end_ns": None, "ttft_ms": None, "e2e_ms": None,
                        "prompt_tokens": None, "cached_tokens": None, "completion_tokens": None,
                        "error": repr(e)[:300]})
        with self.cond:
            self.rows.append(rec)
            self.f.write(json.dumps(rec) + NL)
            self.f.flush()
            self.inflight -= 1
            self.busy.discard(sid)
            self.cond.notify_all()

    def run_window(self, schedule, limit):
        threads = []
        for seq, (sid, k) in enumerate(schedule):
            with self.cond:
                self.cond.wait_for(lambda: self.inflight < limit and sid not in self.busy)
                self.inflight += 1
                self.busy.add(sid)
                at_send = self.inflight         # this request included
            t = threading.Thread(target=self._one, args=(sid, k, seq, at_send))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def run_closed(self, queue, limit):
        todo = iter(queue)

        def work(w):
            while True:
                with self.cond:
                    s = next(todo, None)
                if s is None:
                    return
                for k in range(len(s["prompts"])):
                    with self.cond:
                        seq, self.seq = self.seq, self.seq + 1
                        self.inflight += 1
                        at_send = self.inflight
                    self._one(s["sid"], k, seq, at_send, worker=w)

        threads = [threading.Thread(target=work, args=(w,)) for w in range(limit)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def load_sessions(a, tok):
    if a.workload == "synth":
        if None in (a.turns, a.p0, a.step):
            raise SystemExit("--workload synth needs --turns, --p0 and --step")
        vocab = token_ids(tok, "e5-vocab", 4096)
        return fit(synth_sessions(a.n_sessions, a.turns, a.p0, a.step, a.seed, vocab), a.max_len, a.n_out)
    if not a.path:
        raise SystemExit(f"--workload {a.workload} needs --path")
    return real_sessions(a.workload, a.path, a.tokenizer_dir, a.chat_template, a.seed, a.n_sessions,
                         a.thinking, a.max_len, a.n_out, tools_json=a.tools_json, tok=tok)


def run(a):
    u = urllib.parse.urlparse(a.url)
    host, port = u.hostname, u.port
    try:
        status, _ = http_get(host, port, "/health")
        if status != 200:
            raise RuntimeError(f"health {status}")
    except Exception as e:  # noqa: BLE001
        print(f"server unreachable: {e!r}", flush=True)
        return 6

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    sessions = load_sessions(a, tok)
    pred = infinite_cache_prediction(sessions, a.arch)
    schedule = build_schedule(sessions, a.mode, a.live, a.seed)
    n_req = sum(len(s["prompts"]) for s in sessions)
    print(f"{a.label}: {len(sessions)} sessions, {n_req} requests, mode {a.mode}, "
          f"in flight {a.inflight}, live {a.live}", flush=True)

    if not reset_prefix_cache(host, port):
        print("prefix cache reset failed", flush=True)
        return 5
    before = read_server_metrics(host, port)
    controls = run_controls(host, port, a, tok)
    problems = judge_controls(controls)
    stem = os.path.splitext(a.out)[0]
    with open(a.out, "w", encoding="utf-8", newline=NL) as f:
        for r in controls:
            f.write(json.dumps(r) + NL)
        if problems:
            print("CONTROL FAILED: " + "; ".join(problems), flush=True)
            return 4
        print("controls ok: " + json.dumps(
            [(r["control"], r["prompt_tokens"], r["cached_tokens"]) for r in controls]), flush=True)
        if not reset_prefix_cache(host, port):
            print("prefix cache reset failed after the controls", flush=True)
            return 5
        sampler = Sampler(host, port, a.metrics_out or stem + ".metrics.jsonl", a.metrics_interval)
        sampler.start()
        runner = Runner(a, host, port, sessions, pred, f)
        t0 = time.time()
        try:
            if a.mode == "window":
                runner.run_window(schedule, a.inflight)
            else:
                runner.run_closed(session_queue(sessions, a.seed), a.inflight)
        finally:
            wall = time.time() - t0
            sampler.stop()
    after = read_server_metrics(host, port)

    sent = [r for r in runner.rows + controls if r.get("prompt_tokens")]
    n_errors = sum(1 for r in runner.rows if r["error"])
    summary = {
        "label": a.label, "args": vars(a), "n_sessions": len(sessions), "n_requests": len(runner.rows),
        "n_errors": n_errors, "wall_seconds": wall,
        "sum_prompt_tokens": sum(r["prompt_tokens"] for r in sent),
        "sum_cached_tokens": sum(r["cached_tokens"] for r in sent),
        "sum_pred_inf": sum(r["pred_inf"] for r in runner.rows if not r["error"]),
        "counters_before": before, "counters_after": after,
        "delta": {n: after[n] - before[n] for n in COUNTERS if n in after and n in before},
        "controls": controls, "cross_prefix": cross_session_prefix(sessions),
    }
    with open(stem + ".summary.json", "w", encoding="utf-8", newline=NL) as f:
        json.dump(summary, f, indent=1)
    keys = ("n_sessions", "n_requests", "n_errors", "sum_prompt_tokens", "sum_cached_tokens",
            "sum_pred_inf", "wall_seconds")
    print(a.label, json.dumps({k: summary[k] for k in keys}), flush=True)
    return 1 if n_errors else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--workload", required=True, choices=["synth", "smith", "wildchat"])
    ap.add_argument("--path", default=None, help="parquet file of a recorded workload")
    ap.add_argument("--chat-template", default=None)
    ap.add_argument("--tools-json", default=None)
    ap.add_argument("--thinking", default="off", choices=["on", "off"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-sessions", type=int, required=True)
    ap.add_argument("--turns", type=int, default=None, help="synth: requests per session")
    ap.add_argument("--p0", type=int, default=None, help="synth: first prompt length")
    ap.add_argument("--step", type=int, default=None, help="synth: tokens added per turn")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--n-out", type=int, required=True, help="output tokens of every request")
    ap.add_argument("--mode", required=True, choices=["closed", "window"])
    ap.add_argument("--inflight", type=int, required=True, help="C: most requests in flight")
    ap.add_argument("--live", type=int, default=None, help="window: sessions active at once (default C)")
    ap.add_argument("--arch", required=True, choices=list(ARCHS), help="rule for pred_inf")
    ap.add_argument("--control-len", type=int, default=1500, help="prompt tokens of a control request")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics-out", default=None, help="default: <out stem>.metrics.jsonl")
    ap.add_argument("--metrics-interval", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.label is None:
        a.label = os.path.splitext(os.path.basename(a.out))[0]
    if a.live is None:
        a.live = a.inflight
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
