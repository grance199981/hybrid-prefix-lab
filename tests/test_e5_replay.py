"""experiments/e5_cache_pressure/pressure_replay.py: the window schedule, and an
end-to-end run against a fake server (http.server, threaded, in-process) whose
cache policy is known:

  the fake server keeps whole prompts in an LRU list with a token budget and
  reports floor(min(D, P - 1) / 64) * 64, D = longest prefix shared with a
  prompt it still keeps.
  known good: a large budget -> every request hits exactly pred_inf (fa64 rule);
  known bad:  budget 0 -> nothing is ever hit, and the control request stops the run.
"""
import collections
import http.server
import json
import os
import sys
import threading
import time
import types
import zlib

import pytest

from hplab.client import parse_metrics

# the experiment's modules import each other by plain name, as when run as scripts
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "e5_cache_pressure"))
import pressure_replay as e5  # noqa: E402
from lru_sim import lcp  # noqa: E402
from pressure_sessions import synth_sessions  # noqa: E402


# ------------------------------------------------------------------- schedule
def sessions_of(n, seed=0):
    return [{"sid": f"s{i}", "prompts": [[i] * (10 + k) for k in range(2 + (i + seed) % 4)]}
            for i in range(n)]


@pytest.mark.parametrize("live", [1, 3, 8, 50])
def test_window_schedule_covers_every_request_once_in_session_order(live):
    sessions = sessions_of(11)
    sched = e5.build_schedule(sessions, "window", live, seed=4)
    assert sched == e5.build_schedule(sessions, "window", live, seed=4)
    assert sched != e5.build_schedule(sessions, "window", live, seed=5)
    assert sorted(sched) == sorted((s["sid"], k) for s in sessions for k in range(len(s["prompts"])))
    turns = {s["sid"]: len(s["prompts"]) for s in sessions}
    pos = {s["sid"]: q for q, s in enumerate(e5.session_queue(sessions, 4))}
    seen, active, peak, finished = collections.Counter(), set(), 0, 0
    for sid, k in sched:
        assert k == seen[sid]               # each session's turns in order
        if k == 0:                          # admitted in queue order, as a replacement:
            assert finished >= pos[sid] - live + 1      # enough earlier sessions are over
        seen[sid] += 1
        active.add(sid)
        peak = max(peak, len(active))
        if seen[sid] == turns[sid]:
            active.discard(sid)
            finished += 1
    assert peak <= min(live, len(sessions))      # started and unfinished never exceeds `live`
    if live == 1:                                 # one session at a time, in queue order
        assert sched == [(s["sid"], k) for s in e5.session_queue(sessions, 4)
                         for k in range(len(s["prompts"]))]


def test_closed_mode_has_no_global_schedule():
    assert e5.build_schedule(sessions_of(3), "closed", 2, seed=1) is None


def test_parse_metrics_sums_label_sets_and_leaves_out_absent_names():
    text = "\n".join([
        "# HELP vllm:prefix_cache_queries_total x",
        'vllm:prefix_cache_queries_total{engine="0"} 100.0',
        'vllm:prefix_cache_queries_total{engine="1"} 20.0',
        'vllm:prefix_cache_hits_total{engine="0"} 64.0',
        'vllm:prefix_cache_hits_created{engine="0"} 1.7e9',
        'vllm:kv_cache_usage_perc{engine="0"} 0.25',
        'vllm:kv_block_idle_before_evict_seconds_count{engine="0"} 7.0',
        'vllm:kv_block_idle_before_evict_seconds_sum{engine="0"} 99.0',
    ])
    assert parse_metrics(text, set(e5.GAUGES + e5.COUNTERS)) == {
        "vllm:prefix_cache_queries": 120.0, "vllm:prefix_cache_hits": 64.0,
        "vllm:kv_cache_usage_perc": 0.25, "vllm:kv_block_idle_before_evict_seconds_count": 7.0}


def test_judge_controls_good_and_bad():
    def rec(p, c, err=None):
        return {"prompt_tokens": p, "cached_tokens": c, "error": err}
    assert e5.judge_controls([rec(1500, 0), rec(1500, 1152), rec(1500, 0)]) == []
    assert e5.judge_controls([rec(1500, 64), rec(1500, 1472), rec(1500, 0)])     # stale cache
    assert e5.judge_controls([rec(1500, 0), rec(1500, 0), rec(1500, 0)])         # caching off
    assert e5.judge_controls([rec(1500, 0), rec(1500, 1472), rec(1500, 64)])     # B hit
    assert e5.judge_controls([rec(1500, 0), rec(None, 0, "boom"), rec(1500, 0)])


# ----------------------------------------------------------------- fake server
class FakeTok:
    """One token per word, as hplab.prompts.token_ids needs."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + zlib.crc32(w.encode()) % 50000 for w in text.split()]}


class Fake:
    def __init__(self, budget, delay=0.01):
        self.budget, self.delay = budget, delay
        self.lock = threading.Lock()
        self.kept = collections.OrderedDict()      # prompt tuple -> None, least recent first
        self.queries = self.hits = self.resets = 0
        self.inflight = self.max_inflight = 0
        self.flying_first = collections.Counter()  # first token id -> requests in flight
        self.same_session_overlap = False
        self.arrivals = []                         # (first token id, prompt length)

    def complete(self, req):
        ids = req["prompt"]
        with self.lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            self.same_session_overlap |= self.flying_first[ids[0]] > 0
            self.flying_first[ids[0]] += 1
            self.arrivals.append((ids[0], len(ids)))
            d = max((lcp(ids, list(p)) for p in self.kept), default=0)
            cached = min(d, len(ids) - 1) // 64 * 64
            self.queries += len(ids)
            self.hits += cached
        time.sleep(self.delay)                     # the request is "running"
        with self.lock:
            self.kept[tuple(ids)] = None
            self.kept.move_to_end(tuple(ids))
            while self.kept and sum(map(len, self.kept)) > self.budget:
                self.kept.popitem(last=False)
            self.inflight -= 1
            self.flying_first[ids[0]] -= 1
        usage = {"prompt_tokens": len(ids), "completion_tokens": req["max_tokens"],
                 "prompt_tokens_details": {"cached_tokens": cached} if cached else None}
        return [{"choices": [{"text": "x"}]}, {"choices": [{"text": "", "finish_reason": "length"}]},
                {"choices": [], "usage": usage}]

    def metrics(self):
        with self.lock:
            return (f'vllm:prefix_cache_queries_total{{engine="0"}} {float(self.queries)}\n'
                    f'vllm:prefix_cache_hits_total{{engine="0"}} {float(self.hits)}\n'
                    f'vllm:num_preemptions_total{{engine="0"}} 0.0\n'
                    f'vllm:kv_cache_usage_perc{{engine="0"}} 0.5\n'
                    f'vllm:num_requests_running{{engine="0"}} {float(self.inflight)}\n')


def make_handler(fake):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, text, ctype="text/plain"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()                     # HTTP/1.0: the body ends when the socket closes
            self.wfile.write(text.encode())

        def do_GET(self):
            self._send(fake.metrics() if self.path == "/metrics" else "ok")

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/reset_prefix_cache":
                with fake.lock:
                    fake.kept.clear()
                    fake.resets += 1
                return self._send(json.dumps({"success": True}), "application/json")
            chunks = fake.complete(json.loads(raw))
            self._send("".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n",
                       "text/event-stream")
    return Handler


@pytest.fixture
def client(monkeypatch, tmp_path):
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda d: FakeTok())
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    servers = []

    def go(budget, mode="window", inflight=3, live=4, extra=()):
        fake = Fake(budget)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(fake))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        out = str(tmp_path / f"{mode}_{budget}.jsonl")
        rc = e5.main(["--url", f"http://127.0.0.1:{srv.server_address[1]}", "--model", "m",
                      "--tokenizer-dir", "x", "--workload", "synth", "--seed", "3", "--n-sessions", "7",
                      "--turns", "4", "--p0", "130", "--step", "70", "--max-len", "4096", "--n-out", "4",
                      "--mode", mode, "--inflight", str(inflight), "--live", str(live), "--arch", "fa64",
                      "--control-len", "200", "--metrics-interval", "0.02", "--out", out, *extra])
        rows = [json.loads(line) for line in open(out, encoding="utf-8")] if os.path.exists(out) else []
        stem = out[:-6]
        summary = (json.load(open(stem + ".summary.json", encoding="utf-8"))
                   if os.path.exists(stem + ".summary.json") else None)
        metrics = ([json.loads(line) for line in open(stem + ".metrics.jsonl", encoding="utf-8")]
                   if os.path.exists(stem + ".metrics.jsonl") else [])
        return rc, rows, summary, metrics, fake

    yield go
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def the_sessions():
    """What the client builds for the arguments above (same vocabulary, same seed)."""
    vocab = e5.token_ids(FakeTok(), "e5-vocab", 4096)
    return synth_sessions(7, 4, 130, 70, 3, vocab)


def test_window_run_against_a_server_that_keeps_everything(client):
    rc, rows, summary, metrics, fake = client(budget=10**9, inflight=3, live=4)
    sess = [r for r in rows if r["sid"] != "control"]
    assert rc == 0 and fake.resets == 2 and [r["control"] for r in rows[:3]] == ["A", "A_again", "B"]
    assert len(sess) == 28 and all(r["error"] is None for r in sess)
    assert all(r["prompt_tokens"] == r["P"] and r["completion_tokens"] == 4 for r in sess)
    # the server follows the fa64 rule with nothing evicted: every request hits pred_inf
    assert all(r["cached_tokens"] == r["pred_inf"] for r in sess)
    assert all(r["cached_tokens"] > 0 for r in sess if r["k"] > 0)
    assert all(r["lcp_prev"] == {str(j): 130 + 70 * j for j in range(r["k"])} for r in sess)
    # concurrency: never more than C, really more than one, never two turns of a session
    assert 2 <= fake.max_inflight <= 3 and not fake.same_session_overlap
    assert all(1 <= r["inflight_at_send"] <= 3 for r in sess)
    # dispatch order is the schedule
    schedule = e5.build_schedule(the_sessions(), "window", 4, 3)
    assert [(r["sid"], r["k"]) for r in sorted(sess, key=lambda r: r["seq"])] == schedule
    # ... and the server saw every request once. A request is dispatched only when at most
    # C - 1 earlier ones are unfinished, so it cannot arrive more than C - 1 places early.
    # (How late it may arrive is up to thread timing; with one request in flight the
    # order is exact, see the next test.)
    first = {s["prompts"][0][0]: s["sid"] for s in the_sessions()}
    arrived = [(first[tok], (n - 130) // 70) for tok, n in fake.arrivals[3:]]
    assert sorted(arrived) == sorted(schedule)
    assert all(i >= schedule.index(req) - 2 for i, req in enumerate(arrived))
    # the summary reconciles with /metrics (controls included), and the sampler wrote samples
    d = summary["delta"]
    assert d["vllm:prefix_cache_queries"] == summary["sum_prompt_tokens"] == fake.queries
    assert d["vllm:prefix_cache_hits"] == summary["sum_cached_tokens"] == fake.hits
    assert d["vllm:num_preemptions"] == 0 and "vllm:kv_block_idle_before_evict_seconds_count" not in d
    assert summary["n_errors"] == 0 and summary["n_requests"] == 28 and summary["wall_seconds"] > 0
    assert set(summary["cross_prefix"].values()) == {0}
    assert len(metrics) >= 2 and all(m["vllm:kv_cache_usage_perc"] == 0.5 and "t_ns" in m for m in metrics)


def test_one_request_in_flight_reaches_the_server_in_schedule_order(client):
    rc, rows, _, _, fake = client(budget=10**9, inflight=1, live=3)
    first = {s["prompts"][0][0]: s["sid"] for s in the_sessions()}
    assert rc == 0 and fake.max_inflight == 1
    assert [(first[tok], (n - 130) // 70) for tok, n in fake.arrivals[3:]] == \
        e5.build_schedule(the_sessions(), "window", 3, 3)


def test_closed_run_sends_each_worker_s_sessions_back_to_back(client):
    rc, rows, summary, _, fake = client(budget=10**9, mode="closed", inflight=2)
    sess = sorted((r for r in rows if r["sid"] != "control"), key=lambda r: r["seq"])
    assert rc == 0 and len(sess) == 28 and [r["seq"] for r in sess] == list(range(28))
    assert fake.max_inflight == 2 and not fake.same_session_overlap
    assert all(r["cached_tokens"] == r["pred_inf"] for r in sess)
    for w in (0, 1):
        mine = [(r["sid"], r["k"]) for r in sess if r["worker"] == w]
        sids = list(dict.fromkeys(sid for sid, _ in mine))
        assert mine == [(sid, k) for sid in sids for k in range(4)]     # one session after another
    started = [r["sid"] for r in sess if r["k"] == 0]
    assert sorted(started) == sorted(s["sid"] for s in the_sessions())
    assert summary["delta"]["vllm:prefix_cache_hits"] == summary["sum_cached_tokens"]


def test_a_small_budget_shows_up_as_hits_below_the_prediction(client):
    # 400 tokens: enough for the control prompt (200), not for four live sessions
    rc, rows, summary, _, _ = client(budget=400, inflight=3, live=4)
    sess = [r for r in rows if r["sid"] != "control" and r["k"] > 0]
    assert rc == 0 and all(r["cached_tokens"] <= r["pred_inf"] for r in sess)
    assert sum(r["cached_tokens"] for r in sess) < 0.8 * sum(r["pred_inf"] for r in sess)
    assert summary["delta"]["vllm:prefix_cache_hits"] == summary["sum_cached_tokens"]


def test_control_stops_the_run_when_the_server_never_caches(client):
    rc, rows, summary, _, fake = client(budget=0)
    assert rc == 4 and summary is None
    assert [r["control"] for r in rows] == ["A", "A_again", "B"]
    assert all(r["cached_tokens"] == 0 for r in rows) and len(fake.arrivals) == 3


def test_unreachable_server_and_failed_reset(client, monkeypatch, tmp_path):
    out = str(tmp_path / "x.jsonl")
    args = ["--model", "m", "--tokenizer-dir", "x", "--workload", "synth", "--seed", "1",
            "--n-sessions", "2", "--turns", "2", "--p0", "130", "--step", "70", "--n-out", "4",
            "--mode", "window", "--inflight", "1", "--arch", "fa64", "--out", out]
    assert e5.main(["--url", "http://127.0.0.1:9", *args]) == 6
    monkeypatch.setattr(e5, "reset_prefix_cache", lambda host, port: False)
    rc, rows, summary, _, fake = client(budget=10**9)
    assert rc == 5 and rows == [] and summary is None and fake.arrivals == []
