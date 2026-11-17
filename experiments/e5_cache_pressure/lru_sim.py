"""A CPU-only model of vLLM's block pool (commit 1479deb0), to predict which
prefix-cache hits survive when the pool is tight. Standard library only.

Queue rules, as read from the vLLM source (vllm/v1/core/block_pool.py and
kv_cache_manager.py at 1479deb0):
  - one pool for attention blocks and, for the hybrid model, state blocks;
  - allocation pops from the FRONT of the free queue; a cached block popped
    this way loses its hash (lazy eviction);
  - a finished request frees its blocks tail first: blocks with a hash are
    appended to the back in that order, blocks without one go to the front;
  - a cache hit takes the hit blocks out of the free queue until the request ends.

Full attention ("fa", 64-token blocks): only full blocks get a hash; the hit is
the longest chain of cached blocks that open the prompt, at most (P - 1) // 64.

Hybrid ("hy", 576-token blocks, unit 576 or 16): a request holds 2 state blocks
and leaves one cached state checkpoint at c = floor((P - 1) / unit) * unit. A
later request hits at the largest checkpoint h whose state block is cached, that
lies inside the common prefix with its producer, and whose attention blocks for
[0, h) are all cached; for h inside a block that includes the producer's partly
filled tail block. With unit 16, a hit inside a block is copied (one more
attention block, one more state block), and writing a checkpoint inside a block
takes one more state block.
  rules M0: the checkpoint's state block is freed when the request ends, after
            its attention blocks (so it is evicted later than they are);
  rules M1: it goes back to the free queue when the prompt has been processed,
            so at the end it sits ahead of the request's attention blocks.

Block identity is per session: a cached block is valid for a later request of
the same session iff the tokens it covers lie inside the common prefix with the
request that produced it. Prefixes shared between sessions are not modelled, and
neither is preemption: a request that finds too few free blocks is marked failed.

The rule sets M0 and M1 are the two readings of the source that the measurement
has to tell apart; upstream issue #61281 describes M1.

Input: the event list of a schedule (events_from_schedule) or the records of
pressure_replay.py (events_from_trace). Output: the hit of every request and
the eviction counts. The command line replays one client record file and
compares the simulated hits with the measured ones.

Usage:
  python experiments/e5_cache_pressure/lru_sim.py --trace <client jsonl> --arch fa|hy
             [--unit 16] [--rules M0] --pool-blocks N [--json out.json]
"""
import argparse
import collections
import json

FA_BLOCK = 64
HY_BLOCK = 576


def cdiv(a, b):
    return -(-a // b)


def lcp(a, b):
    """Longest common prefix of two lists. Same result as hplab.cache_model.lcp,
    but on plain lists and without numpy, so the simulator stays standard library."""
    n, lo = min(len(a), len(b)), 0
    while lo < n:
        hi = min(lo + 4096, n)
        if a[lo:hi] != b[lo:hi]:
            while a[lo] == b[lo]:
                lo += 1
            return lo
        lo = hi
    return n


class Block:
    """A pool block. `hashed` blocks are in the prefix cache under `key`:
    (sid, "a", block index) or (sid, "s", checkpoint position). `prod` is the
    request (k) that filled it; an attention block is `full`, or holds the
    producer's checkpoint position in `part`, or both."""
    __slots__ = ("id", "hashed", "key", "prod", "full", "part", "ref")

    def __init__(self, i):
        self.id, self.ref = i, 0
        self.clear()

    def clear(self):
        self.hashed, self.key, self.prod, self.full, self.part = False, None, None, False, None


class Pool:
    def __init__(self, n):
        self.n = n
        self.free = collections.OrderedDict((i, Block(i)) for i in range(n))   # first item = front
        self.attn = collections.defaultdict(list)     # (sid, "a", index) -> cached blocks
        self.state = collections.defaultdict(list)    # sid -> cached state checkpoints
        self.evictions = self.evictions_state = self.peak = 0

    def _index(self, b):
        return self.state[b.key[0]] if b.key[1] == "s" else self.attn[b.key]

    def alloc(self, count):
        """Pop `count` blocks from the front, or None (and nothing popped) if too few are free."""
        if len(self.free) < count:
            return None
        out = []
        for _ in range(count):
            _, b = self.free.popitem(last=False)
            if b.hashed:
                self.evictions += 1
                self.evictions_state += b.key[1] == "s"
                self._index(b).remove(b)
                b.clear()
            b.ref = 1
            out.append(b)
        self.peak = max(self.peak, self.n - len(self.free))
        return out

    def touch(self, b):
        if b.ref == 0:
            del self.free[b.id]
        b.ref += 1
        self.peak = max(self.peak, self.n - len(self.free))

    def cache(self, b, key, prod, full=False, part=None):
        b.hashed, b.key, b.prod, b.full, b.part = True, key, prod, full, part
        self._index(b).append(b)

    def release(self, blocks):
        """Free in the given order: hashed blocks to the back, the others to the front."""
        for b in blocks:
            b.ref -= 1
            if b.ref == 0:
                self.free[b.id] = b
                if not b.hashed:
                    self.free.move_to_end(b.id, last=False)


class _Sim:
    def __init__(self, pool_blocks, arch, unit, rules):
        self.pool, self.arch, self.rules = Pool(pool_blocks), arch, rules
        self.B = FA_BLOCK if arch == "fa" else HY_BLOCK
        self.unit = self.B if arch == "fa" else (unit or HY_BLOCK)
        self.live, self.rows, self.failures = {}, [], 0

    # ---- lookup
    def _full_block(self, req, i):
        need = (i + 1) * self.B
        for b in self.pool.attn.get((req["sid"], "a", i), ()):
            if b.full and need <= req["lcp_prev"].get(b.prod, 0):
                return b
        return None

    def _chain(self, req, h):
        """Cached attention blocks covering tokens [0, h), or None."""
        chain = []
        for i in range(h // self.B):
            b = self._full_block(req, i)
            if b is None:
                return None
            chain.append(b)
        if h % self.B:      # the producer's partly filled tail block
            for b in self.pool.attn.get((req["sid"], "a", h // self.B), ()):
                if b.part == h and h <= req["lcp_prev"].get(b.prod, 0):
                    return chain + [b]
            return None
        return chain

    def _lookup(self, req):
        """(hit tokens, hit attention blocks, hit state block or None)."""
        if self.arch == "fa":
            chain = []
            for i in range((req["P"] - 1) // self.B):
                b = self._full_block(req, i)
                if b is None:
                    break
                chain.append(b)
            return len(chain) * self.B, chain, None
        for s in sorted(self.pool.state.get(req["sid"], ()), key=lambda b: -b.key[2]):
            h = s.key[2]
            if h <= min(req["lcp_prev"].get(s.prod, 0), req["P"] - 1):
                chain = self._chain(req, h)
                if chain is not None:
                    return h, chain, s
        return 0, [], None

    # ---- events
    def start(self, req):
        P, B = req["P"], self.B
        hit, chain, hit_state = self._lookup(req)
        ckpt_pos = (P - 1) // self.unit * self.unit if self.arch == "hy" else 0
        n_attn = cdiv(P, B) - hit // B        # a hit inside a block is copied into a new block
        n_state = 2 + (hit % B != 0) + (ckpt_pos % B != 0) if self.arch == "hy" else 0
        held = chain + ([hit_state] if hit_state else [])
        for b in held:
            self.pool.touch(b)
        st = {"req": req, "hit": hit, "chain": chain, "hit_state": hit_state, "attn": [], "state": [],
              "ckpt_pos": ckpt_pos, "ckpt": None,
              "row": {"sid": req["sid"], "k": req["k"], "hit": hit, "blocks_allocated": 0, "failed": False}}
        self.rows.append(st["row"])
        new = self.pool.alloc(n_attn + n_state)
        if new is None:
            return self._fail(st)
        st["attn"], st["state"] = new[:n_attn], new[n_attn:]
        st["row"]["blocks_allocated"] = len(new)
        self.live[(req["sid"], req["k"])] = st

    def _fail(self, st):
        st["row"]["failed"] = True
        self.failures += 1
        self.live.pop((st["req"]["sid"], st["req"]["k"]), None)
        self.pool.release(st["attn"] + st["state"])     # no hash: back to the front
        self.pool.release(reversed(st["chain"] + ([st["hit_state"]] if st["hit_state"] else [])))

    def _checkpoint(self, st):
        if st["ckpt_pos"] > 0:
            st["ckpt"] = st["state"].pop()
            self.pool.cache(st["ckpt"], (st["req"]["sid"], "s", st["ckpt_pos"]), st["req"]["k"])
            self.pool.release([st["ckpt"]])

    def prompt_done(self, req):
        st = self.live.get((req["sid"], req["k"]))
        if st is None:
            return
        if self.arch == "hy" and self.rules == "M1":
            self._checkpoint(st)                         # back in the free queue before decode
        n = cdiv(req["P"] + req["n_out"], self.B) - cdiv(req["P"], self.B)
        if n:
            new = self.pool.alloc(n)
            if new is None:
                return self._fail(st)
            st["attn"] += new
            st["row"]["blocks_allocated"] += n

    def end(self, req):
        st = self.live.pop((req["sid"], req["k"]), None)
        if st is None:
            return
        B, total, c = self.B, req["P"] + req["n_out"], st["ckpt_pos"]
        for i, b in enumerate(st["attn"], start=st["hit"] // B):
            full = (i + 1) * B <= total
            part = c if c % B and c // B == i else None
            if full or part is not None:
                self.pool.cache(b, (req["sid"], "a", i), req["k"], full, part)
        self.pool.release(reversed(st["chain"] + st["attn"]))     # tail block first
        if self.arch == "hy":
            if self.rules == "M0":
                self._checkpoint(st)                     # after the attention blocks
            self.pool.release(st["state"])
            if st["hit_state"]:
                self.pool.release([st["hit_state"]])


def simulate(events, pool_blocks, arch, unit=None, rules="M0"):
    """events: time-ordered ("start" | "prompt_done" | "end", req), with
    req = {"sid", "k", "P", "n_out", "lcp_prev": {earlier k: common prefix length}}.
    Prompt blocks and state blocks are allocated at "start", the blocks for the
    output at "prompt_done". Returns (rows, summary): one row per request in
    start order, {"sid", "k", "hit", "blocks_allocated", "failed"}."""
    assert arch in ("fa", "hy") and rules in ("M0", "M1"), (arch, rules)
    assert arch == "fa" or unit in (None, 576, 16), unit
    sim = _Sim(pool_blocks, arch, unit, rules)
    for name, req in events:
        getattr(sim, name)(req)
    return sim.rows, {"evictions": sim.pool.evictions, "evictions_state": sim.pool.evictions_state,
                      "alloc_failures": sim.failures, "peak_in_use": sim.pool.peak}


# ------------------------------------------------------------------ event lists
def make_requests(sessions, n_out):
    """{(sid, k): req} for sessions of the form {"sid", "prompts": [token id lists]}."""
    out = {}
    for s in sessions:
        ps = s["prompts"]
        for k, p in enumerate(ps):
            out[(s["sid"], k)] = {"sid": s["sid"], "k": k, "P": len(p), "n_out": n_out,
                                  "lcp_prev": {j: lcp(p, ps[j]) for j in range(k)}}
    return out


def events_from_schedule(sessions, schedule, n_out, overlap=1):
    """schedule: [(sid, k)] in dispatch order. overlap = 1 gives a serial run
    (start, prompt_done, end per request). With overlap > 1 up to that many
    requests are in flight: a request's prompt is processed right after it
    starts, and the oldest request in flight ends only when the next one needs
    its place or belongs to the same session."""
    reqs = make_requests(sessions, n_out)
    events, flying = [], []
    for sid, k in schedule:
        while flying and (len(flying) >= overlap or any(r["sid"] == sid for r in flying)):
            events.append(("end", flying.pop(0)))
        r = reqs[(sid, k)]
        events += [("start", r), ("prompt_done", r)]
        flying.append(r)
    return events + [("end", r) for r in flying]


def events_from_trace(rows):
    """Events from the records of pressure_replay.py: start at t_send_ns, prompt_done at
    t_send + TTFT, end at t_end_ns. Returns (events, assumed): `assumed` is True
    when a record carries no lcp_prev and every prompt was taken to extend the
    earlier prompts of its session."""
    rows = sorted((r for r in rows if r.get("sid") != "control" and not r.get("error")
                   and r.get("t_send_ns") is not None), key=lambda r: (r["sid"], r["k"]))
    timed, seen, assumed = [], collections.defaultdict(dict), False
    for n, r in enumerate(rows):
        prev = r.get("lcp_prev")
        if prev is None:
            assumed = True
            prev = {j: min(p, r["P"]) for j, p in seen[r["sid"]].items()}
        seen[r["sid"]][r["k"]] = r["P"]
        req = {"sid": r["sid"], "k": r["k"], "P": r["P"], "n_out": r["n_out"],
               "lcp_prev": {int(j): v for j, v in prev.items()}}
        t0, t1 = r["t_send_ns"], r["t_end_ns"]
        tp = t1 if r.get("ttft_ms") is None else min(t1, t0 + int(r["ttft_ms"] * 1e6))
        timed += [(t0, 0, n, "start", req), (tp, 1, n, "prompt_done", req), (t1, 2, n, "end", req)]
    timed.sort(key=lambda e: e[:3])
    return [(name, req) for _, _, _, name, req in timed], assumed


def min_pool_without_loss(sessions, schedule, arch, unit, rules, n_out, lo, hi, overlap=1):
    """Smallest pool in [lo, hi] (blocks) for which no request fails and every hit
    equals the hit with a pool too large to evict anything. Binary search, so it
    takes "no loss" to be monotone in the pool size; the result p always has no
    loss while p - 1 has (or p == lo). None if even `hi` loses something."""
    events = events_from_schedule(sessions, schedule, n_out, overlap)
    B = FA_BLOCK if arch == "fa" else HY_BLOCK
    big = sum(cdiv(r["P"] + r["n_out"], B) + 5 for name, r in events if name == "start")
    hits = lambda rows: [(r["sid"], r["k"], r["hit"]) for r in rows]      # noqa: E731
    want = hits(simulate(events, big, arch, unit, rules)[0])

    def lossless(pool):
        rows, summary = simulate(events, pool, arch, unit, rules)
        return summary["alloc_failures"] == 0 and hits(rows) == want

    if not lossless(hi):
        return None
    if lossless(lo):
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if lossless(mid) else (mid, hi)
    return hi


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="jsonl written by pressure_replay.py")
    ap.add_argument("--arch", required=True, choices=["fa", "hy"])
    ap.add_argument("--unit", type=int, default=576, choices=[576, 16])
    ap.add_argument("--rules", default="M0", choices=["M0", "M1"])
    ap.add_argument("--pool-blocks", type=int, required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    with open(a.trace, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    events, assumed = events_from_trace(rows)
    sim_rows, summary = simulate(events, a.pool_blocks, a.arch, a.unit, a.rules)
    measured = {(r["sid"], r["k"]): r for r in rows if r.get("sid") != "control"}
    out = []
    print(f"{'sid':24s} {'k':>3s} {'P':>6s} {'measured':>8s} {'sim':>6s}")
    for s in sim_rows:
        m = measured[(s["sid"], s["k"])]
        out.append({**s, "P": m["P"], "cached_tokens": m["cached_tokens"]})
        print(f"{s['sid']:24s} {s['k']:3d} {m['P']:6d} {m['cached_tokens']:8d} {s['hit']:6d}"
              + ("  FAILED" if s["failed"] else "") + ("" if s["hit"] == m["cached_tokens"] else "  *"))
    summary.update({"n": len(out), "n_equal": sum(1 for r in out if r["hit"] == r["cached_tokens"]),
                    "arch": a.arch, "unit": a.unit if a.arch == "hy" else None, "rules": a.rules,
                    "pool_blocks": a.pool_blocks, "lcp_assumed_from_prompt_lengths": assumed})
    print(json.dumps(summary))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline=chr(10)) as f:
            json.dump({"summary": summary, "rows": out}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
