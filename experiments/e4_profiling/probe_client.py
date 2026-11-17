"""Request patterns for the profiling runs. Every request is a /v1/completions call
whose prompt is a list of token ids, so the server sees exactly P prompt tokens and
no chat template is involved. The client pauses between requests (or groups of
concurrent requests), so each one is a separate burst of GPU work on a profiler
timeline. Started by scripts/run_probe.sh, which passes --url, --model,
--tokenizer-dir, --lengths, --out-tokens, --reps, --warmup, --label and --out.

Patterns (--pattern):
  cold    (Q1, and the cold-prompt probes of E7 and E8) one request at a time with
          P prompt tokens and --out-tokens output tokens; the prefix cache is reset
          before every request, so nothing is ever hit. One JSON line per request.
  groups  (Q2, Q3, E9) first one cold prefill-only request per --lengths value
          (--out-tokens output tokens), then for every --batches value B a group of
          B requests sent together, each a 16-token prompt and --decode-tokens
          output tokens, so the GPU spends the group in decode steps of batch size
          about B. Cache reset before every group. One JSON line per group.
  long    (Q2b) decode at a long context C without the prefill of B * C tokens in
          the same burst: each group is sent twice, first "warm" (cache reset, the B
          prompts of C tokens, 1 output token each) and then "decode" (no reset, the
          same B prompts, --decode-tokens output tokens). With C = 576 * k + 1 both
          models find the first C - 1 tokens in the cache (576 is the Qwen3.5 block
          and a multiple of the 64-token full-attention block), so the burst is
          decode steps at a context of C .. C + decode_tokens. Contexts too short to
          hit the cache (C - 1 < 64) are sent cold. --lengths gives optional
          prefill-only groups.

Exit code 1 when a measured request failed, had a wrong prompt length or an
unexpected cache hit.

Usage:
  probe_client.py --pattern cold --url http://127.0.0.1:8061 --model qwen35
      --tokenizer-dir MODEL_DIR --lengths 16 560 2000 --reps 5 --label h4_C1_nsys --out out.jsonl
"""
import argparse
import json
import threading
import time
import urllib.parse

from hplab.client import completion_body, post_json, stream_completion
from hplab.prompts import token_ids

DECODE_PROMPT = 16     # prompt length of a decode request in the `groups` pattern
MIN_HIT = 64           # smallest prefix either model can hit (`long` pattern)
DEFAULTS = {"cold": {"reps": 5, "out_tokens": 4}, "groups": {"reps": 3, "out_tokens": 1},
            "long": {"reps": 3, "out_tokens": 1}}


def long_plan(lengths, contexts, batches):
    """[(phase, tokens, batch, reset_before, expected_cached_tokens)] of one repetition."""
    plan = [("prefill", P, 1, True, 0) for P in lengths]
    for C in contexts:
        for B in batches:
            if C - 1 < MIN_HIT:
                plan.append(("decode", C, B, True, 0))
            else:
                plan.append(("warm", C, B, True, 0))
                plan.append(("decode", C, B, False, C - 1))
    return plan


class Client:
    def __init__(self, a, tok):
        self.a, self.tok = a, tok
        u = urllib.parse.urlparse(a.url)
        self.host, self.port = u.hostname, u.port

    def reset(self):
        st, data = post_json(self.host, self.port, "/reset_prefix_cache")
        if st != 200:
            raise RuntimeError(f"cache reset failed: {st} {data[:200]!r}")
        time.sleep(self.a.gap)

    def body(self, seed, length, n_out):
        return completion_body(self.a.model, token_ids(self.tok, seed, length), n_out)

    def send_together(self, bodies):
        """Send the requests at once, one thread each. Returns (t_send_ns, t_end_ns, results)."""
        results = [None] * len(bodies)

        def run(i):
            try:
                results[i] = stream_completion(self.host, self.port, bodies[i])
                results[i]["error"] = None
            except Exception as e:  # noqa: BLE001 - recorded
                results[i] = {"error": repr(e)[:300]}

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(bodies))]
        t0 = time.time_ns()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return t0, time.time_ns(), results

    # ------------------------------------------------------------------ cold
    def cold(self, length, rep, kind):
        a = self.a
        self.reset()
        body = self.body(f"{a.label}-{kind}-{length}-{rep}", length, a.out_tokens)
        rec = {"label": a.label, "kind": kind, "P": length, "rep": rep}
        try:
            rec.update(stream_completion(self.host, self.port, body))
            rec["error"] = None
        except Exception as e:  # noqa: BLE001 - recorded
            rec["error"] = repr(e)[:300]
        time.sleep(a.gap)
        return rec

    # ---------------------------------------------------------------- groups
    @staticmethod
    def group_record(a, kind, phase, P, B, rep, t0, t1, results, extra):
        ok = [r for r in results if not r["error"]]
        rec = {"label": a.label, "kind": kind, "phase": phase, "P": P, "B": B, "rep": rep,
               "t_send_ns": t0, "t_end_ns": t1, "wall_ms": (t1 - t0) / 1e6,
               "n_errors": B - len(ok),
               "n_wrong_length": sum(1 for r in ok if r["prompt_tokens"] != P)}
        rec.update(extra(ok))
        rec.update({
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in ok),
            "ttft_ms": [round(r["ttft_ms"], 2) if r["ttft_ms"] is not None else None for r in ok],
            "e2e_ms": [round(r["e2e_ms"], 2) for r in ok],
            "error": None if len(ok) == B else results[[bool(r["error"]) for r in results].index(True)]["error"]})
        return rec

    def group(self, phase, P, B, rep, kind):
        a = self.a
        self.reset()
        n_out = a.out_tokens if phase == "prefill" else a.decode_tokens
        bodies = [self.body(f"{a.label}-{kind}-{phase}-{P}-{B}-{rep}-{i}", P, n_out) for i in range(B)]
        t0, t1, results = self.send_together(bodies)
        rec = self.group_record(a, kind, phase, P, B, rep, t0, t1, results,
                                lambda ok: {"n_cache_hits": sum(1 for r in ok if r["cached_tokens"])})
        time.sleep(a.gap)
        return rec

    # ------------------------------------------------------------------ long
    def long_group(self, phase, P, B, reset, expected, rep, kind):
        a = self.a
        if reset:
            self.reset()
        n_out = {"prefill": a.out_tokens, "warm": 1}.get(phase, a.decode_tokens)
        # the seed does not contain the phase: warm and decode send the same prompts
        bodies = [self.body(f"{a.label}-{kind}-{P}-{B}-{rep}-{i}", P, n_out) for i in range(B)]
        t0, t1, results = self.send_together(bodies)

        def cached(ok):
            c = [r["cached_tokens"] for r in ok]
            return {"expected_cached": expected, "n_wrong_cached": sum(1 for x in c if x != expected),
                    "cached_min": min(c) if c else None, "cached_max": max(c) if c else None}

        rec = self.group_record(a, kind, phase, P, B, rep, t0, t1, results, cached)
        time.sleep(a.gap)
        return rec


def run_cold(cl, a, f):
    n_bad = 0
    for rep in range(a.warmup):
        for length in a.lengths:
            f.write(json.dumps(cl.cold(length, rep, "warmup")) + "\n")
    for rep in range(a.reps):
        for length in a.lengths:
            r = cl.cold(length, rep, "measure")
            f.write(json.dumps(r) + "\n")
            f.flush()
            if r["error"] or r["prompt_tokens"] != length or r["cached_tokens"] != 0:
                n_bad += 1
    print(f"{a.label}: {a.reps * len(a.lengths)} requests, {n_bad} with an error, a wrong "
          f"prompt length or a cache hit", flush=True)
    return n_bad


def run_groups(cl, a, f):
    plan = [("prefill", P, 1) for P in a.lengths] + [("decode", DECODE_PROMPT, B) for B in a.batches]
    n_bad = 0
    for kind, reps in (("warmup", a.warmup), ("measure", a.reps)):
        for rep in range(reps):
            for phase, P, B in plan:
                r = cl.group(phase, P, B, rep, kind)
                f.write(json.dumps(r) + "\n")
                f.flush()
                if kind == "measure" and (r["n_errors"] or r["n_wrong_length"]):
                    n_bad += 1
    print(f"{a.label}: {a.reps * len(plan)} groups, {n_bad} with an error or a wrong prompt length",
          flush=True)
    return n_bad


def run_long(cl, a, f):
    plan = long_plan(a.lengths, a.contexts, a.batches)
    n_bad = 0
    for kind, reps in (("warmup", a.warmup), ("measure", a.reps)):
        for rep in range(reps):
            for phase, P, B, reset, expected in plan:
                r = cl.long_group(phase, P, B, reset, expected, rep, kind)
                f.write(json.dumps(r) + "\n")
                f.flush()
                if kind == "measure" and (r["n_errors"] or r["n_wrong_length"] or r["n_wrong_cached"]):
                    n_bad += 1
    print(f"{a.label}: {a.reps * len(plan)} groups, {n_bad} with an error, a wrong prompt length "
          f"or an unexpected cache hit", flush=True)
    return n_bad


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=sorted(DEFAULTS), default="cold")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--lengths", type=int, nargs="*", default=[],
                    help="prompt lengths (cold); prefill-only lengths (groups, long)")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64], help="groups, long")
    ap.add_argument("--contexts", type=int, nargs="+", default=[16, 577, 1153, 2305], help="long")
    ap.add_argument("--decode-tokens", type=int, default=128, help="groups, long")
    ap.add_argument("--reps", type=int, default=None, help="default 5 (cold) or 3")
    ap.add_argument("--out-tokens", type=int, default=None,
                    help="output tokens of a cold or prefill request; default 4 (cold) or 1")
    ap.add_argument("--gap", type=float, default=0.8, help="pause after each reset and request, s")
    ap.add_argument("--label", required=True)
    ap.add_argument("--warmup", type=int, default=0, help="warm-up repetitions first, written with kind 'warmup'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.pattern != "long" and not a.lengths:
        ap.error(f"--lengths is required with --pattern {a.pattern}")
    for k, v in DEFAULTS[a.pattern].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def main(argv=None):
    a = parse_args(argv)
    from transformers import AutoTokenizer
    cl = Client(a, AutoTokenizer.from_pretrained(a.tokenizer_dir))
    run = {"cold": run_cold, "groups": run_groups, "long": run_long}[a.pattern]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        n_bad = run(cl, a, f)
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
