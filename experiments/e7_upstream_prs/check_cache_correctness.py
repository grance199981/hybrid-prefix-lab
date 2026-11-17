"""Correctness control for a patched server: the model's output must be the same,
up to numerical noise, whether the prompt is computed from scratch or continued
from a cached state.

For each prompt length P and each cut c (how many tokens of the prompt are sent
first to be cached):
  cold:  reset the prefix cache; send the P-token prompt.
  warm:  reset; send the first c tokens (this leaves a checkpoint); send the
         P-token prompt again; it must report cached tokens > 0.
Compared: the log-probabilities of the first generated token (the five most
likely tokens of the cold run, looked up in the warm run). `max_dlogprob` is
the largest absolute difference. Whether the greedy text of 48 tokens is
identical is recorded but not judged by --tol.

The default prompt counts upwards ("1000 1001 1002 ..."), so the next token is
nearly certain and a wrong state shows. With random words (--prompt random) the
next-token distribution is flat: the top five change with tiny numerical
differences and the greedy text can differ even on an unpatched server, because
a different split of the prefill changes the rounding. That prompt only shows
the noise level.

Known good: the unpatched server; run it first without --tol to measure the
noise level, then judge the patched server with --tol set from that.
Known bad:  --self-test shifts one warm log-probability by 5 and must fail.

Output: one JSON line per comparison on stdout; with --json the list of rows.
scripts/run_probe.sh runs it on every server it starts when PROBE_CHECK points
to this file.

Usage: python experiments/e7_upstream_prs/check_cache_correctness.py --url http://127.0.0.1:8061
          --model qwen35 --tokenizer-dir <dir> [--lengths 2000 5000 9000] [--tol 0.3]
          [--json out.json] [--self-test]
Exit codes: 0 ok (or no --tol given), 1 a difference above --tol or a warm request
            without a cache hit, 6 server unreachable.
"""
import argparse
import json
import sys
import urllib.parse

from hplab.client import post_json
from hplab.prompts import token_ids


def complete(host, port, model, ids, n_out, logprobs=None):
    body = {"model": model, "prompt": ids, "max_tokens": n_out, "min_tokens": n_out, "temperature": 0.0,
            "seed": 0, "stream": False}
    if logprobs:
        body["logprobs"] = logprobs
    status, data = post_json(host, port, "/v1/completions", body, timeout=600)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {data[:300]!r}")
    d = json.loads(data)
    details = (d.get("usage") or {}).get("prompt_tokens_details") or {}
    lp = (d["choices"][0].get("logprobs") or {}).get("top_logprobs") or [{}]
    return d["choices"][0]["text"], int(details.get("cached_tokens") or 0), dict(lp[0] or {})


def prompt_ids(tok, kind, P):
    """counting: "1000 1001 1002 ..." cut to P tokens, so the next token is nearly certain and a wrong
    state shows; random: unrelated words (flat next-token distribution: only useful to see the noise)."""
    if kind == "random":
        return token_ids(tok, f"corr-{P}", P)
    text = " ".join(str(i) for i in range(1000, 1000 + P))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= P, (len(ids), P)
    return [int(x) for x in ids[:P]]


def reset(host, port):
    status, data = post_json(host, port, "/reset_prefix_cache")
    if status != 200:
        raise RuntimeError(f"reset_prefix_cache HTTP {status}: {data[:200]!r}")


def max_diff(cold, warm):
    """Largest |difference| over the cold run's tokens; a token missing from the warm top list counts as 99."""
    if not cold:
        return 99.0
    return max(abs(v - warm[t]) if t in warm else 99.0 for t, v in cold.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[2000, 5000, 9000])
    ap.add_argument("--out-tokens", type=int, default=48)
    ap.add_argument("--prompt", default="counting", choices=["counting", "random"])
    ap.add_argument("--tol", type=float, default=None, help="largest allowed max_dlogprob; omit to only measure")
    ap.add_argument("--json", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    u = urllib.parse.urlparse(a.url)
    host, port = u.hostname, u.port
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    rows, bad = [], 0
    try:
        for P in a.lengths:
            ids = prompt_ids(tok, a.prompt, P)
            reset(host, port)
            cold_text, cached_cold, cold_lp = complete(host, port, a.model, ids, a.out_tokens, logprobs=5)
            for cut in sorted({P - 200, P - 77, P // 2}):   # short, odd and long recomputed tails
                reset(host, port)
                complete(host, port, a.model, ids[:cut], 1)
                warm_text, cached, warm_lp = complete(host, port, a.model, ids, a.out_tokens, logprobs=5)
                if a.self_test and cut == P // 2 and warm_lp:
                    k = next(iter(warm_lp))
                    warm_lp[k] += 5.0
                d = max_diff(cold_lp, warm_lp)
                top_same = bool(cold_lp) and bool(warm_lp) and max(cold_lp, key=cold_lp.get) == max(warm_lp, key=warm_lp.get)
                ok = cached > 0 and (a.tol is None or d <= a.tol)
                bad += 0 if ok else 1
                top_cold = max(cold_lp, key=cold_lp.get) if cold_lp else None
                rows.append({"P": P, "cut": cut, "prompt": a.prompt,
                             "top1_logprob_cold": round(cold_lp[top_cold], 5) if top_cold else None,
                             "top1_logprob_warm": round(warm_lp.get(top_cold, float("nan")), 5) if top_cold else None,
                             "cached_cold": cached_cold, "cached_warm": cached,
                             "max_dlogprob": round(d, 5), "top1_same": top_same, "text_equal": warm_text == cold_text,
                             "n_top": len(cold_lp), "ok": ok})
                print(json.dumps(rows[-1]), flush=True)
    except OSError as e:
        print(f"server unreachable: {e!r}")
        return 6
    worst = max((r["max_dlogprob"] for r in rows), default=None)
    print(f"{len(rows)} comparisons, {bad} not ok, largest max_dlogprob {worst}, tol {a.tol}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rows, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
