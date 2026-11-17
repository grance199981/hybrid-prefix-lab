"""Count the prefill steps of each replayed request from the server's iteration log,
and compare them with the step counts that the scheduler rule gives
(hplab.cache_model.count_steps).

The server runs with --enable-logging-iteration-details and logs one line per engine iteration:
    Engine 000: Iteration(12): 1 context requests, 256 context tokens, 0 generation requests, ...
The replay sends one request at a time, so the iterations with context tokens belong to the requests in
the order in which they were sent.  A request that finishes inside its last prefill iteration leaves no
generation-only iteration behind it, so runs of prefill iterations cannot simply be split at the gaps.
Instead each request takes iterations until their context tokens add up to the tokens it had to
compute (prompt tokens minus cached tokens).  If the sum overshoots, the match has failed and the
tool stops matching that run: nothing is guessed.

Input: the server log and the replay records of that server, in the order they ran.
Output: the share of requests whose measured step count equals the rule's, the
(simulated -> measured) pairs, and with --json one row per request.

Usage:
  python experiments/e7_upstream_prs/steps_from_log.py --log serve_C1a.log
      --records C1a_wildchat_thinkoff_R_s1_seed4.jsonl [more .jsonl in the order they ran]
      --unit 64 [--alignment 64] [--rule 66938] [--json out.json]
  (--unit 0 for the default configuration)
"""
import argparse
import json
import os
import re
from collections import Counter

from hplab.cache_model import count_steps

LINE = re.compile(r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, (\d+) generation requests")


def parse_log(lines):
    """[(iteration index, context requests, context tokens)] for the iterations that prefill."""
    out = []
    for line in lines:
        m = LINE.search(line)
        if m and int(m.group(3)) > 0:
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def load_records(paths):
    """Requests in the order of the files and of the lines in each file; failed requests are dropped."""
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("error") or r.get("prompt_tokens") is None:
                    continue
                out.append({"file": os.path.basename(p), "sid": r.get("sid"), "k": r.get("k"),
                            "control": r.get("control"), "prompt": r["prompt_tokens"],
                            "cached": r.get("cached_tokens") or 0})
    return out


def match(iterations, requests):
    """Give each request the iterations whose context tokens add up to prompt - cached.
    Returns (matched requests with 'steps' and 'chunks', error or None)."""
    i, done = 0, []
    for r in requests:
        need = r["prompt"] - r["cached"]
        got, chunks = 0, []
        while got < need and i < len(iterations):
            idx, nreq, tok = iterations[i]
            if nreq != 1:
                return done, f"iteration {idx} has {nreq} context requests; the replay is not one request at a time"
            got += tok
            chunks.append(tok)
            i += 1
        if got != need:
            return done, (f"request {len(done)} ({r['file']} sid={r['sid']} k={r['k']}) needs {need} tokens, "
                          f"the log gives {got} in {chunks}")
        done.append(dict(r, steps=len(chunks), chunks=chunks))
    left = len(iterations) - i
    return done, (f"{left} prefill iterations are left over after the last request" if left else None)


def compare(matched, unit, alignment, rule):
    rows = []
    for r in matched:
        sim = count_steps(r["cached"], r["prompt"], unit, alignment, rule=rule)
        rows.append(dict(r, sim=sim))
    same = sum(r["steps"] == r["sim"] for r in rows)
    pairs = Counter((r["sim"], r["steps"]) for r in rows)
    return rows, {
        "n": len(rows), "same": same, "share_same": round(same / len(rows), 4) if rows else None,
        "simulated_vs_measured": {f"{a}->{b}": n for (a, b), n in sorted(pairs.items())},
        "measured_steps": dict(sorted(Counter(r["steps"] for r in rows).items())),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--unit", type=int, required=True, help="0 for the default configuration")
    ap.add_argument("--alignment", type=int, default=None)
    ap.add_argument("--rule", default="66938", choices=("1479deb0", "66938"))
    ap.add_argument("--json")
    a = ap.parse_args()
    with open(a.log, encoding="utf-8", errors="replace") as fh:
        iterations = parse_log(fh)
    requests = load_records(a.records)
    matched, error = match(iterations, requests)
    rows, res = compare(matched, a.unit or None, a.alignment, a.rule)
    res.update({"requests": len(requests), "prefill_iterations": len(iterations), "matched": len(matched),
                "match_error": error, "unit": a.unit, "alignment": a.alignment, "rule": a.rule})
    print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False, indent=1))
    for r in [r for r in rows if r["steps"] != r["sim"]][:12]:
        print(f"  differs: {r['file']} sid={r['sid']} k={r['k']} cached={r['cached']} prompt={r['prompt']} "
              f"simulated {r['sim']} measured {r['steps']} chunks {r['chunks']}")
    if a.json:
        res["rows"] = [{k: r[k] for k in ("file", "sid", "k", "cached", "prompt", "steps", "sim", "chunks")} for r in rows]
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
