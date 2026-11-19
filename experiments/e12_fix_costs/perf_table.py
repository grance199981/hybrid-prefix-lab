"""E12: performance and hit summary per cell of E5-style runs (pressure_replay.py output).

For every file group <CFG>_<STAGE>_<WORKLOAD>_kv<MiB>_out<N>_seed<S> in one or more e5/ directories,
seeds pooled per cell (cfg, workload, pool, output length):
  n               scored requests (no control, no error)
  ttft_p50/p90    time to first token, ms (all scored requests: turn 0 included, so a single-turn
                  workload is scored too)
  tpot_p50        time per output token after the first, ms: (e2e - ttft) / (n_out - 1); the output
                  length is fixed by min_tokens = max_tokens
  req_per_s       requests / wall seconds, wall seconds summed over seeds
  out_tok_per_s   completion tokens / wall seconds
  preemptions     sum of the /metrics delta
  usage_max       largest sampled vllm:kv_cache_usage_perc
  retention       sum cached / sum infinite-cache prediction over turns k > 0 (None without k > 0)
  cached_share    sum cached / sum prompt tokens over all scored requests
Usage: perf_table.py DIR [DIR ...] [--json out.json]
Standard library only.
"""
import argparse
import collections
import glob
import json
import os
import re

LABEL = re.compile(r"(?P<cfg>[A-Z]\d)_(?P<stage>[A-Z])_(?P<wl>[a-z]+)_kv(?P<mib>\d+)_out(?P<out>\d+)_seed(?P<seed>\d+)$")


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (i - lo), 2)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect(dirs):
    cells = collections.defaultdict(lambda: {"rows": [], "wall": 0.0, "preempt": 0.0, "usage": 0.0, "seeds": 0,
                                             "errors": 0})
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.summary.json"))):
            stem = os.path.basename(f)[:-len(".summary.json")]
            m = LABEL.search(stem)
            if not m:
                continue
            key = (m["cfg"], m["wl"], int(m["mib"]), int(m["out"]))
            c = cells[key]
            s = json.load(open(f, encoding="utf-8"))
            c["wall"] += s.get("wall_seconds") or 0.0
            c["preempt"] += (s.get("delta") or {}).get("vllm:num_preemptions", 0.0)
            c["errors"] += s.get("n_errors", 0)
            c["seeds"] += 1
            rows = os.path.join(d, stem + ".jsonl")
            if os.path.exists(rows):
                c["rows"] += [r for r in read_jsonl(rows) if r.get("sid") != "control" and not r.get("error")
                              and r.get("ttft_ms") is not None]
            met = os.path.join(d, stem + ".metrics.jsonl")
            if os.path.exists(met):
                c["usage"] = max([c["usage"]] + [r.get("vllm:kv_cache_usage_perc", 0.0) for r in read_jsonl(met)])
    return cells


def summarize(cells):
    out = []
    for (cfg, wl, mib, n_out), c in sorted(cells.items()):
        rows = c["rows"]
        ttft = [r["ttft_ms"] for r in rows]
        tpot = [(r["e2e_ms"] - r["ttft_ms"]) / (n_out - 1) for r in rows
                if n_out > 1 and r.get("e2e_ms") is not None]
        later = [r for r in rows if r.get("k", 0) > 0]
        pred = sum(r.get("pred_inf") or 0 for r in later)
        comp = sum(r.get("completion_tokens") or 0 for r in rows)
        out.append({
            "cfg": cfg, "workload": wl, "kv_mib": mib, "out": n_out, "seeds": c["seeds"], "n": len(rows),
            "errors": c["errors"],
            "ttft_p50": pct(ttft, 0.5), "ttft_p90": pct(ttft, 0.9),
            "ttft_later_p50": pct([r["ttft_ms"] for r in later], 0.5),
            "tpot_p50": pct(tpot, 0.5),
            "req_per_s": round(len(rows) / c["wall"], 4) if c["wall"] else None,
            "out_tok_per_s": round(comp / c["wall"], 2) if c["wall"] else None,
            "wall_s": round(c["wall"], 1), "preemptions": int(c["preempt"]), "usage_max": round(c["usage"], 3),
            "retention": round(sum(r["cached_tokens"] for r in later) / pred, 4) if pred else None,
            "cached_share": round(sum(r["cached_tokens"] for r in rows) / max(1, sum(r.get("prompt_tokens") or 0
                                                                              for r in rows)), 4),
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = summarize(collect(a.dirs))
    for r in res:
        print(" ".join(f"{k}={v}" for k, v in r.items()))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
