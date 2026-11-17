"""Per-cell table of the cache-pressure runs, from the records of pressure_replay.py.

A cell is (stage, workload, config, pool MiB, output length); its seeds are pooled.
Per cell: retention (cached tokens / infinite-cache prediction over the requests
with k > 0, control requests excluded), the shares of requests that hit exactly
the prediction ("full"), nothing ("zero"), more ("above") or something in between
("other"), recompute median and p90, median TTFT, preemptions, peak cache usage,
request errors and wall time. These are the numbers the README quotes.

Output: one line per cell on stdout; with --json a list of cell rows.

Usage: python experiments/e5_cache_pressure/pressure_table.py
           <dir with the *.jsonl / *.summary.json / *.metrics.jsonl> [--stage S] [--json out.json]
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics as st

TAG = re.compile(r"^(?P<cfg>[A-Z]\d)_(?P<stage>[A-Z])_(?P<wl>[a-z]+)_kv(?P<kv>\d+)_out(?P<out>\d+)_seed(?P<seed>\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cells = collections.defaultdict(lambda: {"rows": [], "preempt": 0, "usage": 0.0, "seeds": 0, "errors": 0, "wall": 0.0})
    for f in sorted(glob.glob(os.path.join(a.dir, "*.summary.json"))):
        tag = os.path.basename(f)[:-len(".summary.json")]
        m = TAG.match(tag)
        if not m or (a.stage and m["stage"] != a.stage):
            continue
        c = cells[(m["stage"], m["wl"], m["cfg"], int(m["kv"]), int(m["out"]))]
        s = json.load(open(f, encoding="utf-8"))
        c["preempt"] += int(s["delta"].get("vllm:num_preemptions", 0))
        c["errors"] += s["n_errors"]
        c["wall"] += s["wall_seconds"]
        c["seeds"] += 1
        with open(os.path.join(a.dir, tag + ".jsonl"), encoding="utf-8") as fh:
            c["rows"] += [r for r in map(json.loads, fh) if r.get("sid") != "control" and r["k"] > 0 and not r.get("error")]
        mp = os.path.join(a.dir, tag + ".metrics.jsonl")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as fh:
                c["usage"] = max([c["usage"]] + [json.loads(l).get("vllm:kv_cache_usage_perc", 0.0) for l in fh])
    out = []
    for key in sorted(cells):
        c = cells[key]
        rows = c["rows"]
        n = len(rows)
        pred = sum(r["pred_inf"] for r in rows)
        cached = sum(r["cached_tokens"] for r in rows)
        full = sum(1 for r in rows if r["cached_tokens"] == r["pred_inf"])
        zero = sum(1 for r in rows if r["cached_tokens"] == 0)
        above = sum(1 for r in rows if r["cached_tokens"] > r["pred_inf"])
        ttft = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
        row = {"stage": key[0], "workload": key[1], "cfg": key[2], "kv_mib": key[3], "out": key[4], "seeds": c["seeds"],
               "n": n, "retention": round(cached / pred, 4) if pred else None,
               "full": round(full / n, 3), "zero": round(zero / n, 3), "above": round(above / n, 3),
               "other": round((n - full - zero - above) / n, 3),
               "recompute_median": st.median(r["P"] - r["cached_tokens"] for r in rows),
               "recompute_p90": sorted(r["P"] - r["cached_tokens"] for r in rows)[int(0.9 * (n - 1))],
               "ttft_median_ms": round(st.median(ttft), 1) if ttft else None,
               "preemptions": c["preempt"], "usage_max": round(c["usage"], 3), "errors": c["errors"],
               "wall_s": round(c["wall"])}
        out.append(row)
        print(" ".join(f"{k}={v}" for k, v in row.items() if k not in ("stage", "workload", "seeds")))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
