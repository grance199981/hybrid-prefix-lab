"""Median TTFT per prompt length of the cold-prompt probe, per configuration, and
the differences to the default configuration.

Input: a run directory of scripts/run_probe.sh in plain mode (no profiler) with
the configurations h4_C0 (default), h4_C1 (--prefix-match-unit 16) and h4_C64
(--prefix-match-unit 64): h4_C0_plain.jsonl, h4_C1_plain.jsonl, h4_C64_plain.jsonl.
Only the measured requests (kind "measure") without an error are used.
Output: the table on stdout (with a check that no request hit the cache), and
with --json {config: {length: {n, median, min, max, cached_max}}}.

Usage: python experiments/e7_upstream_prs/probe_table.py <run dir> [--json out.json]
"""
import argparse
import json
import os
import statistics as st


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") != "measure" or r.get("error"):
                continue
            out.setdefault(r["P"], []).append(r)
    return out


def table(directory, configs=("h4_C0", "h4_C1", "h4_C64")):
    res = {}
    for c in configs:
        p = os.path.join(directory, f"{c}_plain.jsonl")
        if not os.path.exists(p):
            continue
        res[c] = {}
        for length, rows in sorted(load(p).items()):
            t = [r["ttft_ms"] for r in rows]
            res[c][length] = {"n": len(t), "median": round(st.median(t), 1), "min": round(min(t), 1),
                              "max": round(max(t), 1),
                              "cached_max": max(int(r.get("cached_tokens") or 0) for r in rows)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = table(a.directory)
    lengths = sorted(res.get("h4_C0", {}))
    print("length  " + "  ".join(f"{c:>22s}" for c in res))
    for length in lengths:
        cells = []
        for c in res:
            x = res[c].get(length)
            cells.append(f"{x['median']:7.1f} [{x['min']:.0f}-{x['max']:.0f}] n={x['n']}" if x else "-")
        print(f"{length:6d}  " + "  ".join(f"{s:>22s}" for s in cells))
    print("minus the default configuration (medians)")
    for c in res:
        if c == "h4_C0":
            continue
        print(f"  {c}: " + "  ".join(f"{length}: {res[c][length]['median'] - res['h4_C0'][length]['median']:+.1f}"
                                     for length in lengths if length in res[c]))
    print("cold check (largest cached_tokens per cell, expected 0):",
          {c: max(v["cached_max"] for v in res[c].values()) for c in res})
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
