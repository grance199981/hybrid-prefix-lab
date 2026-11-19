"""E12 B: which checkpoint did the requests that still lack only the state (with PR #61282) need?

For every short request (hit < infinite-cache prediction) classified "state_only" by analyze_e10.py,
look at its own session: which earlier turn's prompt checkpoint equals the infinite-cache prediction
(pred_inf). With #61282 each request holds the state at its own prompt checkpoint until it ends and
frees it last, so the hypothesis (written down before the run, see README.md) is that these requests
need the checkpoint of a turn before the previous one, which nobody holds.
Usage: b_leftover.py <hook+e5 dir of one arm> [--unit 16] [--json out.json]
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e10_state_eviction"))
import analyze_e10 as A  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--unit", type=int, default=16)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    clients, events = A.load_dir(a.dir)
    kinds, _ = A.group_kinds(events)
    out = collections.defaultdict(lambda: collections.Counter())
    examples = []
    for label, rows in clients.items():
        cell = A.cell_of(label)
        f = A.analyze_file(label, rows, events, kinds, 2000, 250)
        sessions = collections.defaultdict(dict)
        for r in rows:
            if r.get("sid") != "control" and not r.get("error"):
                sessions[r["sid"]][r["k"]] = r
        # re-run the matching to get each request's admission lookup (analyze_file does not return it)
        lookups = collections.defaultdict(list)
        t0 = min(r["t_send_ns"] for r in rows if r.get("t_send_ns"))
        t1 = max(r.get("t_end_ns") or r["t_send_ns"] for r in rows if r.get("t_send_ns"))
        for e in events:
            if e["ev"] == "lookup" and t0 <= e["t"] <= t1:
                lookups[e["rid"]].append(e)
        first = sorted((ls[0]["t"], rid, ls[0]["P"]) for rid, ls in lookups.items())
        taken = set()
        for r in sorted((r for r in rows if r.get("sid") != "control" and not r.get("error")),
                        key=lambda r: r["t_send_ns"] or 0):
            P = r.get("prompt_tokens") or r["P"]
            cand = [rid for t, rid, p in first if r["t_send_ns"] <= t <= (r.get("t_end_ns") or t1)
                    and p == P and rid not in taken]
            if len(cand) != 1:
                continue
            taken.add(cand[0])
            adm = [e for e in lookups[cand[0]] if e.get("preempted", 0) == 0] or lookups[cand[0]]
            adm = adm[-1]
            if r["k"] == 0 or r["cached_tokens"] >= r["pred_inf"]:
                continue
            attn, state = A.split_groups(adm["groups"], kinds)
            if A.classify(attn, state, r["pred_inf"]) != "state_only":
                continue
            s = sessions[r["sid"]]
            need = r["pred_inf"]
            src = "unknown"
            for j in range(r["k"] - 1, -1, -1):
                if j in s and A.checkpoint(s[j]["P"], a.unit) == need:
                    src = "previous turn" if j == r["k"] - 1 else f"turn k-{r['k'] - j}"
                    break
            else:
                prev = s.get(r["k"] - 1)
                if prev and need < A.checkpoint(prev["P"], a.unit):
                    src = "below the previous checkpoint (block or other state)"
            out[cell][src] += 1
            if len(examples) < 12:
                examples.append({"cell": cell, "sid": r["sid"], "k": r["k"], "P": P, "pred_inf": need,
                                 "hit": r["cached_tokens"], "attn": attn, "state": state, "source": src,
                                 "prev_P": s.get(r["k"] - 1, {}).get("P")})
    res = {"by_cell": {c: dict(v) for c, v in out.items()}, "examples": examples}
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
