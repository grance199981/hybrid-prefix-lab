"""E10: join the hook's events to the client's request records and classify the misses.

Input: one or more directories, each holding client records (*.jsonl written by
pressure_replay.py or cpu_replay.py, directly or under e5/) and the hook's events
(hook/events_*.jsonl). Standard library only.

For each client file (one label = config, pool, output length, seed):
  - the admission lookup of a request is its last `lookup` event before any preemption;
    synthetic requests are matched by session (rank of the prompt's first token among
    the sessions of that file) and turn ((P - p0) / step); controls by prompt length
    inside their own time window;
  - retention = sum of hits / sum of infinite-cache predictions, turns k > 0;
  - every k > 0 request with hit 0 is classified, with c_prev the previous turn's
    prompt checkpoint, attention = the attention group's own longest hit, state = the
    smallest of the Mamba groups' own longest hits:
        state_only   attention >= c_prev > state
        both         attention <  c_prev and state < c_prev
        attn_only    state >= c_prev > attention
    `first32` repeats the count over the 32 earliest such requests of the file;
  - at each request's final `end` event: is the state at its own prompt checkpoint
    still held by somebody (ref > 0), in the free queue (cached, ref 0), or gone;
  - evictions by group kind (only those that happened while handing out new blocks).

Usage: analyze_e10.py DIR [DIR ...] [--p0 2000 --step 250] [--json out.json]
"""
import argparse
import collections
import glob
import json
import os
import re


def checkpoint(P, unit=16):
    return max((P - 1) // unit * unit, 0)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dir(d):
    files = [f for f in glob.glob(os.path.join(d, "*.jsonl")) + glob.glob(os.path.join(d, "e5", "*.jsonl"))
             if not f.endswith(".metrics.jsonl")]
    clients = {}
    for f in sorted(files):
        rows = read_jsonl(f)
        if rows and "label" in rows[0]:
            clients[rows[0]["label"]] = rows
    events = []
    for f in glob.glob(os.path.join(d, "hook", "events_*.jsonl")):
        events += read_jsonl(f)
    events.sort(key=lambda e: e["t"])
    return clients, events


def group_kinds(events):
    inits = [e for e in events if e["ev"] == "init"]
    kinds = {tuple(e["groups"]) for e in inits}
    assert len(kinds) <= 1, f"different group layouts in one directory: {kinds}"
    return list(kinds.pop()) if kinds else None, inits


def split_groups(lengths, kinds):
    attn = [n for n, k in zip(lengths, kinds) if k == "attn"]
    state = [n for n, k in zip(lengths, kinds) if k == "mamba"]
    return (min(attn) if attn else None), (min(state) if state else None)


def classify(attn, state, c_prev):
    if attn is None or state is None:
        return "unknown"
    if attn >= c_prev > state:
        return "state_only"
    if state >= c_prev > attn:
        return "attn_only"
    if attn < c_prev and state < c_prev:
        return "both"
    return "neither"            # both reach c_prev but the hit was 0: should not happen


def analyze_file(label, rows, events, kinds, p0, step):
    t0 = min(r["t_send_ns"] for r in rows if r.get("t_send_ns"))
    t1 = max(r.get("t_end_ns") or r["t_send_ns"] for r in rows if r.get("t_send_ns"))
    ev = [e for e in events if t0 <= e["t"] <= t1]
    synth = [r for r in rows if r.get("sid") != "control" and not r.get("error")]
    controls = [r for r in rows if r.get("sid") == "control"]

    def is_synth(P):
        return P >= p0 and (P - p0) % step == 0

    lookups = collections.defaultdict(list)
    ends = {}
    for e in ev:
        if e["ev"] == "lookup":
            lookups[e["rid"]].append(e)
        elif e["ev"] == "end":
            ends[e["rid"]] = e                      # the last one wins (a preemption also frees)
    # pressure_sessions.synth_sessions spells session i in its first `width` tokens, in base L =
    # number of distinct ids in its vocabulary: token d is vocab[i // L**d % L], vocab sorted.
    # With L >= n_sessions the first token alone tells the sessions apart.
    tok0s = sorted({ls[0]["tok0"] for ls in lookups.values() if is_synth(ls[0]["P"])})
    rank = {t: i for i, t in enumerate(tok0s)}
    n_sess = len({r["sid"] for r in synth})
    L, width = len(tok0s), 1
    while L > 1 and L ** width < n_sess:
        width += 1

    def session(e):
        head = e.get("head") or [e["tok0"]]
        return sum(rank[head[d]] * L ** d for d in range(width))

    def admission(ls):
        pre = [e for e in ls if e.get("preempted", 0) == 0]
        return (pre or ls)[-1]

    by_key = {}
    windowed = not any(str(r["sid"]).startswith("synth-") for r in synth)
    ambiguous = 0
    if windowed:
        # Recorded sessions (no session id in the tokens): a request is the one lookup id whose
        # first lookup falls inside the request's own send..end window and has the same prompt
        # length. Two candidates (or none) leave the request unmatched; `agree` checks the rest.
        first = sorted((ls[0]["t"], rid, ls[0]["P"]) for rid, ls in lookups.items())
        taken = set()
        for r in sorted(synth, key=lambda r: r["t_send_ns"] or 0):
            if not r.get("t_send_ns"):
                continue
            P = r.get("prompt_tokens") or r["P"]
            end = r.get("t_end_ns") or t1
            cand = [rid for t, rid, p in first if r["t_send_ns"] <= t <= end and p == P and rid not in taken]
            if len(cand) == 1:
                taken.add(cand[0])
                by_key[(r["sid"], r["k"])] = (cand[0], admission(lookups[cand[0]]))
            elif len(cand) > 1:
                ambiguous += 1
    else:
        for rid, ls in lookups.items():
            if not is_synth(ls[0]["P"]):
                continue
            adm = admission(ls)
            by_key[(f"synth-{session(adm)}", (adm["P"] - p0) // step)] = (rid, adm)

    out = {"label": label, "n_sessions_seen": len({k[0] for k in by_key}), "n": 0, "matched": 0, "agree": 0,
           "ambiguous": ambiguous, "hit": 0, "pred": 0, "zero": 0, "short": 0, "classes": collections.Counter(),
           "short_classes": collections.Counter(),
           "first32": collections.Counter(), "end": collections.Counter(), "evict": collections.Counter(),
           "controls": {}}
    zero_rows = []
    for r in sorted(synth, key=lambda r: r["t_send_ns"] or 0):
        hit_client = r.get("cached_tokens")
        m = by_key.get((r["sid"], r["k"]))
        if m is not None:
            out["matched"] += 1
            rid, adm = m
            if hit_client is not None and adm["hit"] == hit_client:
                out["agree"] += 1
            e = ends.get(rid)
            if e is not None:
                st = e["state"]
                if not st or not all(s["present"] for s in st):
                    out["end"]["gone"] += 1
                elif all((s["ref"] or 0) > 0 for s in st):
                    out["end"]["held"] += 1
                else:
                    out["end"]["free_queue"] += 1
        if r["k"] == 0:
            continue
        hit = hit_client if hit_client is not None else (m[1]["hit"] if m else None)
        if hit is None:
            continue
        out["n"] += 1
        out["hit"] += hit
        out["pred"] += r["pred_inf"]
        c_prev = r["pred_inf"]
        if hit < c_prev:
            # short of the infinite-cache hit (zero or an earlier checkpoint): which part
            # stopped short of it
            cls = classify(*split_groups(m[1]["groups"], kinds), c_prev) if m and kinds else "unmatched"
            out["short"] += 1
            out["short_classes"][cls] += 1
        if hit == 0:
            out["zero"] += 1
            cls = classify(*split_groups(m[1]["groups"], kinds), c_prev) if m and kinds else "unmatched"
            out["classes"][cls] += 1
            zero_rows.append(cls)
    for cls in zero_rows[:32]:
        out["first32"][cls] += 1
    for e in ev:
        if e["ev"] == "evict" and e.get("alloc"):
            out["evict"][e.get("kind") or "?"] += 1
    for r in controls:
        win = [e for e in ev if e["ev"] == "lookup" and e["P"] == r["P"]
               and r["t_send_ns"] <= e["t"] <= r["t_send_ns"] + int((r.get("ttft_ms") or 0) * 1e6) + 1]
        if not win:
            out["controls"][r["control"]] = None
            continue
        e = win[-1]
        attn, state = split_groups(e["groups"], kinds) if kinds else (None, None)
        out["controls"][r["control"]] = {"hit": e["hit"], "client": r.get("cached_tokens"),
                                         "attn": attn, "state": state}
    out["retention"] = round(out["hit"] / out["pred"], 4) if out["pred"] else None
    for k in ("classes", "short_classes", "first32", "end", "evict"):
        out[k] = dict(out[k])
    return out


def control_check(c):
    """K2: B (new prompt) misses in every group; A_again hits in every group, and the
    lookup agrees with the client."""
    b, a = c.get("B"), c.get("A_again")
    if not b or not a:
        return None
    ok_b = b["hit"] == 0 and b["attn"] == 0 and b["state"] == 0
    ok_a = a["hit"] > 0 and a["hit"] == a["client"] and a["attn"] >= a["hit"] and a["state"] >= a["hit"]
    return ok_b and ok_a


CELL = re.compile(r"(?:(?P<cfg>[A-Z]\d)_[A-Z]_[a-z]+_kv(?P<mib>\d+)|cpu_kv(?P<blk>\d+)b)_out(?P<out>\d+)_seed(?P<seed>\d+)")


def cell_of(label):
    m = CELL.search(label)
    if not m:
        return label
    pool = f"{m['mib']}MiB" if m["mib"] else f"{m['blk']}blocks"
    return f"{pool}_out{m['out']}"


def summarize(per_file):
    cells = collections.defaultdict(list)
    for f in per_file:
        cells[cell_of(f["label"])].append(f)
    out = {}
    for cell, fs in sorted(cells.items()):
        hit, pred = sum(f["hit"] for f in fs), sum(f["pred"] for f in fs)
        agg = {k: collections.Counter() for k in ("classes", "short_classes", "first32", "end", "evict")}
        for f in fs:
            for k in agg:
                agg[k].update(f[k])
        zero = sum(f["zero"] for f in fs)
        short = sum(f["short"] for f in fs)
        out[cell] = {"files": len(fs), "n": sum(f["n"] for f in fs), "retention": round(hit / pred, 4) if pred else None,
                     "zero": zero, "short": short, "matched": sum(f["matched"] for f in fs),
                     "agree": sum(f["agree"] for f in fs), "ambiguous": sum(f["ambiguous"] for f in fs),
                     "share_short_state_only": (round(agg["short_classes"]["state_only"] / short, 3)
                                                if short else None),
                     "share_state_only": round(agg["classes"]["state_only"] / zero, 3) if zero else None,
                     "share_both": round(agg["classes"]["both"] / zero, 3) if zero else None,
                     "first32_state_only": (round(agg["first32"]["state_only"] / sum(agg["first32"].values()), 3)
                                            if agg["first32"] else None),
                     "end_not_held": (round((agg["end"]["free_queue"] + agg["end"]["gone"]) / sum(agg["end"].values()), 3)
                                      if agg["end"] else None),
                     "controls_ok": [control_check(f["controls"]) for f in fs],
                     **{k: dict(v) for k, v in agg.items()}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--p0", type=int, default=2000)
    ap.add_argument("--step", type=int, default=250)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    per_file, layout = [], None
    for d in a.dirs:
        clients, events = load_dir(d)
        kinds, inits = group_kinds(events)
        layout = layout or {"groups": kinds, "init": inits[:1]}
        for label, rows in clients.items():
            per_file.append(analyze_file(label, rows, events, kinds, a.p0, a.step))
    res = {"layout": layout, "cells": summarize(per_file), "files": per_file}
    for cell, s in res["cells"].items():
        print(f"{cell:22s} n={s['n']:4d} retention={s['retention']}  zero={s['zero']:4d}  "
              f"state_only={s['share_state_only']}  both={s['share_both']}  first32_state_only={s['first32_state_only']}  "
              f"short={s['short']} short_state_only={s['share_short_state_only']}  "
              f"end_not_held={s['end_not_held']}  agree={s['agree']}/{s['matched']}  ambiguous={s['ambiguous']}  "
              f"controls={s['controls_ok']}  "
              f"evict={s['evict']}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
