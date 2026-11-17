"""Read the per-request records of a four-pass replay.

A four-pass run (scripts/run_replay.sh with REPLAY_CONFIGS="C0a C1a C1b C0b")
writes one JSON-lines file per pass, dataset, thinking mode and seed:

    <pass>_<dataset>_think<on|off>_R_s1_seed<seed>.jsonl

C0 is the default server configuration and C1 the configuration under test. The
order C0a, C1a, C1b, C0b cancels a steady drift of the machine over the run.
"""
import collections
import glob
import json
import os

BLOCKS = ("C0a", "C1a", "C1b", "C0b")
DATASETS = ("wildchat", "smith", "rebench")
THINKS = ("on", "off")


def load_dir(d):
    """{(pass, dataset, thinking): {(seed, sid, k): row}} for the R-replay records
    in directory d. Control rows and failed requests are left out; k is the
    request's index in its session (0 = first request)."""
    cells = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        parts = os.path.basename(f)[:-6].split("_")
        if len(parts) != 6 or parts[0] not in BLOCKS:
            continue
        block, ds, th, mode, s, seed = parts
        if mode != "R" or s != "s1":
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("sid") == "control" or r.get("error"):
                    continue
                cells[(block, ds, th[5:])][(int(seed[4:]), r["sid"], r["k"])] = r
    return cells


def pairs(cells, ds, ths, subset, metric):
    """[[C0a, C1a, C1b, C0b] rows of one request] for the requests of dataset ds
    and thinking modes ths present in all four passes with `metric` measured in
    each. subset: "first" (k == 0) or "scored" (k > 0)."""
    out = []
    for th in ths:
        per = [cells.get((b, ds, th), {}) for b in BLOCKS]
        for key in sorted(set(per[0]).intersection(*per[1:])):
            if (key[2] == 0) != (subset == "first"):
                continue
            rows = [p[key] for p in per]
            if all(r.get(metric) is not None for r in rows):
                out.append(rows)
    return out
