"""Group the requests of a four-pass replay by how many more prefill steps the
--prefix-match-unit configuration needs than the default one, by the
scheduler's own rule, and report the paired TTFT difference of each group.

The step count is hplab.cache_model.count_steps, a transcription of
vllm/v1/core/sched/scheduler.py:_mamba_block_aligned_split and of
vllm/v1/kv_cache_interface.py:is_mamba_prefill_checkpoint_valid at vLLM 1479deb0:
  --alignment omitted : the backend cannot save the prompt checkpoint inside the
                        final chunk (vLLM main with the Triton GDN kernels)
  --alignment 64      : it can, on a 64-token grid (PR #65186)
  --rule 66938        : the validity check after upstream PR #66938
                        (`checkpoint_col >= query_start // block` instead of
                        `checkpoint_col > (query_start - 1) // block`)
These are step counts computed from each request's prompt length and cache
hit, not measured ones; steps_from_log.py measures them from the server log.
Not modelled: speculative decoding, shared-prefix junctions, several requests
in one batch (the replay sends one request at a time).

Input: a directory with the request records of the four passes C0a, C1a, C1b,
C0b (scripts/run_replay.sh, R replay, interleave 1). A request counts if it has
a TTFT in all four passes; its difference is mean(C1a, C1b) - mean(C0a, C0b).
Output: groups by extra steps (all requests, and first requests only) and by
tail length (prompt minus the hit under the unit), printed and with --json saved.

Usage: python experiments/e7_upstream_prs/split_steps.py <dir> --unit 64
           [--alignment 64] [--rule 1479deb0|66938] [--json out.json]
"""
import argparse
import glob
import json
import os
import re
import statistics as st

from hplab.cache_model import count_steps


def load_pass(directory, cfg):
    out = {}
    for f in glob.glob(os.path.join(directory, f"{cfg}_*_R_s1_seed*.jsonl")):
        m = re.match(rf"{cfg}_(\w+?)_think(on|off)_R_s1_seed(\d+)\.jsonl", os.path.basename(f))
        if not m:
            continue
        ds, th, seed = m.groups()
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("sid") == "control" or r.get("error") or r.get("ttft_ms") is None:
                    continue
                out[(ds, th, seed, r["sid"], r["k"])] = r
    return out


def paired_rows(directory):
    """One row per request that has a TTFT in all four passes."""
    passes = {c: load_pass(directory, c) for c in ("C0a", "C1a", "C1b", "C0b")}
    rows = []
    for key, a in passes["C0a"].items():
        if not all(key in passes[c] for c in passes):
            continue
        b, c, d = passes["C1a"][key], passes["C1b"][key], passes["C0b"][key]
        rows.append({
            "dataset": key[0], "k": key[4], "prompt": a["prompt_tokens"],
            "cached_default": a["cached_tokens"], "cached_unit": b["cached_tokens"],
            "diff": (b["ttft_ms"] + c["ttft_ms"]) / 2 - (a["ttft_ms"] + d["ttft_ms"]) / 2,
        })
    return rows


def summary(g):
    diffs = [r["diff"] for r in g]
    return {
        "n": len(g),
        "ttft_diff_median": round(st.median(diffs), 1),
        "share_slower": round(sum(x > 0 for x in diffs) / len(diffs), 3),
        "saved_tokens_median": st.median(r["cached_unit"] - r["cached_default"] for r in g),
        "tail_unit_min_max": [min(r["prompt"] - r["cached_unit"] for r in g),
                              max(r["prompt"] - r["cached_unit"] for r in g)],
    }


def group_by_steps(rows, unit, alignment, rule):
    out = {}
    for r in rows:
        s0 = count_steps(r["cached_default"], r["prompt"], None, alignment, rule=rule)
        s1 = count_steps(r["cached_unit"], r["prompt"], unit, alignment, rule=rule)
        out.setdefault(r["dataset"], {}).setdefault(s1 - s0, []).append(r)
    return {ds: {str(d): summary(g) for d, g in sorted(groups.items())} for ds, groups in out.items()}


TAIL_BUCKETS = ((1, 64), (65, 512), (513, 576), (577, 1152), (1153, None))


def group_by_tail(rows):
    """Tail = tokens left to compute under the unit.  Requests without a hit in either configuration
    ("cold") and requests with a hit under the unit ("hit") are kept apart."""
    out = {}
    for r in rows:
        if r["cached_unit"] == 0 and r["cached_default"] == 0:
            kind = "cold"
        elif r["cached_unit"] > 0:
            kind = "hit"
        else:
            continue
        tail = r["prompt"] - r["cached_unit"]
        for lo, hi in TAIL_BUCKETS:
            if tail >= lo and (hi is None or tail <= hi):
                label = f"{lo}-{hi}" if hi else f"{lo}+"
                out.setdefault(r["dataset"], {}).setdefault(kind, {}).setdefault(label, []).append(r)
    order = [f"{lo}-{hi}" if hi else f"{lo}+" for lo, hi in TAIL_BUCKETS]
    return {ds: {kind: {lab: summary(g[lab]) for lab in order if lab in g} for kind, g in kinds.items()}
            for ds, kinds in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory")
    ap.add_argument("--unit", type=int, required=True)
    ap.add_argument("--alignment", type=int, default=None, help="64 with PR #65186; omit for main")
    ap.add_argument("--rule", default="1479deb0", choices=("1479deb0", "66938"),
                    help="validity rule: vLLM 1479deb0, or main after PR #66938")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = paired_rows(a.directory)
    res = {"unit": a.unit, "alignment": a.alignment, "rule": a.rule, "n_requests": len(rows),
           "by_extra_steps": group_by_steps(rows, a.unit, a.alignment, a.rule),
           "first_requests_by_extra_steps": group_by_steps([r for r in rows if r["k"] == 0], a.unit, a.alignment, a.rule),
           "by_tail": group_by_tail(rows)}
    print(f"{len(rows)} requests; unit {a.unit}; alignment {a.alignment}; rule {a.rule}")
    for title in ("by_extra_steps", "first_requests_by_extra_steps"):
        print(f"== {title}")
        for ds, groups in res[title].items():
            for d, g in groups.items():
                print(f"  {ds:9s} extra steps {d:>2}: n={g['n']:4d}  TTFT diff median {g['ttft_diff_median']:+6.1f} ms  "
                      f"slower {g['share_slower']:.2f}  saved tokens {g['saved_tokens_median']}  tail {g['tail_unit_min_max']}")
    print("== by_tail")
    for ds, kinds in res["by_tail"].items():
        for kind, groups in kinds.items():
            for lab, g in groups.items():
                print(f"  {ds:9s} {kind:4s} tail {lab:>9}: n={g['n']:4d}  TTFT diff median {g['ttft_diff_median']:+6.1f} ms  "
                      f"slower {g['share_slower']:.2f}  saved tokens {g['saved_tokens_median']}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
