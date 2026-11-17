"""Post-hoc split of the R3 four-pass comparison: with --prefix-match-unit 64, the
paired TTFT difference PR minus base build per request, grouped by how many
fewer (or more) prefill steps the PR build takes than the base build. Step
counts come from the scheduler rule of PR #66938 (hplab.cache_model.count_steps,
alignment 64 on the PR build, none on the base build).

Input: a directory with two sub-directories, main/ (base build) and pr/ (PR
build), each holding the four passes C0a, C1a, C1b, C0b as written by
scripts/run_replay.sh. Only the two unit-64 passes (C1a, C1b) and the requests
after the first of each session are used; a request counts when it has a TTFT
in all four of those passes.
Output: per dataset the median, mean, share at least 10 ms faster, and the
groups by step difference, printed and with --json saved.

Usage: python experiments/e7_upstream_prs/pair_by_steps.py <dir with main/ and pr/> [--json out.json]
"""
import argparse
import collections
import json
import os
import statistics as st

from hplab.cache_model import count_steps
from hplab.records import DATASETS, THINKS, load_dir


def by_step_difference(m, p):
    res = {}
    for ds in DATASETS:
        groups = collections.defaultdict(list)
        allv = []
        for th in THINKS:
            keys = set(m[("C1a", ds, th)]) & set(m[("C1b", ds, th)]) & set(p[("C1a", ds, th)]) & set(p[("C1b", ds, th)])
            for key in sorted(keys):
                if key[2] == 0:
                    continue
                ma, mb = m[("C1a", ds, th)][key], m[("C1b", ds, th)][key]
                pa, pb = p[("C1a", ds, th)][key], p[("C1b", ds, th)][key]
                if any(r.get("ttft_ms") is None for r in (ma, mb, pa, pb)):
                    continue
                d = (pa["ttft_ms"] + pb["ttft_ms"]) / 2 - (ma["ttft_ms"] + mb["ttft_ms"]) / 2
                sm = count_steps(ma["cached_tokens"], ma["prompt_tokens"], 64, None, rule="66938")
                sp = count_steps(pa["cached_tokens"], pa["prompt_tokens"], 64, 64, rule="66938")
                groups[sp - sm].append(d)
                allv.append(d)
        res[ds] = {"n": len(allv), "median": round(st.median(allv), 1), "mean": round(st.fmean(allv), 1),
                   "share_faster_by_10": round(sum(v <= -10 for v in allv) / len(allv), 3),
                   "by_step_difference_pr_minus_base": {str(k): {"n": len(v), "share": round(len(v) / len(allv), 3),
                                                                 "median": round(st.median(v), 1)}
                                                        for k, v in sorted(groups.items())}}
        print(ds, json.dumps(res[ds]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory", help="contains main/ and pr/")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = by_step_difference(load_dir(os.path.join(a.directory, "main")), load_dir(os.path.join(a.directory, "pr")))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
