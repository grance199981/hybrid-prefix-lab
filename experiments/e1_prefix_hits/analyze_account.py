"""Summarise the offline accounting records (the *.jsonl files from account.py).

Pools the three seeds for each (dataset, model, thinking) cell and prints:
  - session / request counts and how the requests split by position
    (right after a user message, or inside a chain of tool calls);
  - the distribution of g_obs = P_prev - LCP(prompt, previous prompt), with the
    share equal to the value the chat template predicts (G_CHAT / G_CHAIN below:
    the tokens after "<|im_start|>assistant\n" in the generation prompt that the
    re-rendered history does not repeat);
  - whether D_R equals P_prev - g_obs (i.e. the previous prompt is the best match);
  - predicted hit rate and per-request recompute for each cache policy;
  - how often the unit-16 prompt-tail checkpoint is unusable, by residue.

Expects records written with --hybrid-block 528 (column hy528), as in the E0 run.

Usage: python experiments/e1_prefix_hits/analyze_account.py <dir with *.jsonl> [--json out.json]
"""
import collections
import glob
import json
import os
import statistics
import sys

from hplab.stats import percentile

G_CHAT = {("qwen35", "on"): 2, ("qwen35", "off"): 4, ("qwen3", "on"): 0, ("qwen3", "off"): 4}
G_CHAIN = {("qwen35", "on"): 1, ("qwen35", "off"): 0, ("qwen3", "on"): 0, ("qwen3", "off"): 4}
POLICIES = ["fa16_R", "fa16_I", "hy528", "hy16", "hy_exact_end_I"]


def load(d):
    cells = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        dataset, model, think, seed = os.path.basename(path)[:-6].split("_")
        key = (dataset, model, think[5:])
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["seed"] = seed
                cells[key].append(r)
    return cells


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def cell_stats(key, rows):
    dataset, model, think = key
    rs = [r for r in rows if r["k"] > 0]
    out = {"dataset": dataset, "model": model, "thinking": think,
           "sessions": len({(r["seed"], r["sid"]) for r in rows}),
           "requests": len(rs),
           "prompt_median": statistics.median(r["P"] for r in rs),
           "prompt_p90": percentile(sorted(r["P"] for r in rs), 0.9)}
    for pos, name in ((True, "after_user"), (False, "in_chain")):
        sub = [r for r in rs if r["after_user_query"] == pos]
        if not sub:
            continue
        expected = (G_CHAT if pos else G_CHAIN)[(model, think)]
        hist = collections.Counter(r["g_obs"] for r in sub)
        out[name] = {
            "n": len(sub),
            "g_expected": expected,
            "g_match_pct": pct(hist[expected], len(sub)),
            "g_top": hist.most_common(4),
            "D_is_prev_prompt_pct": pct(sum(r["D_R"] == r["P_prev"] - r["g_obs"] for r in sub), len(sub)),
            "reply_reusable_pct": pct(sum(r["D_I"] > r["P_prev"] for r in sub), len(sub)),
        }
    total_p = sum(r["P"] for r in rs)
    for pol in POLICIES:
        rec = sorted(r["P"] - r[pol] for r in rs)
        out[pol] = {"hit_rate_pct": pct(sum(r[pol] for r in rs), total_p),
                    "recompute_median": statistics.median(rec),
                    "recompute_p90": percentile(rec, 0.9)}
    # unit-16 tail checkpoint of the previous request unusable for this request
    miss = [r for r in rs if (r["P_prev"] - 1) // 16 * 16 > r["P_prev"] - r["g_obs"]]
    out["hy16_prev_ckpt_unusable_pct"] = pct(len(miss), len(rs))
    out["hy16_prev_ckpt_unusable_residues"] = sorted(collections.Counter(r["residue16"] for r in miss).items())
    return out


def main():
    d = sys.argv[1]
    cells = load(d)
    stats = [cell_stats(k, v) for k, v in sorted(cells.items())]
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w", encoding="utf-8", newline=chr(10)) as f:
            json.dump(stats, f, indent=1)
    for s in stats:
        print(f"\n=== {s['dataset']} | {s['model']} | thinking {s['thinking']} | "
              f"{s['sessions']} sessions, {s['requests']} requests, "
              f"prompt median {s['prompt_median']:.0f} p90 {s['prompt_p90']}")
        for name in ("after_user", "in_chain"):
            if name in s:
                x = s[name]
                print(f"  {name:10s} n={x['n']:6d}  g expected {x['g_expected']}: {x['g_match_pct']:6.2f}%  "
                      f"top {x['g_top']}  D=prev prompt {x['D_is_prev_prompt_pct']:6.2f}%  "
                      f"reply reusable {x['reply_reusable_pct']:6.2f}%")
        for pol in POLICIES:
            x = s[pol]
            print(f"  {pol:15s} hit {x['hit_rate_pct']:6.2f}%  recompute median {x['recompute_median']:7.0f}  "
                  f"p90 {x['recompute_p90']:7d}")
        print(f"  hy16: previous tail checkpoint unusable {s['hy16_prev_ckpt_unusable_pct']:.2f}%  "
              f"residues {s['hy16_prev_ckpt_unusable_residues']}")


if __name__ == "__main__":
    main()
