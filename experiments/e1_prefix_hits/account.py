"""Offline accounting of prefix-cache reuse (CPU only, no model weights, no server).

For each sampled session and each request in it, render the prompt with the
model's own chat template and tokenizer, then compute

  P      prompt length in tokens
  D_R    longest common token prefix with any earlier *prompt* of the session
         ("recorded replay": the model's own output is never carried back)
  D_I    same, but earlier requests also contribute the tokens the template
         would re-render for their assistant reply ("ideal round trip": upper
         bound on what carrying the reply back verbatim could make reusable)

and the hit each cache policy would give (hplab.cache_model.predict_hits):
fa16_R / fa16_I (full attention, 16-token blocks), hy<block> and hy16 (hybrid
model, one state saved at the prompt-tail checkpoint), hy_exact_end_I (hybrid
model with a state kept at the exact end of every earlier request).

Output, per (dataset, model, thinking, seed): <tag>.jsonl with one record per
request and <tag>.summary.json with hit rates and recompute medians per policy,
tag = <dataset>_<label>_think<on|off>_seed<seed>.

Usage (in an environment with transformers and pyarrow):
  python experiments/e1_prefix_hits/account.py --dataset wildchat --path <parquet> \
      --models qwen35=<tokenizer dir>,qwen3=<tokenizer dir> \
      --seed 1 --n-sessions 300 --out <dir> [--hybrid-block 576]
Sessions are read once and rendered for every model and both thinking settings.
"""
import argparse
import json
import os

from hplab.cache_model import FA_BLOCK, HYBRID_BLOCK, policy_names, predict_hits, summarize
from hplab.sessions import LOADERS, render_session


def account(tok, sessions, thinking, max_len, out_path, aligns=(HYBRID_BLOCK, FA_BLOCK)):
    """Render every session, write one record per request to out_path, return the summary."""
    rows, n_trunc, n_err, n_used = [], 0, 0, 0
    with open(out_path, "w", encoding="utf-8", newline=chr(10)) as f:
        for sid, messages, tools in sessions:
            try:
                prompts, ideals, flags, truncated = render_session(
                    tok, messages, tools, thinking, max_len)
            except Exception as e:  # template refused the session; count, do not hide
                n_err += 1
                print(f"render error {sid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            n_trunc += truncated
            if len(prompts) < 2:
                continue
            n_used += 1
            for r, fl in zip(predict_hits(prompts, ideals, aligns), flags):
                r.update(fl)
                r["sid"] = sid
                r.setdefault("g_obs", None)
                rows.append(r)
                f.write(json.dumps(r) + chr(10))
    summary = summarize(rows, policy_names(aligns))
    summary.update({"sessions_loaded": len(sessions), "sessions_truncated": n_trunc,
                    "sessions_render_error": n_err, "sessions_used": n_used})
    return summary


def main():
    ap = argparse.ArgumentParser(description="Offline prefix-cache accounting (CPU only).")
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--path", required=True)
    ap.add_argument("--tools-json", default=None)
    ap.add_argument("--models", required=True, help="label=tokenizer_dir[,label=dir...]")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-sessions", type=int, required=True)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--hybrid-block", type=int, default=HYBRID_BLOCK,
                    help="hybrid model block size: 528 on vLLM 0.32, 576 on vLLM 1479deb0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    os.makedirs(args.out, exist_ok=True)
    sessions = list(LOADERS[args.dataset](args.path, args.n_sessions, args.seed,
                                          tools_json=args.tools_json))
    print(f"loaded {len(sessions)} sessions from {args.dataset} (seed {args.seed})", flush=True)
    for spec in args.models.split(","):
        label, tok_dir = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(tok_dir)
        for thinking in ("on", "off"):
            tag = f"{args.dataset}_{label}_think{thinking}_seed{args.seed}"
            summary = account(tok, sessions, thinking == "on", args.max_len,
                              os.path.join(args.out, tag + ".jsonl"),
                              aligns=(args.hybrid_block, FA_BLOCK))
            summary.update({"tag": tag, "args": vars(args), "tokenizer_dir": tok_dir})
            with open(os.path.join(args.out, tag + ".summary.json"), "w",
                      encoding="utf-8", newline=chr(10)) as f:
                json.dump(summary, f, indent=1)
            print(tag, json.dumps({k: v for k, v in summary.items()
                                   if k.startswith(("n_", "sessions_"))}), flush=True)


if __name__ == "__main__":
    main()
