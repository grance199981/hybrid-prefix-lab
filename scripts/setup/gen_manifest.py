"""Build a download manifest (scripts/setup/manifests/*.json) from the official
huggingface.co API: repo, revision, and per file the size and sha256 (LFS) or git
blob sha1. Files later fetched from a mirror are verified against it by fetch_assets.py.
Needs access to huggingface.co. Standard library only.

Targets (the shipped manifests were generated with these; revisions move upstream,
so a regenerated manifest pins whatever is current):
  datasets         the three dataset files and the Qwen3-4B tokenizer -> manifests/datasets.json
  qwen3-4b         Qwen3-4B weights                                    -> manifests/qwen3-4b.json
  qwen3.5-9b       Qwen3.5-9B weights (E3)                             -> manifests/qwen3.5-9b.json
  qwen3.5-27b-fp8  Qwen3.5-27B-FP8 weights (E6)                        -> manifests/qwen3.5-27b-fp8.json
  qwen3.5-4b       Qwen3.5-4B weights; no manifest is shipped for it (the runs used an
                   existing copy whose chat template equals refs/qwen3.5-4b_chat_template.jinja)

Usage: python scripts/setup/gen_manifest.py datasets > scripts/setup/manifests/datasets.json
"""
import json
import sys
import urllib.request

WANTED_DATASETS = [
    ("datasets", "nebius/SWE-rebench-openhands-trajectories", "data/swe-rebench-openhands",
     ["trajectories.parquet", "tools.json", "config.toml", "README.md", "LICENSE"]),
    ("datasets", "SWE-bench/SWE-smith-trajectories", "data/swe-smith-trajectories",
     ["data/tool-00004-of-00008.parquet", "README.md"]),
    ("datasets", "allenai/WildChat-1M", "data/wildchat-1m",
     ["data/train-00008-of-00014.parquet", "README.md"]),
    ("models", "Qwen/Qwen3-4B", "models/Qwen3-4B-tokenizer",
     ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]),
]

# Model weights. None = every file in the repo. The 27B model is the official FP8 build:
# the bf16 weights (55.6 GB) do not fit one 48 GB GPU.
WANTED = {
    "datasets": WANTED_DATASETS,
    "qwen3-4b": [("models", "Qwen/Qwen3-4B", "models/Qwen3-4B", None)],
    "qwen3.5-9b": [("models", "Qwen/Qwen3.5-9B", "models/Qwen3.5-9B", None)],
    "qwen3.5-27b-fp8": [("models", "Qwen/Qwen3.5-27B-FP8", "models/Qwen3.5-27B-FP8", None)],
    "qwen3.5-4b": [("models", "Qwen/Qwen3.5-4B", "models/Qwen3.5-4B", None)],
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "hf-asset-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    assets = []
    if len(sys.argv) != 2 or sys.argv[1] not in WANTED:
        sys.exit("usage: gen_manifest.py {" + ",".join(WANTED) + "}")
    for kind, repo, local_dir, paths in WANTED[sys.argv[1]]:
        info = get(f"https://huggingface.co/api/{kind}/{repo}?blobs=true")
        by_path = {s["rfilename"]: s for s in info["siblings"]}
        files = []
        for p in (paths if paths is not None else sorted(by_path)):
            s = by_path[p]  # KeyError here means the file list changed upstream
            lfs = s.get("lfs")
            files.append({
                "path": p,
                "size": lfs["size"] if lfs else s["size"],
                "sha256": lfs["sha256"] if lfs else None,
                "git_blob_sha1": None if lfs else s["blobId"],
            })
        card = info.get("cardData") or {}
        assets.append({
            "kind": kind, "repo": repo, "revision": info["sha"], "local_dir": local_dir,
            "license": card.get("license"),
            "total_bytes": sum(f["size"] for f in files), "files": files,
        })
    json.dump({"generated_from": "https://huggingface.co/api", "assets": assets},
              sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
