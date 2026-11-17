#!/usr/bin/env bash
# Create the measurement environment $HP_ROOT/envs/vllm-main: a venv with vLLM pinned
# to main @ 1479deb0 (the build every number of E1-E6 was measured on).
#
# - A system Python 3.12 is used only as the base interpreter; nothing is installed into it.
# - All caches stay under ROOT.
# - vLLM comes from the per-commit wheel index of the pinned commit, so the installed
#   code is exactly the commit whose source the experiments cite.
# - Dependencies may come from a PyPI mirror (PIP_INDEX_MIRROR). pip writes an install
#   report with the sha256 of every archive; check it afterwards against pypi.org with
#   scripts/setup/verify_install_report.py.
# - The server's system nvcc was too old for FlashInfer's JIT sampler; the launchers
#   set VLLM_USE_FLASHINFER_SAMPLER=0 (scripts/serve_lib.sh), nothing to do here.
#
# Usage: BASE_PY=<python3.12> bash scripts/setup/setup_env.sh $HP_ROOT
# Output: ROOT/envs/vllm-main, and in ROOT/logs the install reports, the pip freeze and
#         `vllm serve --help=all` (the flag reference for this build).
set -euo pipefail

ROOT=${1:?usage: setup_env.sh <experiment root>}
BASE_PY=${BASE_PY:-python3.12}
ENV=$ROOT/envs/vllm-main
VLLM_COMMIT=1479deb0392f1e7d2a69399e4150202284c24509
VLLM_VERSION="0.34.0rc1.dev176+g1479deb03"
INDEX_URL=${PIP_INDEX_MIRROR:-https://pypi.org/simple}

export PIP_CACHE_DIR=$ROOT/cache/pip
export TMPDIR=$ROOT/tmp
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_INDEX_URL=$INDEX_URL
export PIP_RETRIES=10
export PIP_DEFAULT_TIMEOUT=60
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$ROOT/logs" "$ROOT/envs"

if [ -e "$ENV" ]; then
  echo "refusing to overwrite existing env: $ENV" >&2
  exit 2
fi

echo "[$(date -Is)] base python: $($BASE_PY -V)"
$BASE_PY -m venv "$ENV"
PY=$ENV/bin/python
$PY -m pip install -U pip wheel setuptools

echo "[$(date -Is)] installing vllm==$VLLM_VERSION (commit $VLLM_COMMIT)"
echo "index: $INDEX_URL"
$PY -m pip install "vllm==$VLLM_VERSION" --extra-index-url "https://wheels.vllm.ai/$VLLM_COMMIT" --report "$ROOT/logs/vllm_main_install_report.json"

# Freeze what vLLM pinned so the helper packages below cannot move it.
$PY -m pip freeze | grep -i -E '^(torch|torchvision|torchaudio|transformers|tokenizers|vllm|flashinfer-python|triton|numpy|safetensors)==' \
  > "$ROOT/logs/vllm_main_core_constraints.txt"
echo "core constraints:"; cat "$ROOT/logs/vllm_main_core_constraints.txt"

echo "[$(date -Is)] installing test / data helpers"
$PY -m pip install pytest pyarrow -c "$ROOT/logs/vllm_main_core_constraints.txt" --report "$ROOT/logs/vllm_main_install_report_helpers.json"

echo "[$(date -Is)] pip check"
$PY -m pip check || echo "PIP_CHECK_REPORTED_CONFLICTS"

$PY -m pip freeze > "$ROOT/logs/env_vllm_main_freeze.txt"
$PY - <<'EOF'
import importlib.metadata as md
for p in ["vllm", "torch", "transformers", "tokenizers", "flashinfer-python", "triton", "numpy", "openai", "pyarrow", "pytest"]:
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p}: NOT INSTALLED")
import torch
print("torch.cuda build:", torch.version.cuda)
EOF

echo "[$(date -Is)] recording CLI flags"
CUDA_VISIBLE_DEVICES= "$ENV/bin/vllm" serve --help=all > "$ROOT/logs/vllm_main_serve_help.txt" 2>&1 \
  || CUDA_VISIBLE_DEVICES= "$ENV/bin/vllm" serve --help > "$ROOT/logs/vllm_main_serve_help.txt" 2>&1 \
  || echo "HELP_DUMP_FAILED"
wc -l "$ROOT/logs/vllm_main_serve_help.txt"
echo "[$(date -Is)] SETUP_ENV_DONE"
