#!/bin/bash
# Build a clean environment for LatentMAS, because a leak from ~/.local breaks python3.10 environments here.
# Run this script on the login node, because it needs network access.
#
# Root cause of the broken `latentmas` environment, and of any python3.10 environment on this
# account: ~/.local/lib/python3.10/site-packages holds 931 MB of user-site packages, and
# site.ENABLE_USER_SITE is True. That folder is therefore on sys.path of every python3.10
# environment. Its nvidia-nccl-cu12 2.19.3 shadows the nccl that the environment's torch
# installed, which causes "undefined symbol: ncclCommResume". Rebuilding the environment does
# not fix that. PYTHONNOUSERSITE does.
#
# The leak also makes a bare `pip install` destructive. pip treats the ~/.local packages as
# existing installations and uninstalls them from your home folder.
# PYTHONNOUSERSITE=1 below hides ~/.local from the interpreter and from pip.
set -euo pipefail

export PYTHONNOUSERSITE=1
export PIP_USER=0

PUB=/pub/$USER
ENVDIR=$PUB/envs/latentmas
export HF_HOME=$PUB/hf

mkdir -p "$PUB/envs" "$HF_HOME"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p "$ENVDIR" python=3.10
conda activate "$ENVDIR"

# The CUDA 12.8 wheels cover sm_80 (A30, A100) and sm_89 (L40S, RTX6000 Ada).
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
# transformers stays pinned, because 5.x removed Cache.to_legacy_cache and from_legacy_cache,
# which the original _truncate_past in methods/latent_mas.py called, and deprecated torch_dtype=.
# Version 4.57.1 matches the API of the paper's time.
pip install --no-cache-dir "transformers==4.57.1" accelerate datasets numpy tqdm matplotlib

python - <<'PY'
import site, sys, torch, transformers
leaked = [p for p in sys.path if ".local" in p]
assert not leaked, f"~/.local leaked into sys.path: {leaked}"
print("user-site visible:", site.ENABLE_USER_SITE, "| leaked paths:", leaked)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("nccl", torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else "n/a")
print("transformers", transformers.__version__)
PY

echo
echo "Env ready at $ENVDIR"
echo "Activate with: conda activate $ENVDIR"
