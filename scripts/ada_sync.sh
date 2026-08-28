#!/bin/bash
#SBATCH --job-name=neus-sync
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode091
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sync-%j.out

# Build the venv and prefetch the model. MUST run as a job, not on the login
# node: the login node is CentOS 7 with glibc 2.17 and no CXXABI_1.3.9, so
# nothing importing torch or numpy works there, and its `ulimit -v` cap kills
# the uv resolver with a bogus "memory allocation of ~700 KB failed".
#
#   sbatch scripts/ada_sync.sh
#
# First build takes ~30 min (~3.5 GB of CUDA wheels onto NFS); hence the 2 h
# limit. A job that runs short cannot be extended.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Never reuse a .venv built elsewhere: a torch linked against a different set of
# CUDA libraries than the ones beside it fails as "CUDA unknown error" from
# torch._C._cuda_init(), while nvidia-smi and cuInit all succeed. Misleading
# signature, and rm -rf is the only fix.
uv lock
rm -rf .venv
uv sync

uv run --no-sync python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# Compute nodes have outbound network, so cache the model here too.
MODEL="${MODEL:-facebook/bart-base}"
uv run --no-sync python - <<PY
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
name = "${MODEL}"
AutoTokenizer.from_pretrained(name)
AutoModelForSeq2SeqLM.from_pretrained(name)
print("cached", name)
PY

echo "sync done. now: sbatch scripts/ada_run.sh"
