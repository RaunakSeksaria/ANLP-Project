#!/bin/bash
#SBATCH --job-name=detoxify-fallacy
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/detoxify-%j.out
#SBATCH --error=logs/detoxify-%j.err

# Scores all 51,055 texts (6.03M tokens) across the four fallacy datasets with
# Detoxify 'original-small' (albert-base-v2). One GPU is plenty — the 164-GPU
# allowance buys nothing here, the job is a few minutes of compute.
#
#   sbatch scripts/slurm_toxicity.sh
#
# CPU-only fallback (no --gres, --cpus-per-task=32):
#   uv run python scripts/score_all.py --device cpu --threads 32 --batch-size 64

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# --- environment -------------------------------------------------------------
# Adjust to your cluster: either module load, or activate the uv venv directly.
# module load cuda/12.4 python/3.12

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

# Compute nodes are often offline. Pre-fetch on the login node once:
#   uv run python -c "from detoxify import Detoxify; Detoxify('original-small')"
# then uncomment:
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

nvidia-smi || echo "no GPU visible - will fall back to CPU"

# --- run ---------------------------------------------------------------------
srun uv run python scripts/score_all.py \
    --batch-size 256 \
    --threads "${SLURM_CPUS_PER_TASK:-8}"

echo "done. results in data/fallacy/toxicity/*.jsonl"
