#!/bin/bash
#SBATCH --job-name=hard-subset
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode091
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --output=logs/hard-%j.out

# Score every original article of the dedup and full1024 releases with the POLITICS v2
# ensemble, then write hard_{left,right}_neutral.jsonl. Needs on Ada: the venv (ada_sync.sh),
# {left,right}_neutral_{dedup,full1024}.jsonl at the repo root, and
# data/classifiers/politics-ideology-classifier/v2/. Resumable: resubmit if it times out.
#
#   sbatch scripts/ada_hard_subset.sh            # TOP_FRAC=0.2 by default
#   TOP_FRAC=0.1 sbatch scripts/ada_hard_subset.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs/hard

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd scripts
uv run --no-sync python make_hard_subset.py score
uv run --no-sync python make_hard_subset.py select --top-frac "${TOP_FRAC:-0.2}"
