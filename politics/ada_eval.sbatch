#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --qos=medium
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=03:00:00
#SBATCH --job-name=politics-v2-eval
#SBATCH --output=logs/%x-%j.out
# After all seeds finish: calibration + ensemble report, then score stories_10k with the seed ensemble.
set -euo pipefail
cd ~/politics_v2
export HF_HOME=/scratch/$USER/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false; mkdir -p "$HF_HOME"
PY=~/ANLP/A2/.venv/bin/python
$PY aggregate_v2.py --runs runs --out v2_results
$PY score_v2.py --weights runs/seed*/seed*_best.pt --calibration v2_results/calibration.json \
    --stories data/stories_10k.jsonl --out v2_results
