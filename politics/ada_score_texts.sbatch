#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --qos=medium
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=0-06:00:00
#SBATCH --job-name=politics-score
#SBATCH --output=logs/%x-%j.out
# sbatch --export=ALL,TEXTS=data/e2long_texts.jsonl,OUT=v2_results/e2long_scores.csv ada_score_texts.sbatch
set -euo pipefail
cd ~/politics_v2
export HF_HOME=/scratch/$USER/hf; mkdir -p $HF_HOME
~/ANLP/A2/.venv/bin/python -u score_texts.py --weights runs/seed4{2,3,4,5}/seed4?_best.pt \
    --calibration v2_results/calibration.json --texts "$TEXTS" --out "$OUT"
