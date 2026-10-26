#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --qos=medium
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=2-00:00:00
#SBATCH --job-name=politics-v2
#SBATCH --output=logs/%x-%j.out
# One POLITICS v2 seed. SEED required; EXTRA passes extra args (e.g. --limit 300 --epochs 1 for a smoke run).
#   sbatch --export=ALL,SEED=42 train_seed.sbatch
set -euo pipefail
cd ~/politics_v2
export HF_HOME=/scratch/$USER/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"
PY=~/ANLP/A2/.venv/bin/python
echo "node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | seed ${SEED:?}"
$PY -u train_v2.py --articles data/baly_articles.csv --splits data/splits/random \
  --seed "$SEED" --out runs/seed${SEED} --half-checkpoints ${EXTRA:-}
