#!/bin/bash
#SBATCH --job-name=neus-triplet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/neus-%j.out
#SBATCH --error=logs/neus-%j.err

# Both runs of the experiment, back to back, in one job:
#   baseline  generation loss only          (~30-45 min)
#   triplet   generation + event-aware loss (~60-90 min)
#
# Then the publisher-leakage probe on both, which is what the comparison is for.
#
#   sbatch scripts/ada_run.sh
#
# Check the SBATCH lines against your own allocation before submitting --
# partition and account names differ per Ada project.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

source .venv-ada/bin/activate
mkdir -p logs runs

nvidia-smi || echo "no GPU visible"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- 1. baseline: generation loss only ----------------------------------------
echo "=== baseline ==="
python scripts/train_neus.py \
    --run-name baseline \
    --triplet-weight 0 \
    --epochs 5

# --- 2. triplet: generation + event-aware representation loss ------------------
echo "=== triplet ==="
python scripts/train_neus.py \
    --run-name triplet \
    --triplet-weight 1.0 \
    --triplet-margin 1.0 \
    --epochs 5

# --- 3. publisher leakage on both, plus the lexical floor ----------------------
echo "=== probes ==="
python scripts/probe.py --features tfidf   --tag tfidf
python scripts/probe.py --features vectors --tag baseline --vectors runs/baseline/vectors.npz
python scripts/probe.py --features vectors --tag triplet  --vectors runs/triplet/vectors.npz

# --- 4. summary metrics on the generations -------------------------------------
echo "=== generation metrics ==="
python scripts/eval_generations.py

echo
echo "done. results:"
echo "  runs/{baseline,triplet}/test_generations.txt"
echo "  data/prepared/probe_results.json"
echo "  runs/generation_metrics.json"
