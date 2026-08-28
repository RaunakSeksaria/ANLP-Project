#!/bin/bash
#SBATCH --job-name=neus-triplet
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode091
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --output=logs/neus-%j.out
#SBATCH --error=logs/neus-%j.err

# The whole experiment in one job:
#   1. baseline   generation loss only
#   2. triplet    generation + event-aware representation loss
#   3. probes     publisher leakage on both, plus the TF-IDF lexical floor
#   4. metrics    ROUGE and framing bias on the generations
#
#   sbatch scripts/ada_run.sh
#
# Constraints this script is built around, from temp/06-ada-handoff.md:
#   * RTX 2080 Ti = 10.6 GiB usable. BART-large + a triplet stream does not fit,
#     so the default model is bart-base. Try bart-large only after this works.
#   * Turing (sm_75) has no bf16. fp16 or nothing.
#   * MaxMemPerCPU is 3000 MB -- `--mem` is rejected outright, always
#     --mem-per-cpu.
#   * QOS medium caps 4 running jobs / 4 GPUs / 40 CPUs / 8 submitted.
#   * --constraint=2080ti is required: feature-less nodes are GTX 1080 Ti
#     (Pascal sm_61), which current torch wheels no longer compile for.
#   * gnode091 has 11 GB of host RAM instead of 128 GB.
#   * --no-sync below is deliberate. Never let a run job touch the resolver;
#     `uv sync` has its own job (see the header of this file's companion
#     commands in docs/ada-compute-requirements.md).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p logs runs "$HF_HOME"

# Compute nodes DO have outbound network, so the model downloads here rather
# than on the login node -- which cannot import torch at all (CentOS 7,
# glibc 2.17, no CXXABI_1.3.9).

MODEL="${MODEL:-facebook/bart-base}"
EPOCHS="${EPOCHS:-5}"

nvidia-smi
uv run --no-sync python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

echo "=== 1/4 baseline (generation loss only) ==="
uv run --no-sync python scripts/train_neus.py \
    --run-name baseline --model "$MODEL" --epochs "$EPOCHS" \
    --triplet-weight 0 --precision fp16

echo "=== 2/4 triplet (generation + event-aware loss) ==="
uv run --no-sync python scripts/train_neus.py \
    --run-name triplet --model "$MODEL" --epochs "$EPOCHS" \
    --triplet-weight 1.0 --triplet-margin 1.0 --precision fp16

echo "=== 3/4 publisher-leakage probes ==="
uv run --no-sync python scripts/probe.py --features tfidf   --tag tfidf
uv run --no-sync python scripts/probe.py --features vectors --tag baseline \
    --vectors runs/baseline/vectors.npz
uv run --no-sync python scripts/probe.py --features vectors --tag triplet \
    --vectors runs/triplet/vectors.npz

echo "=== 4/4 generation metrics ==="
uv run --no-sync python scripts/eval_generations.py

echo
echo "done. results:"
echo "  runs/{baseline,triplet}/test_generations.txt"
echo "  data/prepared/probe_results.json"
echo "  runs/generation_metrics.json"
