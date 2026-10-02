#!/bin/bash
#SBATCH --job-name=side-neutral
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode091
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

# Joint LED training on Ada with the same settings as the Kaggle side-neutral
# runs (kaggle/runner.py, MODE=full), then test evaluation of the best checkpoint.
# Submit from the repo root after building the splits (splits/build.sh):
#
#   sbatch scripts/ada/ada_train_side_neutral.sh                       # ideology negatives, lambda 0.1
#   LAMBDA=0.35 DATA=dataset/side_neutral_full_publisher/prepared \
#     RUN=ada-side_neutral_full_publisher-lambda035 sbatch scripts/ada/ada_train_side_neutral.sh
#   DRY_RUN=1 sbatch scripts/ada/ada_train_side_neutral.sh             # check env + data, no training
#
# W&B: export WANDB_API_KEY before sbatch, or keep WANDB_KEY in .env; without a
# key, or if wandb will not import, the run continues with --disable-wandb.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs/ada

LAMBDA="${LAMBDA:-0.1}"
DATA="${DATA:-dataset/side_neutral_full/prepared}"
RUN="${RUN:-ada-side_neutral_full-lambda${LAMBDA//./}}"
OUT="runs/ada/${RUN}_${SLURM_JOB_ID:-manual}"

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | lambda ${LAMBDA} | data ${DATA} | out ${OUT}"
nvidia-smi -L || true
[ -f "${DATA}/manifest.json" ] || { echo "missing ${DATA}; build it with splits/build.sh"; exit 1; }
[ -d .venv ] || uv sync   # first build ~30 min; must happen inside a job, never on the login node

if [ -z "${WANDB_API_KEY:-}" ] && [ -f .env ]; then
    WANDB_API_KEY="$(grep -E '^WANDB_KEY=' .env | cut -d= -f2- || true)"
fi
UV=(uv run --no-sync)
WANDB_ARGS=(--disable-wandb)
if [ -n "${WANDB_API_KEY:-}" ] && uv run --no-sync --with wandb python -c "import wandb" 2>/dev/null; then
    export WANDB_API_KEY
    UV=(uv run --no-sync --with wandb)
    WANDB_ARGS=(--wandb-entity manasagrawal206-iiit-hyderabad --wandb-project anlp-project --wandb-run-name "${RUN}")
    echo "wandb enabled: manasagrawal206-iiit-hyderabad/anlp-project/${RUN}"
else
    echo "wandb disabled (no key, or wandb failed to import)"
fi

"${UV[@]}" python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"

TRAIN=(python -u training/src/scripts/train_joint.py
    --data-dir "${DATA}" --output-dir "${OUT}/train"
    --model-name allenai/led-base-16384
    --train-batch-size 1 --eval-batch-size 4 --gradient-accumulation-steps 8
    --max-epochs 4 --early-stopping-patience 2
    --learning-rate 3e-5 --weight-decay 0.01 --warmup-ratio 0.10 --max-grad-norm 1.0
    --max-input-length 512 --max-target-length 512 --generation-max-new-tokens 512
    --triplet-lambda "${LAMBDA}" --triplet-margin 0.2 --attention-window 512
    --seed 42 --device cuda --precision fp32 --overwrite-output-dir
    "${WANDB_ARGS[@]}")

if [ -n "${DRY_RUN:-}" ]; then
    "${UV[@]}" "${TRAIN[@]}" --dry-run --disable-wandb | tail -5
    exit 0
fi

"${UV[@]}" "${TRAIN[@]}"

"${UV[@]}" python -u training/src/scripts/evaluate_joint.py \
    --data-dir "${DATA}" --checkpoint "${OUT}/train/best" --output-dir "${OUT}/test_eval" \
    --split test --eval-batch-size 4 --device cuda --overwrite-output-dir

# Keep the best model and metrics; drop optimizer state (~1.3 GB each) and the
# last-epoch model so the run fits the 25 GB home quota.
rm -rf "${OUT}/train/last"
find "${OUT}" -name trainer_state.pt -delete
du -sh "${OUT}"
echo "done: ${OUT}"
