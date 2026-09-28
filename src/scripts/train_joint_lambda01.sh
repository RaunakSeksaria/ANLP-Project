#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --exclude=gnode[001-012,016-017,019-023,025-035,037-038,040-041]
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=triple_loss_batch.logs
#SBATCH --mail-type=END
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in
#SBATCH --job-name=anlp_lambda01

# Conda's non-interactive activation hook can reference shell variables that
# are unset in Slurm jobs, so do not enable Bash nounset (`set -u`) here.
set -eo pipefail

export HF_HOME=/scratch/${USER}/hf_cache
export TRANSFORMERS_CACHE=/scratch/${USER}/hf_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT=/home2/manas.agrawal/anlp/ANLP-Project
if [ ! -f "${PROJECT_ROOT}/src/scripts/train_joint.py" ]; then
    echo "ERROR: training script not found at ${PROJECT_ROOT}/src/scripts/train_joint.py"
    exit 1
fi
cd "${PROJECT_ROOT}"

echo "========================================================"
echo "BEFORE ENVIRONMENT ACTIVATION: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Slurm job ID: ${SLURM_JOB_ID:-not-set}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Allocated GPUs: ${SLURM_GPUS_ON_NODE:-unknown}"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-not-active}"
echo "Python before activation: $(command -v python || true)"
echo "Running environment command: conda activate torch_env"

source /home2/manas.agrawal/miniforge3/etc/profile.d/conda.sh
conda activate torch_env

echo "AFTER ENVIRONMENT ACTIVATION: $(date)"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-not-active}"
echo "Python after activation: $(command -v python)"
python --version
python -c 'import torch, transformers; print("PyTorch:", torch.__version__); print("Transformers:", transformers.__version__); print("CUDA available:", torch.cuda.is_available()); print("Visible GPU count:", torch.cuda.device_count())'
nvidia-smi -L || true
echo "========================================================"

TRAIN_COMMAND=(
    python -u "${PROJECT_ROOT}/src/scripts/train_joint.py"
    --data-dir "${PROJECT_ROOT}/dataset/neutral_articles_10k_prepared"
    --output-dir "${PROJECT_ROOT}/outputs/joint_mean_lambda01"
    --model-name allenai/led-base-16384
    --train-batch-size 1
    --eval-batch-size 1
    --gradient-accumulation-steps 8
    --max-epochs 10
    --learning-rate 3e-5
    --weight-decay 0.01
    --warmup-ratio 0.10
    --max-grad-norm 1.0
    --early-stopping-patience 3
    --early-stopping-min-delta 0.0
    --max-input-length 512
    --max-target-length 512
    --generation-max-new-tokens 512
    --triplet-lambda 0.1
    --triplet-margin 0.2
    --attention-window 512
    --seed 42
    --num-workers 0
    --device cuda
    --precision fp32
    --wandb-entity manasagrawal206-iiit-hyderabad
    --wandb-project anlp-project
    --wandb-run-name joint-mean-lambda01
    --overwrite-output-dir
)

echo "BEFORE TRAINING COMMAND: $(date)"
printf 'Running command:'
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'

if "${TRAIN_COMMAND[@]}"; then
    TRAIN_STATUS=0
else
    TRAIN_STATUS=$?
fi

echo "AFTER TRAINING COMMAND: $(date)"
echo "Training exit status: ${TRAIN_STATUS}"
if [ "${TRAIN_STATUS}" -eq 0 ]; then
    echo "Lambda-0.1 joint training completed successfully."
else
    echo "Lambda-0.1 joint training failed; inspect triple_loss_batch.logs."
fi
exit "${TRAIN_STATUS}"
