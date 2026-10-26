#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --exclude=gnode[001-012,016-017,019-023,025-035,037-038,040-041,066-067]
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=side_neutral_baseline_%j.logs
#SBATCH --job-name=side_neutral_baseline

set -eo pipefail
PROJECT_ROOT=/home2/manas.agrawal/anlp/ANLP-Project
cd "${PROJECT_ROOT}"
ANLP_OUTPUT_ROOT=${PROJECT_ROOT}/outputs
ANLP_RUN_ID=${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S-%N)}
RUN_OUTPUT_DIR="${ANLP_OUTPUT_ROOT}/side_neutral_baseline_${ANLP_RUN_ID}"
echo "Default run artifacts: $(hostname):${RUN_OUTPUT_DIR}"
source /home2/manas.agrawal/miniforge3/etc/profile.d/conda.sh
conda activate torch_env
export HF_HOME=${HF_HOME:-/scratch/${USER}/hf_cache}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export TOKENIZERS_PARALLELISM=false

exec python -u src/scripts/train_baseline.py \
    --data-dir dataset/side_neutral/prepared \
    --output-dir "${RUN_OUTPUT_DIR}" \
    --model-name allenai/led-base-16384 \
    --train-batch-size 1 --eval-batch-size 1 \
    --gradient-accumulation-steps 8 --max-epochs 10 \
    --learning-rate 3e-5 --weight-decay 0.01 --warmup-ratio 0.10 \
    --max-grad-norm 1.0 --early-stopping-patience 3 \
    --max-input-length 512 --max-target-length 512 \
    --generation-max-new-tokens 512 \
    --seed 42 --num-workers 0 --device cuda --precision fp32 \
    --gradient-checkpointing "$@"
