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
#SBATCH --output=side_neutral_lambda01_attraction_%j.logs
#SBATCH --mail-type=END
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in
#SBATCH --job-name=side_neutral_lambda01_attraction

set -eo pipefail
PROJECT_ROOT=/home2/manas.agrawal/anlp/ANLP-Project
DATA_DIR=${ANLP_DATA_DIR:-${PROJECT_ROOT}/dataset/side_neutral/prepared}
ANLP_OUTPUT_ROOT=${PROJECT_ROOT}/outputs
ANLP_RUN_ID=${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S-%N)}
RUN_OUTPUT_DIR="${ANLP_OUTPUT_ROOT}/side_neutral_lambda01_attraction_${ANLP_RUN_ID}"

TRAIN_COMMAND=(
    python -u "${PROJECT_ROOT}/src/scripts/train_joint.py"
    --data-dir "${DATA_DIR}"
    --output-dir "${RUN_OUTPUT_DIR}"
    --model-name allenai/led-base-16384
    --train-batch-size 1 --eval-batch-size 1
    --gradient-accumulation-steps 8 --max-epochs 10
    --learning-rate 3e-5 --weight-decay 0.01 --warmup-ratio 0.10
    --max-grad-norm 1.0 --early-stopping-patience 3 --early-stopping-min-delta 0.0
    --max-input-length 512 --max-target-length 512 --generation-max-new-tokens 512
    --triplet-lambda 0.1 --triplet-margin 0.2 --attention-window 512
    --triplet-objective attraction --positive-distance-threshold 0.5 --positive-beta 0.3
    --seed 42 --num-workers 0 --device cuda --precision fp32
    --wandb-entity manasagrawal206-iiit-hyderabad
    --wandb-project anlp-project --wandb-run-name side-neutral-lambda01-attraction
    --require-wandb
)

# This branch must never load a model, initialize CUDA, or touch W&B.
if [ "${1:-}" = "--dry-run" ]; then
    shift
    data_status=0
    echo "DRY RUN ONLY — no model, optimizer, scheduler, or job command will run"
    echo "data_dir=${DATA_DIR}"
    echo "output_dir=${RUN_OUTPUT_DIR}"
    echo "objective=attraction alpha=0.2 tau=0.5 beta=0.3 schedule=constant lambda=0.1"
    echo "wandb=required mode=online project=anlp-project entity=manasagrawal206-iiit-hyderabad run=side-neutral-lambda01-attraction"
    echo "model=allenai/led-base-16384 batch=1 accumulation=8 epochs=10 lr=3e-5 weight_decay=0.01 warmup_ratio=0.10 max_grad_norm=1.0 patience=3 seed=42 precision=fp32"
    echo "input_limit=512 target_limit=512 generation_limit=512 attention_window=512"
    for split in train validation test; do
        for kind in baseline triplets; do
            path="${DATA_DIR}/${split}/${kind}.jsonl"
            if [ -f "${path}" ]; then
                echo "${split}_${kind}=${path}"
            else
                echo "MISSING DATA: ${path}"
                data_status=1
            fi
        done
    done
    if [ -f "${DATA_DIR}/train/triplets.jsonl" ]; then
        train_examples=$(wc -l < "${DATA_DIR}/train/triplets.jsonl")
        updates_per_epoch=$(( (train_examples + 7) / 8 ))
        echo "train_examples=${train_examples} planned_optimizer_updates=$(( updates_per_epoch * 10 ))"
        echo "planned_warmup_updates=$(( (updates_per_epoch * 10 + 5) / 10 )) beta_at_all_updates=0.3"
    else
        echo "planned_optimizer_updates=unavailable until the intended dataset is restored"
        echo "planned_warmup_updates=unavailable beta_at_all_updates=0.3"
    fi
    printf 'training command:'; printf ' %q' "${TRAIN_COMMAND[@]}" "$@"; printf '\n'
    exit "${data_status}"
fi

export HF_HOME=${HF_HOME:-/scratch/${USER}/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online
cd "${PROJECT_ROOT}"
source /home2/manas.agrawal/miniforge3/etc/profile.d/conda.sh
conda activate torch_env
"${TRAIN_COMMAND[@]}" "$@"
