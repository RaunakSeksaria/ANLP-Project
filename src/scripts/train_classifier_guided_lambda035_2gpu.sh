#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=classifier_guided_lambda035.logs
#SBATCH --mail-type=END
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in
#SBATCH --job-name=classifier_guided_lambda035

set -eo pipefail
PROJECT_ROOT=/home2/manas.agrawal/anlp/ANLP-Project
DATA_DIR=${ANLP_DATA_DIR:-${PROJECT_ROOT}/dataset/side_neutral/prepared}
CLASSIFIER_CHECKPOINT=${ANLP_CLASSIFIER_CHECKPOINT:-${PROJECT_ROOT}/external/politics_ideology_classifier/politics_media.pt}
ANLP_RUN_ID=${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S-%N)}
RUN_OUTPUT_DIR="${PROJECT_ROOT}/outputs/classifier_guided_lambda035_${ANLP_RUN_ID}"

TRAIN_COMMAND=(
    torchrun --standalone --nnodes=1 --nproc-per-node=2
    "${PROJECT_ROOT}/src/scripts/train_classifier_guided.py"
    --data-dir "${DATA_DIR}" --output-dir "${RUN_OUTPUT_DIR}"
    --classifier-checkpoint "${CLASSIFIER_CHECKPOINT}"
    --model-name allenai/led-base-16384
    --train-batch-size 1 --eval-batch-size 1
    --gradient-accumulation-steps 4 --max-epochs 10
    --learning-rate 3e-5 --weight-decay 0.01 --warmup-ratio 0.10
    --max-grad-norm 1.0 --early-stopping-patience 3
    --max-input-length 512 --generation-max-new-tokens 512 --attention-window 512
    --triplet-lambda 0.35
    --seed 42 --num-workers 0
    --wandb-entity manasagrawal206-iiit-hyderabad
    --wandb-project anlp-project --wandb-run-name classifier-guided-lambda035
)

if [ "${ANLP_DISABLE_WANDB_FOR_TEST:-0}" = 1 ]; then
    TRAIN_COMMAND+=(--disable-wandb)
    WANDB_SETTING=disabled-for-test
else
    TRAIN_COMMAND+=(--require-wandb)
    WANDB_SETTING=required-online
fi

if [ "${1:-}" = "--dry-run" ]; then
    shift
    echo "DRY RUN ONLY — no torchrun, training, W&B, or scheduler execution"
    echo "data_dir=${DATA_DIR} classifier_checkpoint=${CLASSIFIER_CHECKPOINT} output_dir=${RUN_OUTPUT_DIR}"
    echo "pretrained=allenai/led-base-16384 pooling=first_token lambda=0.35 margin=0.2 beta=0.3 tau=0.5"
    echo "reward=Centre_probability policy=self_critical_sample_minus_greedy classifier=launch/POLITICS body_only=true classifier_limit=512"
    echo "source_limit=512 generation_limit=512 attention_window=512 seed=42 precision=fp32"
    echo "gpus=2 nodes=1 per_gpu_microbatch=1 accumulation=4 effective_global_batch=8"
    echo "epochs=10 lr=3e-5 weight_decay=0.01 warmup_ratio=0.10 max_grad_norm=1.0 patience=3"
    echo "wandb=${WANDB_SETTING} log=classifier_guided_lambda035.logs"
    if [ -f "${DATA_DIR}/train/triplets.jsonl" ]; then
        train_examples=$(wc -l < "${DATA_DIR}/train/triplets.jsonl")
        updates_per_epoch=$(( (train_examples + 7) / 8 ))
        planned_updates=$(( updates_per_epoch * 10 ))
        echo "train_examples=${train_examples} optimizer_updates_per_epoch=${updates_per_epoch} planned_updates=${planned_updates} warmup_updates=$(( (planned_updates + 5) / 10 ))"
    fi
    for split in train validation test; do
        path="${DATA_DIR}/${split}/triplets.jsonl"
        [ -f "${path}" ] && echo "${split}_sources=${path}" || echo "MISSING DATA: ${path}"
    done
    [ -f "${CLASSIFIER_CHECKPOINT}" ] && echo "classifier_status=present" || echo "classifier_status=MISSING (required before submission)"
    printf 'training command:'; printf ' %q' "${TRAIN_COMMAND[@]}" "$@"; printf '\n'
    exit 0
fi

if [ ! -f "${CLASSIFIER_CHECKPOINT}" ]; then
    echo "ERROR: Required politics_media.pt is missing: ${CLASSIFIER_CHECKPOINT}" >&2
    exit 1
fi
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-/scratch/${USER}/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export TOKENIZERS_PARALLELISM=false
if [ "${ANLP_DISABLE_WANDB_FOR_TEST:-0}" = 1 ]; then
    export WANDB_MODE=disabled
else
    export WANDB_MODE=online
fi
cd "${PROJECT_ROOT}"
source /home2/manas.agrawal/miniforge3/etc/profile.d/conda.sh
conda activate torch_env
"${TRAIN_COMMAND[@]}" "$@"
