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
#SBATCH --output=side_neutral_first_token_margin04_lambda035.logs
#SBATCH --mail-type=END
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in
#SBATCH --job-name=side_neutral_first_token_margin04_lambda035

set -eo pipefail
# Training already logs d_ap, d_an, and generation quality. Held-out retrieval
# must be recomputed from this run's best checkpoint; do not reuse old vectors.
PROJECT_ROOT=/home2/manas.agrawal/anlp/ANLP-Project
DATA_DIR=${ANLP_DATA_DIR:-${PROJECT_ROOT}/dataset/side_neutral/prepared}
ANLP_RUN_ID=${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S-%N)}
RUN_OUTPUT_DIR="${PROJECT_ROOT}/outputs/side_neutral_first_token_margin04_lambda035_${ANLP_RUN_ID}"

TRAIN_COMMAND=(
    torchrun --standalone --nnodes=1 --nproc-per-node=2
    "${PROJECT_ROOT}/src/scripts/train_joint.py"
    --data-dir "${DATA_DIR}" --output-dir "${RUN_OUTPUT_DIR}"
    --model-name allenai/led-base-16384
    --train-batch-size 1 --eval-batch-size 1
    --gradient-accumulation-steps 4 --max-epochs 10
    --learning-rate 3e-5 --weight-decay 0.01 --warmup-ratio 0.10
    --max-grad-norm 1.0 --early-stopping-patience 3 --early-stopping-min-delta 0.0
    --max-input-length 512 --max-target-length 512 --generation-max-new-tokens 512
    --triplet-lambda 0.35 --triplet-margin 0.4 --attention-window 512
    --triplet-objective standard --positive-beta 0.0 --pooling first_token
    --seed 42 --num-workers 0 --device cuda --precision fp32
    --wandb-entity manasagrawal206-iiit-hyderabad
    --wandb-project anlp-project --wandb-run-name side-neutral-first-token-margin04-lambda035
)

# Test-only switch; submitted jobs require online W&B logging.
if [ "${ANLP_DISABLE_WANDB_FOR_TEST:-0}" = 1 ]; then
    TRAIN_COMMAND+=(--disable-wandb)
    WANDB_SETTING=disabled-for-test
else
    TRAIN_COMMAND+=(--require-wandb)
    WANDB_SETTING=required-online
fi

RETRIEVAL_COMMAND=(
    python -u "${PROJECT_ROOT}/src/scripts/evaluate_side_neutral_retrieval.py"
    --data-dir "${DATA_DIR}"
    --checkpoint "${RUN_OUTPUT_DIR}/best"
    --output "${RUN_OUTPUT_DIR}/retrieval_test.json"
    --device cuda --batch-size 2
    --wandb-entity manasagrawal206-iiit-hyderabad
    --wandb-project anlp-project
    --wandb-run-name "side-neutral-first-token-margin04-retrieval-${ANLP_RUN_ID}"
    --require-wandb
)

if [ "${1:-}" = "--dry-run" ]; then
    shift
    status=0
    echo "DRY RUN ONLY — torchrun, training, and W&B are not started"
    echo "data_dir=${DATA_DIR} output_dir=${RUN_OUTPUT_DIR}"
    echo "pooling=first_token objective=standard lambda=0.35 margin=0.4 beta=0.0 attraction=off"
    echo "decoding=greedy num_beams=1 do_sample=false"
    echo "model=allenai/led-base-16384 seed=42 precision=fp32"
    echo "gpus=2 nodes=1 per_gpu_microbatch=1 per_gpu_accumulation=4 global_effective_batch=8"
    echo "epochs=10 lr=3e-5 weight_decay=0.01 warmup_ratio=0.10 max_grad_norm=1.0 patience=3"
    echo "input_limit=512 target_limit=512 generation_limit=512 attention_window=512"
    echo "wandb=${WANDB_SETTING} log=side_neutral_first_token_margin04_lambda035.logs"
    for split in train validation test; do
        for kind in baseline triplets; do
            path="${DATA_DIR}/${split}/${kind}.jsonl"
            if [ -f "${path}" ]; then
                echo "${split}_${kind}=${path}"
            else
                echo "MISSING DATA: ${path}"
                status=1
            fi
        done
    done
    if [ -f "${DATA_DIR}/train/triplets.jsonl" ]; then
        train_examples=$(wc -l < "${DATA_DIR}/train/triplets.jsonl")
        updates_per_epoch=$(( (train_examples + 7) / 8 ))
        total_updates=$(( updates_per_epoch * 10 ))
        echo "train_examples=${train_examples} optimizer_updates_per_epoch=${updates_per_epoch} planned_optimizer_updates=${total_updates} planned_warmup_updates=$(( (total_updates + 5) / 10 ))"
    fi
    printf 'training command:'; printf ' %q' "${TRAIN_COMMAND[@]}" "$@"; printf '\n'
    if [ "${ANLP_DISABLE_WANDB_FOR_TEST:-0}" = 1 ]; then
        echo "post-training retrieval skipped in test mode"
    else
        printf 'post-training retrieval command:'; printf ' %q' "${RETRIEVAL_COMMAND[@]}"; printf '\n'
    fi
    exit "${status}"
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
python -c 'from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)' || {
    echo "ERROR: No Hugging Face login or HF_TOKEN is available to this job." >&2
    exit 1
}
"${TRAIN_COMMAND[@]}" "$@"
if [ "${ANLP_DISABLE_WANDB_FOR_TEST:-0}" = 1 ]; then
    echo "Test mode: skipping post-training retrieval evaluation"
    exit 0
fi

# This reads the selected best checkpoint; it does not alter training or selection.
"${RETRIEVAL_COMMAND[@]}"
