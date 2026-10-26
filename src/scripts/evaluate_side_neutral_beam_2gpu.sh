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
#SBATCH --output=side_neutral_beam_evaluation.logs
#SBATCH --mail-type=END
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in
#SBATCH --job-name=side_neutral_beam_evaluation

set -eo pipefail
PROJECT_ROOT=/home2/manas.agrawal/anlp/ANLP-Project
DATA_DIR=${ANLP_DATA_DIR:-${PROJECT_ROOT}/dataset/side_neutral/prepared}
CHECKPOINT_TYPE=${ANLP_CHECKPOINT_TYPE:-triplet}
CHECKPOINT=${ANLP_CHECKPOINT:-${PROJECT_ROOT}/outputs/side_neutral_first_token_lambda035_21275/best}
ANLP_RUN_ID=${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S-%N)}
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/side_neutral_beam_${CHECKPOINT_TYPE}_${ANLP_RUN_ID}

COMMON_COMMAND=(
    python -u "${PROJECT_ROOT}/src/scripts/evaluate_decoding.py"
    --data-dir "${DATA_DIR}" --checkpoint "${CHECKPOINT}"
    --checkpoint-type "${CHECKPOINT_TYPE}" --decoding-strategy beam
    --eval-batch-size 1 --max-input-length 512 --max-target-length 512
    --generation-max-new-tokens 512 --attention-window 512 --seed 42 --device cuda
    --wandb-entity manasagrawal206-iiit-hyderabad
    --wandb-project anlp-project --require-wandb
)

if [ "${1:-}" = "--dry-run" ]; then
    shift
    status=0
    echo "DRY RUN ONLY — no Python evaluation, W&B, or scheduler command is started"
    echo "data_dir=${DATA_DIR} checkpoint_type=${CHECKPOINT_TYPE} checkpoint=${CHECKPOINT}"
    echo "output_root=${OUTPUT_ROOT} decoding=beam num_beams=4 do_sample=false"
    echo "validation and test run on separate GPUs; batch=1 input=512 target=512 max_new_tokens=512 attention_window=512 seed=42"
    echo "wandb=required-online gpus=2 nodes=1 log=side_neutral_beam_evaluation.logs"
    if [ "${CHECKPOINT_TYPE}" != baseline ] && [ "${CHECKPOINT_TYPE}" != triplet ]; then
        echo "ERROR: ANLP_CHECKPOINT_TYPE must be baseline or triplet" >&2
        status=1
    fi
    if [ ! -f "${CHECKPOINT}/model.safetensors" ]; then
        echo "MISSING CHECKPOINT: ${CHECKPOINT}/model.safetensors" >&2
        status=1
    fi
    for split in validation test; do
        if [ ! -f "${DATA_DIR}/${split}/baseline.jsonl" ]; then
            echo "MISSING DATA: ${DATA_DIR}/${split}/baseline.jsonl" >&2
            status=1
        fi
        command=("${COMMON_COMMAND[@]}" --split "${split}" --output-dir "${OUTPUT_ROOT}/${split}" --wandb-run-name "side-neutral-${CHECKPOINT_TYPE}-beam-${split}-${ANLP_RUN_ID}" "$@")
        printf '%s command:' "${split}"; printf ' %q' "${command[@]}"; printf '\n'
    done
    exit "${status}"
fi

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-/scratch/${USER}/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online
cd "${PROJECT_ROOT}"
source /home2/manas.agrawal/miniforge3/etc/profile.d/conda.sh
conda activate torch_env
mkdir -p "${OUTPUT_ROOT}"

CUDA_VISIBLE_DEVICES=0 "${COMMON_COMMAND[@]}" --split validation --output-dir "${OUTPUT_ROOT}/validation" --wandb-run-name "side-neutral-${CHECKPOINT_TYPE}-beam-validation-${ANLP_RUN_ID}" "$@" &
validation_pid=$!
CUDA_VISIBLE_DEVICES=1 "${COMMON_COMMAND[@]}" --split test --output-dir "${OUTPUT_ROOT}/test" --wandb-run-name "side-neutral-${CHECKPOINT_TYPE}-beam-test-${ANLP_RUN_ID}" "$@" &
test_pid=$!
status=0
wait "${validation_pid}" || status=1
wait "${test_pid}" || status=1
exit "${status}"
