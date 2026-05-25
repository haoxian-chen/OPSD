#!/usr/bin/env bash
# Evaluate the Qwen3-1.7B base model with Qwen3 thinking mode enabled.
# This variant evaluates aime24, aime25, and hmmt25 by default.
#
# Usage:
#   bash eval/run_eval_1b_hkaift_base.sh
#
# Useful overrides:
#   BASE_MODEL=/path/or/hf-id
#   DATASETS="aime24 aime25 hmmt25"  VAL_N=12
#   DATASET=aime25  # backwards-compatible single-dataset override
#   THINKING=1  SKIP_EXISTING=1
#   CUDA_VISIBLE_DEVICES=0,1,2,3  TENSOR_PARALLEL_SIZE=4

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-${MODEL:-Qwen/Qwen3-1.7B}}"
DATASETS="${DATASETS:-${DATASET:-aime24 aime25 hmmt25}}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
VAL_N="${VAL_N:-12}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-42}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
WANDB_PROJECT="${WANDB_PROJECT-OPSD}"
WANDB_GROUP_OVERRIDE="${WANDB_GROUP:-}"

read -r -a RAW_DATASET_LIST <<< "${DATASETS//,/ }"

if (( ${#RAW_DATASET_LIST[@]} == 0 )); then
    echo "error: DATASETS did not contain any entries" >&2
    exit 1
fi

DATASET_LIST=()
for dataset in "${RAW_DATASET_LIST[@]}"; do
    case "$dataset" in
        hmmt)
            DATASET_LIST+=("hmmt25")
            ;;
        math500|amo-bench|aime24|aime25|hmmt25|minerva|amc23)
            DATASET_LIST+=("$dataset")
            ;;
        *)
            echo "error: unknown dataset '$dataset'" >&2
            echo "       Use one of: math500 amo-bench aime24 aime25 hmmt25 minerva amc23" >&2
            echo "       Note: 'hmmt' is accepted as an alias for 'hmmt25'." >&2
            exit 1
            ;;
    esac
done

MODE_ARGS=(--enable_thinking)
MODE_NAME="thinking"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
WANDB_TAGS="${WANDB_TAGS:-eval,1b,hkaift,base,thinking}"
if [[ "${THINKING:-1}" == "0" || "${THINKING:-1}" == "false" || "${THINKING:-1}" == "False" ]]; then
    MODE_ARGS=(--no_thinking)
    MODE_NAME="nonthinking"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-30000}"
    WANDB_TAGS="${WANDB_TAGS:-eval,1b,hkaift,base,nonthinking}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0" .sh)"
BASE_MODEL_NAME="$(basename "$BASE_MODEL")"
cd "$SCRIPT_DIR"
mkdir -p eval_results

result_file_for() {
    local dataset="$1"
    local filename="eval_results_${dataset}_${BASE_MODEL_NAME}"

    filename+="_${MODE_NAME}_temp${TEMPERATURE}_valn${VAL_N}.json"
    printf 'eval_results/%s' "$filename"
}

should_skip_result() {
    local output_file="$1"

    [[ "$SKIP_EXISTING" == "1" && -s "$output_file" ]]
}

echo "[${SCRIPT_NAME}] BASE_MODEL=$BASE_MODEL"
echo "[${SCRIPT_NAME}] mode=$MODE_NAME datasets=${DATASET_LIST[*]} val_n=$VAL_N"
echo "[${SCRIPT_NAME}] cuda_visible_devices=$CUDA_VISIBLE_DEVICES tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
echo "[${SCRIPT_NAME}] skip_existing=${SKIP_EXISTING}"
echo "[${SCRIPT_NAME}] WANDB_PROJECT=${WANDB_PROJECT:-disabled}"

for dataset in "${DATASET_LIST[@]}"; do
    wandb_group="${WANDB_GROUP_OVERRIDE:-eval_qwen31b_hkaift_base_all_${MODE_NAME}_valn${VAL_N}}"
    output_file="$(result_file_for "$dataset")"

    if should_skip_result "$output_file"; then
        echo "[${SCRIPT_NAME}] skipping dataset=$dataset base model; exists: $output_file"
        continue
    fi

    EVAL_ARGS=(
        --base_model "$BASE_MODEL"
        --dataset "$dataset"
        --val_n "$VAL_N"
        --temperature "$TEMPERATURE"
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --seed "$SEED"
        "${MODE_ARGS[@]}"
        --output_file "$output_file"
    )

    if [[ -n "${TOP_P:-}" ]]; then
        EVAL_ARGS+=(--top_p "$TOP_P")
    fi
    if [[ -n "${TOP_K:-}" ]]; then
        EVAL_ARGS+=(--top_k "$TOP_K")
    fi
    if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
        EVAL_ARGS+=(--max_model_len "$MAX_MODEL_LEN")
    fi
    if [[ -n "${NUM_SAMPLES:-}" ]]; then
        EVAL_ARGS+=(--num_samples "$NUM_SAMPLES")
    fi
    if [[ -n "${PRESENCE_PENALTY:-}" ]]; then
        EVAL_ARGS+=(--presence_penalty "$PRESENCE_PENALTY")
    fi
    if [[ -n "${WANDB_PROJECT:-}" ]]; then
        EVAL_ARGS+=(
            --wandb_project "$WANDB_PROJECT"
            --wandb_group "$wandb_group"
            --wandb_tags "$WANDB_TAGS"
            --wandb_step 0
            --wandb_divergence base
            --wandb_run_name "eval_qwen31b_hkaift_base_${dataset}_${MODE_NAME}_valn${VAL_N}"
        )
        if [[ -n "${WANDB_ENTITY:-}" ]]; then
            EVAL_ARGS+=(--wandb_entity "$WANDB_ENTITY")
        fi
    fi

    echo "[${SCRIPT_NAME}] evaluating dataset=$dataset base model output=$output_file"
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python evaluate_math.py \
        "${EVAL_ARGS[@]}"
done

echo "[${SCRIPT_NAME}] all base eval tasks finished"
