#!/usr/bin/env bash
# Evaluate Qwen3-8B Tinker runs for all supported f-divergences with Qwen3
# thinking mode enabled. Step 0 is evaluated once as the shared base model;
# positive steps are evaluated per divergence.
#
# Usage:
#   bash eval/run_eval_8b_tinker_all_0_300_thinking.sh
#
# Defaults mirror scripts/run_opsd_8b_tinker.sh:
#   OUT=/data0/siyanz/opsd
#   RUN_CONFIG=qwen38b_tinker_<divergence_type>
#
# Useful overrides:
#   BASE_MODEL=/path/or/hf-id  OUT=/path/to/output
#   DIVERGENCES="reverse_kl jsd"
#   STEP_LIST=0,25,50,75,100,125,150,175,200,225,250,275,300
#   DATASETS="aime24 aime25 hmmt25"  VAL_N=12
#   DATASET=aime24  # backwards-compatible single-dataset override
#   SKIP_EXISTING=1  # skip evals whose result JSON already exists
#   CUDA_VISIBLE_DEVICES=0,1,2,3  TENSOR_PARALLEL_SIZE=4

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-${MODEL:-/data0/shared/Qwen3-8B}}"
OUT="${OUT:-/data0/siyanz/opsd}"
OUT="${OUT%/}"
DIVERGENCES="${DIVERGENCES:-reverse_kl forward_kl jsd improved_forward_kl improved_jsd}"
STEP_LIST="${STEP_LIST:-0,25,50,75,100,125,150,175,200,225,250,275,300}"
MAX_ALLOWED_STEP="${MAX_ALLOWED_STEP:-300}"
DATASETS="${DATASETS:-${DATASET:-aime24 aime25 hmmt25}}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
VAL_N="${VAL_N:-12}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
WANDB_PROJECT="${WANDB_PROJECT-OPSD}"
WANDB_GROUP_OVERRIDE="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-eval,8b,tinker,thinking}"

read -r -a DIVERGENCE_LIST <<< "$DIVERGENCES"
read -r -a DATASET_LIST <<< "${DATASETS//,/ }"
read -r -a STEPS <<< "${STEP_LIST//,/ }"

if (( ${#DIVERGENCE_LIST[@]} == 0 )); then
    echo "error: DIVERGENCES did not contain any entries" >&2
    exit 1
fi
if (( ${#DATASET_LIST[@]} == 0 )); then
    echo "error: DATASETS did not contain any entries" >&2
    exit 1
fi
if (( ${#STEPS[@]} == 0 )); then
    echo "error: STEP_LIST did not contain any checkpoint steps" >&2
    exit 1
fi

for div in "${DIVERGENCE_LIST[@]}"; do
    case "$div" in
        reverse_kl|forward_kl|jsd|improved_forward_kl|improved_jsd) ;;
        *)
            echo "error: unknown divergence_type '$div'" >&2
            echo "       must be one of: reverse_kl forward_kl jsd improved_forward_kl improved_jsd" >&2
            exit 1
            ;;
    esac
done

for dataset in "${DATASET_LIST[@]}"; do
    case "$dataset" in
        math500|amo-bench|aime24|aime25|hmmt25|minerva|amc23) ;;
        *)
            echo "error: unknown dataset '$dataset'" >&2
            echo "       must be one of: math500 amo-bench aime24 aime25 hmmt25 minerva amc23" >&2
            exit 1
            ;;
    esac
done

for step in "${STEPS[@]}"; do
    if ! [[ "$step" =~ ^[0-9]+$ ]]; then
        echo "error: checkpoint step must be numeric, got '$step'" >&2
        exit 1
    fi
    if (( step > MAX_ALLOWED_STEP )); then
        echo "error: STEP_LIST contains checkpoint-$step, but this script is capped at $MAX_ALLOWED_STEP" >&2
        echo "       Current STEP_LIST=$STEP_LIST" >&2
        echo "       If this was intentional, rerun with MAX_ALLOWED_STEP=$step or use a broader eval script." >&2
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0" .sh)"
BASE_MODEL_NAME="$(basename "$BASE_MODEL")"
cd "$SCRIPT_DIR"

result_file_for() {
    local dataset="$1"
    local run_config="${2:-}"
    local checkpoint_name="${3:-}"
    local filename="eval_results_${dataset}_${BASE_MODEL_NAME}"

    if [[ -n "$run_config" && -n "$checkpoint_name" ]]; then
        filename+="_${run_config}_${checkpoint_name}"
    fi

    filename+="_thinking_temp${TEMPERATURE}_valn${VAL_N}.json"
    printf 'eval_results/%s' "$filename"
}

should_skip_result() {
    local output_file="$1"

    [[ "$SKIP_EXISTING" == "1" && -s "$output_file" ]]
}

echo "[${SCRIPT_NAME}] BASE_MODEL=$BASE_MODEL"
echo "[${SCRIPT_NAME}] OUT=$OUT"
echo "[${SCRIPT_NAME}] mode=thinking datasets=${DATASETS} val_n=$VAL_N steps=${STEP_LIST}"
echo "[${SCRIPT_NAME}] max_allowed_step=${MAX_ALLOWED_STEP}"
echo "[${SCRIPT_NAME}] divergences=${DIVERGENCES}"
echo "[${SCRIPT_NAME}] skip_existing=${SKIP_EXISTING}"
echo "[${SCRIPT_NAME}] WANDB_PROJECT=${WANDB_PROJECT:-disabled}"

for dataset in "${DATASET_LIST[@]}"; do
    wandb_group="${WANDB_GROUP_OVERRIDE:-eval_qwen38b_tinker_all_${dataset}_thinking}"
    EVAL_ARGS=(
        --base_model "$BASE_MODEL"
        --dataset "$dataset"
        --val_n "$VAL_N"
        --temperature "$TEMPERATURE"
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --enable_thinking
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
        )
        if [[ -n "${WANDB_ENTITY:-}" ]]; then
            EVAL_ARGS+=(--wandb_entity "$WANDB_ENTITY")
        fi
    fi

    echo "[${SCRIPT_NAME}] starting dataset=$dataset wandb_group=$wandb_group"

    BASE_DONE=0
    for step in "${STEPS[@]}"; do
        if [[ "$step" == "0" ]]; then
            if [[ "$BASE_DONE" == "0" ]]; then
                output_file="$(result_file_for "$dataset")"
                if should_skip_result "$output_file"; then
                    echo "[${SCRIPT_NAME}] skipping dataset=$dataset shared base model as step 0; exists: $output_file"
                    BASE_DONE=1
                    continue
                fi

                echo "[${SCRIPT_NAME}] evaluating dataset=$dataset shared base model as step 0"
                NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python evaluate_math.py \
                    "${EVAL_ARGS[@]}" \
                    --output_file "$output_file" \
                    --wandb_step 0 \
                    --wandb_divergence base \
                    --wandb_run_name "eval_qwen38b_tinker_base_${dataset}_thinking"
                BASE_DONE=1
            fi
            continue
        fi

        for div in "${DIVERGENCE_LIST[@]}"; do
            run_config="qwen38b_tinker_${div}"
            exp_dir="$OUT/$run_config"
            checkpoint_dir="$exp_dir/checkpoint-$step"
            output_file="$(result_file_for "$dataset" "$run_config" "checkpoint-$step")"

            if should_skip_result "$output_file"; then
                echo "[${SCRIPT_NAME}] skipping dataset=$dataset divergence_type=$div checkpoint-$step; exists: $output_file"
                continue
            fi

            if [[ ! -d "$exp_dir" ]]; then
                echo "error: experiment directory does not exist: $exp_dir" >&2
                echo "       Check OUT, or wait until training has saved checkpoints." >&2
                exit 1
            fi
            if [[ ! -d "$checkpoint_dir" ]]; then
                echo "error: missing checkpoint directory: $checkpoint_dir" >&2
                exit 1
            fi

            echo "[${SCRIPT_NAME}] evaluating dataset=$dataset divergence_type=$div checkpoint-$step"
            NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python evaluate_math.py \
                "${EVAL_ARGS[@]}" \
                --checkpoint_dir "$checkpoint_dir" \
                --output_file "$output_file" \
                --wandb_step "$step" \
                --wandb_divergence "$div" \
                --wandb_run_name "eval_${run_config}_checkpoint-${step}_${dataset}_thinking"
        done
    done
done
