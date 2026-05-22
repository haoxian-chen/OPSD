#!/usr/bin/env bash
# Evaluate Qwen3-8B Tinker runs for all supported f-divergences with Qwen3
# thinking mode enabled. Step 0 is evaluated once as the shared base model;
# positive steps are evaluated per divergence.
#
# Usage:
#   bash eval/run_eval_8b_tinker_all_0_200_thinking.sh
#
# Defaults mirror scripts/run_opsd_8b_tinker.sh:
#   OUT=/data0/siyanz/opsd
#   RUN_CONFIG=qwen38b_tinker_<divergence_type>
#
# Useful overrides:
#   BASE_MODEL=/path/or/hf-id  OUT=/path/to/output
#   DIVERGENCES="reverse_kl jsd"
#   STEP_LIST=0,25,50,75,100,125,150,175,200
#   DATASET=aime24  VAL_N=12
#   CUDA_VISIBLE_DEVICES=0,1,2,3  TENSOR_PARALLEL_SIZE=4

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-${MODEL:-/data0/shared/Qwen3-8B}}"
OUT="${OUT:-/data0/siyanz/opsd}"
OUT="${OUT%/}"
DIVERGENCES="${DIVERGENCES:-reverse_kl forward_kl jsd improved_forward_kl improved_jsd}"
STEP_LIST="${STEP_LIST:-0,25,50,75,100,125,150,175,200}"
MAX_ALLOWED_STEP="${MAX_ALLOWED_STEP:-200}"
DATASET="${DATASET:-aime24}"
VAL_N="${VAL_N:-12}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
WANDB_PROJECT="${WANDB_PROJECT-OPSD}"
WANDB_GROUP="${WANDB_GROUP:-eval_qwen38b_tinker_all_${DATASET}_thinking}"
WANDB_TAGS="${WANDB_TAGS:-eval,8b,tinker,thinking}"

read -r -a DIVERGENCE_LIST <<< "$DIVERGENCES"
read -r -a STEPS <<< "${STEP_LIST//,/ }"

if (( ${#DIVERGENCE_LIST[@]} == 0 )); then
    echo "error: DIVERGENCES did not contain any entries" >&2
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
cd "$SCRIPT_DIR"

EVAL_ARGS=(
    --base_model "$BASE_MODEL"
    --dataset "$DATASET"
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
        --wandb_group "$WANDB_GROUP"
        --wandb_tags "$WANDB_TAGS"
    )
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        EVAL_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

echo "[run_eval_8b_tinker_all_0_200_thinking] BASE_MODEL=$BASE_MODEL"
echo "[run_eval_8b_tinker_all_0_200_thinking] OUT=$OUT"
echo "[run_eval_8b_tinker_all_0_200_thinking] mode=thinking dataset=$DATASET val_n=$VAL_N steps=${STEP_LIST}"
echo "[run_eval_8b_tinker_all_0_200_thinking] max_allowed_step=${MAX_ALLOWED_STEP}"
echo "[run_eval_8b_tinker_all_0_200_thinking] divergences=${DIVERGENCES}"
echo "[run_eval_8b_tinker_all_0_200_thinking] WANDB_PROJECT=${WANDB_PROJECT:-disabled}"

BASE_DONE=0
for step in "${STEPS[@]}"; do
    if [[ "$step" == "0" ]]; then
        if [[ "$BASE_DONE" == "0" ]]; then
            echo "[run_eval_8b_tinker_all_0_200_thinking] evaluating shared base model as step 0"
            NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python evaluate_math.py \
                "${EVAL_ARGS[@]}" \
                --wandb_step 0 \
                --wandb_divergence base \
                --wandb_run_name "eval_qwen38b_tinker_base_${DATASET}_thinking"
            BASE_DONE=1
        fi
        continue
    fi

    for div in "${DIVERGENCE_LIST[@]}"; do
        run_config="qwen38b_tinker_${div}"
        exp_dir="$OUT/$run_config"
        checkpoint_dir="$exp_dir/checkpoint-$step"

        if [[ ! -d "$exp_dir" ]]; then
            echo "error: experiment directory does not exist: $exp_dir" >&2
            echo "       Check OUT, or wait until training has saved checkpoints." >&2
            exit 1
        fi
        if [[ ! -d "$checkpoint_dir" ]]; then
            echo "error: missing checkpoint directory: $checkpoint_dir" >&2
            exit 1
        fi

        echo "[run_eval_8b_tinker_all_0_200_thinking] evaluating divergence_type=$div checkpoint-$step"
        NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python evaluate_math.py \
            "${EVAL_ARGS[@]}" \
            --checkpoint_dir "$checkpoint_dir" \
            --wandb_step "$step" \
            --wandb_divergence "$div" \
            --wandb_run_name "eval_${run_config}_checkpoint-${step}_${DATASET}_thinking"
    done
done
