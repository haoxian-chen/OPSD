#!/usr/bin/env bash
# Evaluate all available checkpoints up to checkpoint-200 for the forward_kl and
# jsd runs produced by scripts/run_opsd_1b_hkaift.sh.
#
# Results are saved by evaluate_math.py under eval/eval_results using its
# standard filename pattern:
#   eval_results_<dataset>_<base>_<run_config>_<checkpoint>_<mode>_temp...json
#
# Usage:
#   bash eval/run_eval_1b_hkaift_forward_jsd_to_200.sh
#
# Useful overrides:
#   BASE_MODEL=/path/or/hf-id  OUT=/path/to/output
#   DIVERGENCES="forward_kl jsd"  MAX_STEP=200
#   DATASET=aime24  VAL_N=12  THINKING=1
#   CUDA_VISIBLE_DEVICES=0,1,2,3  TENSOR_PARALLEL_SIZE=4

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-${MODEL:-Qwen/Qwen3-1.7B}}"
OUT="${OUT:-/home/hanyang/OPSD/runs/1b_tinker}"
OUT="${OUT%/}"
DIVERGENCES="${DIVERGENCES:-forward_kl jsd}"
MAX_STEP="${MAX_STEP:-200}"
DATASET="${DATASET:-aime24}"
VAL_N="${VAL_N:-12}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

read -r -a DIVERGENCE_LIST <<< "$DIVERGENCES"

if (( ${#DIVERGENCE_LIST[@]} == 0 )); then
    echo "error: DIVERGENCES did not contain any entries" >&2
    exit 1
fi
if ! [[ "$MAX_STEP" =~ ^[0-9]+$ ]]; then
    echo "error: MAX_STEP must be numeric, got '$MAX_STEP'" >&2
    exit 1
fi

MODE_ARGS=(--enable_thinking)
MODE_NAME="thinking"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
if [[ "${THINKING:-1}" == "0" || "${THINKING:-1}" == "false" || "${THINKING:-1}" == "False" ]]; then
    MODE_ARGS=(--no_thinking)
    MODE_NAME="nonthinking"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-30000}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p eval_results

EVAL_ARGS=(
    --base_model "$BASE_MODEL"
    --dataset "$DATASET"
    --val_n "$VAL_N"
    --temperature "$TEMPERATURE"
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    --max_new_tokens "$MAX_NEW_TOKENS"
    "${MODE_ARGS[@]}"
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
    EVAL_ARGS+=(--wandb_project "$WANDB_PROJECT")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        EVAL_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
    if [[ -n "${WANDB_GROUP:-}" ]]; then
        EVAL_ARGS+=(--wandb_group "$WANDB_GROUP")
    fi
    if [[ -n "${WANDB_TAGS:-}" ]]; then
        EVAL_ARGS+=(--wandb_tags "$WANDB_TAGS")
    fi
fi

echo "[run_eval_1b_hkaift_forward_jsd_to_200] BASE_MODEL=$BASE_MODEL"
echo "[run_eval_1b_hkaift_forward_jsd_to_200] OUT=$OUT"
echo "[run_eval_1b_hkaift_forward_jsd_to_200] mode=$MODE_NAME dataset=$DATASET val_n=$VAL_N max_step=$MAX_STEP"
echo "[run_eval_1b_hkaift_forward_jsd_to_200] divergences=$DIVERGENCES"
echo "[run_eval_1b_hkaift_forward_jsd_to_200] results_dir=$SCRIPT_DIR/eval_results"

for div in "${DIVERGENCE_LIST[@]}"; do
    case "$div" in
        forward_kl|jsd) ;;
        *)
            echo "error: this script is intended for forward_kl and jsd, got '$div'" >&2
            echo "       Override only with DIVERGENCES=\"forward_kl jsd\" or a subset." >&2
            exit 1
            ;;
    esac

    run_config="qwen31b_tinker_${div}"
    exp_dir="$OUT/$run_config"
    if [[ ! -d "$exp_dir" ]]; then
        echo "error: experiment directory does not exist: $exp_dir" >&2
        echo "       Run scripts/run_opsd_1b_hkaift.sh $div first, or set OUT correctly." >&2
        exit 1
    fi

    CHECKPOINTS=()
    while IFS=$'\t' read -r step checkpoint_dir; do
        if [[ "$step" =~ ^[0-9]+$ && "$step" -le "$MAX_STEP" ]]; then
            CHECKPOINTS+=("$step:$checkpoint_dir")
        fi
    done < <(
        find "$exp_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print |
            awk -F'checkpoint-' 'NF > 1 { print $NF "\t" $0 }' |
            sort -n
    )

    if (( ${#CHECKPOINTS[@]} == 0 )); then
        echo "error: no checkpoint-* directories <= $MAX_STEP found under: $exp_dir" >&2
        exit 1
    fi

    for checkpoint_entry in "${CHECKPOINTS[@]}"; do
        step="${checkpoint_entry%%:*}"
        checkpoint_dir="${checkpoint_entry#*:}"
        checkpoint_name="$(basename "$checkpoint_dir")"
        output_file="eval_results/eval_results_${DATASET}_$(basename "$BASE_MODEL")_${run_config}_${checkpoint_name}_${MODE_NAME}_temp${TEMPERATURE}_valn${VAL_N}.json"

        echo "[run_eval_1b_hkaift_forward_jsd_to_200] evaluating divergence_type=$div $checkpoint_name"
        NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python evaluate_math.py \
            "${EVAL_ARGS[@]}" \
            --checkpoint_dir "$checkpoint_dir" \
            --output_file "$output_file" \
            --wandb_step "$step" \
            --wandb_divergence "$div" \
            --wandb_run_name "eval_${run_config}_${checkpoint_name}_${DATASET}_${MODE_NAME}"
    done
done
