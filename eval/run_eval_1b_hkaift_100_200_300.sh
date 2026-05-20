#!/usr/bin/env bash
# Evaluate checkpoint-100, checkpoint-200, and checkpoint-300 produced by
# scripts/run_opsd_1b_hkaift.sh.
#
# Usage:
#   bash eval/run_eval_1b_hkaift_100_200_300.sh <divergence_type>
#
# Defaults mirror scripts/run_opsd_1b_hkaift.sh:
#   OUT=/home/hanyang/OPSD/runs/1b_tinker
#   RUN_CONFIG=qwen31b_tinker_<divergence_type>
#
# Useful overrides:
#   BASE_MODEL=/path/or/hf-id  OUT=/path/to/output
#   STEP_LIST=100,200,300  DATASET=aime24  VAL_N=12
#   CUDA_VISIBLE_DEVICES=0,1,2,3  TENSOR_PARALLEL_SIZE=4
#   THINKING=1

set -euo pipefail

DIV="${1:?usage: $0 <divergence_type>}"

case "$DIV" in
    reverse_kl|forward_kl|jsd|improved_forward_kl|improved_jsd) ;;
    *)
        echo "error: unknown divergence_type '$DIV'" >&2
        echo "       must be one of: reverse_kl forward_kl jsd improved_forward_kl improved_jsd" >&2
        exit 1
        ;;
esac

RUN_CONFIG="${RUN_CONFIG:-qwen31b_tinker_${DIV}}"
BASE_MODEL="${BASE_MODEL:-${MODEL:-Qwen/Qwen3-1.7B}}"
OUT="${OUT:-/home/hanyang/OPSD/runs/1b_tinker}"
OUT="${OUT%/}"
STEP_LIST="${STEP_LIST:-100,200,300}"
DATASET="${DATASET:-aime24}"
VAL_N="${VAL_N:-12}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-30000}"

if [[ "$OUT" == "$RUN_CONFIG" || "$OUT" == */"$RUN_CONFIG" ]]; then
    EXP_DIR="$OUT"
else
    EXP_DIR="$OUT/$RUN_CONFIG"
fi

if [[ ! -d "$EXP_DIR" ]]; then
    echo "error: experiment directory does not exist: $EXP_DIR" >&2
    echo "       Check OUT/RUN_CONFIG, or wait until training has saved checkpoints." >&2
    exit 1
fi

read -r -a STEPS <<< "${STEP_LIST//,/ }"
if (( ${#STEPS[@]} == 0 )); then
    echo "error: STEP_LIST did not contain any checkpoint steps" >&2
    exit 1
fi

CHECKPOINTS=()
for step in "${STEPS[@]}"; do
    if ! [[ "$step" =~ ^[0-9]+$ ]]; then
        echo "error: checkpoint step must be numeric, got '$step'" >&2
        exit 1
    fi

    checkpoint_dir="$EXP_DIR/checkpoint-$step"
    if [[ ! -d "$checkpoint_dir" ]]; then
        echo "error: missing checkpoint directory: $checkpoint_dir" >&2
        exit 1
    fi
    CHECKPOINTS+=("$checkpoint_dir")
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE_ARGS=(--no_thinking)
MODE_NAME="non-thinking"
if [[ "${THINKING:-0}" == "1" || "${THINKING:-0}" == "true" || "${THINKING:-0}" == "True" ]]; then
    MODE_ARGS=(--enable_thinking)
    MODE_NAME="thinking"
fi

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

echo "[run_eval_1b_hkaift_100_200_300] BASE_MODEL=$BASE_MODEL"
echo "[run_eval_1b_hkaift_100_200_300] EXP_DIR=$EXP_DIR"
echo "[run_eval_1b_hkaift_100_200_300] mode=$MODE_NAME dataset=$DATASET val_n=$VAL_N steps=${STEP_LIST}"

for checkpoint_dir in "${CHECKPOINTS[@]}"; do
    echo "[run_eval_1b_hkaift_100_200_300] evaluating $checkpoint_dir"
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python evaluate_math.py \
        "${EVAL_ARGS[@]}" \
        --checkpoint_dir "$checkpoint_dir"
done
