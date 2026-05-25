#!/usr/bin/env bash
# Evaluate Qwen3-1.7B HKAIFT checkpoints on 7 single-GPU workers.
#
# By default this evaluates all supported divergence runs for all available
# checkpoints in [25, 200], finishing aime25 before starting hmmt25.
#
# Results are saved under eval/eval_results/<divergence-dir>/ using the
# standard filename pattern:
#   eval_results_<dataset>_<base>_<run_config>_<checkpoint>_<mode>_temp...json
#
# Usage:
#   bash eval/run_eval_1b_hkaift_all_7gpu.sh
#
# Useful overrides:
#   BASE_MODEL=/path/or/hf-id  OUT=/path/to/output
#   DIVERGENCES="reverse_kl forward_kl jsd improved_forward_kl improved_reverse_kl improved_jsd"
#   MIN_STEP=25  MAX_STEP=200  STEP_INTERVAL=1
#   DATASETS="aime25 hmmt25"  DATASET=aime25  VAL_N=12  THINKING=1
#   EVAL_GPUS=0,1,2,3,4,5,6  PARALLEL_JOBS=7

set -euo pipefail

SCRIPT_NAME="$(basename "$0" .sh)"

BASE_MODEL="${BASE_MODEL:-${MODEL:-Qwen/Qwen3-1.7B}}"
BASE_MODEL_NAME="$(basename "$BASE_MODEL")"
OUT="${OUT:-/home/hanyang/OPSD/runs/1b_tinker}"
OUT="${OUT%/}"
DIVERGENCES="${DIVERGENCES:-reverse_kl forward_kl jsd improved_forward_kl improved_reverse_kl improved_jsd}"
MIN_STEP="${MIN_STEP:-25}"
MAX_STEP="${MAX_STEP:-200}"
STEP_INTERVAL="${STEP_INTERVAL:-1}"
DATASETS="${DATASETS:-${DATASET:-aime25 hmmt25}}"
VAL_N="${VAL_N:-12}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-42}"
EVAL_GPUS="${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}}"
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
LOG_DIR="${LOG_DIR:-eval_logs}"

read -r -a DIVERGENCE_LIST <<< "$DIVERGENCES"
read -r -a RAW_DATASET_LIST <<< "${DATASETS//,/ }"
read -r -a GPU_LIST <<< "${EVAL_GPUS//,/ }"
PARALLEL_JOBS="${PARALLEL_JOBS:-${#GPU_LIST[@]}}"

if (( ${#DIVERGENCE_LIST[@]} == 0 )); then
    echo "error: DIVERGENCES did not contain any entries" >&2
    exit 1
fi
if (( ${#RAW_DATASET_LIST[@]} == 0 )); then
    echo "error: DATASETS did not contain any entries" >&2
    exit 1
fi
if (( ${#GPU_LIST[@]} == 0 )); then
    echo "error: EVAL_GPUS did not contain any GPU ids" >&2
    exit 1
fi
if ! [[ "$MIN_STEP" =~ ^[0-9]+$ ]]; then
    echo "error: MIN_STEP must be numeric, got '$MIN_STEP'" >&2
    exit 1
fi
if ! [[ "$MAX_STEP" =~ ^[0-9]+$ ]]; then
    echo "error: MAX_STEP must be numeric, got '$MAX_STEP'" >&2
    exit 1
fi
if ! [[ "$STEP_INTERVAL" =~ ^[0-9]+$ ]] || (( STEP_INTERVAL < 1 )); then
    echo "error: STEP_INTERVAL must be a positive integer, got '$STEP_INTERVAL'" >&2
    exit 1
fi
if (( MIN_STEP > MAX_STEP )); then
    echo "error: MIN_STEP=$MIN_STEP exceeds MAX_STEP=$MAX_STEP" >&2
    exit 1
fi
if ! [[ "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || (( PARALLEL_JOBS < 1 )); then
    echo "error: PARALLEL_JOBS must be a positive integer, got '$PARALLEL_JOBS'" >&2
    exit 1
fi
if (( PARALLEL_JOBS > ${#GPU_LIST[@]} )); then
    echo "error: PARALLEL_JOBS=$PARALLEL_JOBS exceeds the number of EVAL_GPUS (${#GPU_LIST[@]})" >&2
    exit 1
fi

for div in "${DIVERGENCE_LIST[@]}"; do
    case "$div" in
        reverse_kl|forward_kl|jsd|improved_forward_kl|improved_reverse_kl|improved_jsd) ;;
        *)
            echo "error: unsupported divergence '$div'" >&2
            echo "       Use one of: reverse_kl forward_kl jsd improved_forward_kl improved_reverse_kl improved_jsd" >&2
            exit 1
            ;;
    esac
done

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
if [[ "${THINKING:-1}" == "0" || "${THINKING:-1}" == "false" || "${THINKING:-1}" == "False" ]]; then
    MODE_ARGS=(--no_thinking)
    MODE_NAME="nonthinking"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-30000}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p eval_results
mkdir -p "$LOG_DIR"

result_dir_for_divergence() {
    local div="$1"
    case "$div" in
        reverse_kl) printf 'qwen31b_reverse_kl' ;;
        forward_kl) printf 'qwen31b_forward_kl' ;;
        jsd) printf 'qwen31b_jsd' ;;
        improved_forward_kl) printf 'qwen31b_improved_forward' ;;
        improved_reverse_kl) printf 'qwen31b_improved_reverse' ;;
        improved_jsd) printf 'qwen31b_improved_jsd' ;;
        *)
            echo "error: unknown divergence '$div'" >&2
            return 1
            ;;
    esac
}

COMMON_EVAL_ARGS=(
    --base_model "$BASE_MODEL"
    --val_n "$VAL_N"
    --temperature "$TEMPERATURE"
    --seed "$SEED"
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    --max_new_tokens "$MAX_NEW_TOKENS"
    "${MODE_ARGS[@]}"
)

if [[ -n "${TOP_P:-}" ]]; then
    COMMON_EVAL_ARGS+=(--top_p "$TOP_P")
fi
if [[ -n "${TOP_K:-}" ]]; then
    COMMON_EVAL_ARGS+=(--top_k "$TOP_K")
fi
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    COMMON_EVAL_ARGS+=(--max_model_len "$MAX_MODEL_LEN")
fi
if [[ -n "${NUM_SAMPLES:-}" ]]; then
    COMMON_EVAL_ARGS+=(--num_samples "$NUM_SAMPLES")
fi
if [[ -n "${PRESENCE_PENALTY:-}" ]]; then
    COMMON_EVAL_ARGS+=(--presence_penalty "$PRESENCE_PENALTY")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    COMMON_EVAL_ARGS+=(--wandb_project "$WANDB_PROJECT")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        COMMON_EVAL_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
    if [[ -n "${WANDB_GROUP:-}" ]]; then
        COMMON_EVAL_ARGS+=(--wandb_group "$WANDB_GROUP")
    fi
    if [[ -n "${WANDB_TAGS:-}" ]]; then
        COMMON_EVAL_ARGS+=(--wandb_tags "$WANDB_TAGS")
    fi
fi

echo "[$SCRIPT_NAME] BASE_MODEL=$BASE_MODEL"
echo "[$SCRIPT_NAME] OUT=$OUT"
echo "[$SCRIPT_NAME] mode=$MODE_NAME datasets=${DATASET_LIST[*]} val_n=$VAL_N min_step=$MIN_STEP max_step=$MAX_STEP step_interval=$STEP_INTERVAL"
echo "[$SCRIPT_NAME] divergences=$DIVERGENCES"
echo "[$SCRIPT_NAME] eval_gpus=$EVAL_GPUS parallel_jobs=$PARALLEL_JOBS tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
echo "[$SCRIPT_NAME] results_dir=$SCRIPT_DIR/eval_results"
echo "[$SCRIPT_NAME] log_dir=$SCRIPT_DIR/$LOG_DIR"

run_eval_task() {
    local gpu="$1"
    local dataset="$2"
    local div="$3"
    local step="$4"
    local checkpoint_dir="$5"
    local output_file="$6"
    local run_config="$7"
    local checkpoint_name
    local log_file

    checkpoint_name="$(basename "$checkpoint_dir")"
    log_file="$LOG_DIR/${run_config}_${checkpoint_name}_${dataset}_${MODE_NAME}.log"

    echo "[$SCRIPT_NAME] starting dataset=$dataset gpu=$gpu divergence_type=$div $checkpoint_name output=$output_file log=$log_file"
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$gpu" python evaluate_math.py \
        "${COMMON_EVAL_ARGS[@]}" \
        --dataset "$dataset" \
        --checkpoint_dir "$checkpoint_dir" \
        --output_file "$output_file" \
        --wandb_step "$step" \
        --wandb_divergence "$div" \
        --wandb_run_name "eval_${run_config}_${checkpoint_name}_${dataset}_${MODE_NAME}" \
        > "$log_file" 2>&1
    echo "[$SCRIPT_NAME] finished dataset=$dataset gpu=$gpu divergence_type=$div $checkpoint_name"
}

for dataset in "${DATASET_LIST[@]}"; do
    TASK_DIVS=()
    TASK_STEPS=()
    TASK_CHECKPOINT_DIRS=()
    TASK_OUTPUT_FILES=()
    TASK_RUN_CONFIGS=()

    for div in "${DIVERGENCE_LIST[@]}"; do
        run_config="qwen31b_tinker_${div}"
        exp_dir="$OUT/$run_config"
        result_subdir="$(result_dir_for_divergence "$div")"
        mkdir -p "eval_results/$result_subdir"

        if [[ ! -d "$exp_dir" ]]; then
            echo "error: experiment directory does not exist: $exp_dir" >&2
            echo "       Run scripts/run_opsd_1b_hkaift.sh $div first, or set OUT correctly." >&2
            exit 1
        fi

        CHECKPOINTS=()
        while IFS=$'\t' read -r step checkpoint_dir; do
            if [[ "$step" =~ ^[0-9]+$ ]]; then
                step_num=$((10#$step))
                if (( step_num >= MIN_STEP && step_num <= MAX_STEP && (step_num - MIN_STEP) % STEP_INTERVAL == 0 )); then
                    CHECKPOINTS+=("$step_num:$checkpoint_dir")
                fi
            fi
        done < <(
            find "$exp_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print |
                awk -F'checkpoint-' 'NF > 1 { print $NF "\t" $0 }' |
                sort -n
        )

        if (( ${#CHECKPOINTS[@]} == 0 )); then
            echo "error: no checkpoint-* directories in [$MIN_STEP, $MAX_STEP] matching STEP_INTERVAL=$STEP_INTERVAL found under: $exp_dir" >&2
            exit 1
        fi

        for checkpoint_entry in "${CHECKPOINTS[@]}"; do
            step="${checkpoint_entry%%:*}"
            checkpoint_dir="${checkpoint_entry#*:}"
            checkpoint_name="$(basename "$checkpoint_dir")"
            output_file="eval_results/${result_subdir}/eval_results_${dataset}_${BASE_MODEL_NAME}_${run_config}_${checkpoint_name}_${MODE_NAME}_temp${TEMPERATURE}_valn${VAL_N}.json"

            TASK_DIVS+=("$div")
            TASK_STEPS+=("$step")
            TASK_CHECKPOINT_DIRS+=("$checkpoint_dir")
            TASK_OUTPUT_FILES+=("$output_file")
            TASK_RUN_CONFIGS+=("$run_config")
        done
    done

    if (( ${#TASK_DIVS[@]} == 0 )); then
        echo "error: no evaluation tasks were created for dataset=$dataset" >&2
        exit 1
    fi

    echo "[$SCRIPT_NAME] queued ${#TASK_DIVS[@]} checkpoint eval tasks for dataset=$dataset"

    for (( batch_start=0; batch_start<${#TASK_DIVS[@]}; batch_start+=PARALLEL_JOBS )); do
        PIDS=()
        LABELS=()

        for (( slot=0; slot<PARALLEL_JOBS && batch_start+slot<${#TASK_DIVS[@]}; slot++ )); do
            task_idx=$((batch_start + slot))
            gpu="${GPU_LIST[$slot]}"
            checkpoint_name="$(basename "${TASK_CHECKPOINT_DIRS[$task_idx]}")"
            label="${dataset}:${TASK_DIVS[$task_idx]}:${checkpoint_name}:gpu${gpu}"

            run_eval_task \
                "$gpu" \
                "$dataset" \
                "${TASK_DIVS[$task_idx]}" \
                "${TASK_STEPS[$task_idx]}" \
                "${TASK_CHECKPOINT_DIRS[$task_idx]}" \
                "${TASK_OUTPUT_FILES[$task_idx]}" \
                "${TASK_RUN_CONFIGS[$task_idx]}" &

            PIDS+=("$!")
            LABELS+=("$label")
        done

        batch_failed=0
        for idx in "${!PIDS[@]}"; do
            if ! wait "${PIDS[$idx]}"; then
                echo "error: eval task failed: ${LABELS[$idx]}" >&2
                batch_failed=1
            fi
        done

        if (( batch_failed != 0 )); then
            echo "error: stopping after failed eval batch; see logs under $SCRIPT_DIR/$LOG_DIR" >&2
            exit 1
        fi
    done

    echo "[$SCRIPT_NAME] finished all eval tasks for dataset=$dataset"
done

echo "[$SCRIPT_NAME] all eval tasks finished"
