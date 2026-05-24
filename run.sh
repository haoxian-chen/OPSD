#!/usr/bin/env bash
# Run the 1B HKAIFT training jobs first, then evaluate their saved checkpoints.
#
# Usage:
#   bash run.sh
#
# Defaults:
#   TRAIN_DIVERGENCES="improved_forward_kl improved_jsd"
#   EVAL_DIVERGENCES="reverse_kl improved_forward_kl improved_reverse_kl improved_jsd"
#   eval uses eval/run_eval_1b_hkaift_reverse_improved_to_50_7gpu.sh
#
# Useful overrides:
#   TRAIN_DIVERGENCES="improved_jsd" bash run.sh
#   EVAL_DIVERGENCES="reverse_kl improved_jsd" bash run.sh
#   MAX_STEP=25 bash run.sh
#   EVAL_GPUS=0,1,2,3,4,5,6 PARALLEL_JOBS=7 bash run.sh
#   RUN_TRAINING=0 bash run.sh
#   RUN_EVAL=0 bash run.sh

set -euo pipefail

SMOKE="${1:-}"
if [[ -n "$SMOKE" && "$SMOKE" != "--smoke" ]]; then
    echo "error: unknown arg '$SMOKE' (only '--smoke' is supported)" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_TRAIN_DIVERGENCES="improved_forward_kl improved_jsd"
DEFAULT_EVAL_DIVERGENCES="reverse_kl improved_forward_kl improved_reverse_kl improved_jsd"
TRAIN_DIVERGENCES="${TRAIN_DIVERGENCES:-$DEFAULT_TRAIN_DIVERGENCES}"
EVAL_DIVERGENCES="${EVAL_DIVERGENCES:-$DEFAULT_EVAL_DIVERGENCES}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval/run_eval_1b_hkaift_reverse_improved_to_50_7gpu.sh}"
if [[ "$EVAL_SCRIPT" == /* ]]; then
    EVAL_SCRIPT_PATH="$EVAL_SCRIPT"
else
    EVAL_SCRIPT_PATH="$ROOT_DIR/$EVAL_SCRIPT"
fi

is_disabled() {
    [[ "$1" == "0" || "$1" == "false" || "$1" == "False" ]]
}

if ! is_disabled "$RUN_TRAINING"; then
    read -r -a TRAIN_LIST <<< "$TRAIN_DIVERGENCES"
    if (( ${#TRAIN_LIST[@]} == 0 )); then
        echo "error: TRAIN_DIVERGENCES did not contain any entries" >&2
        exit 1
    fi

    echo "[run.sh] training divergences=$TRAIN_DIVERGENCES"
    for div in "${TRAIN_LIST[@]}"; do
        echo "[run.sh] starting training divergence_type=$div"
        if [[ -n "$SMOKE" ]]; then
            bash "$ROOT_DIR/scripts/run_opsd_1b_hkaift.sh" "$div" "$SMOKE"
        else
            bash "$ROOT_DIR/scripts/run_opsd_1b_hkaift.sh" "$div"
        fi
        echo "[run.sh] finished training divergence_type=$div"
    done
else
    echo "[run.sh] skipping training because RUN_TRAINING=$RUN_TRAINING"
fi

if ! is_disabled "$RUN_EVAL"; then
    EVAL_MAX_STEP="${MAX_STEP:-}"
    if [[ -n "$SMOKE" && -z "$EVAL_MAX_STEP" ]]; then
        EVAL_MAX_STEP=4
    fi

    echo "[run.sh] running eval script=$EVAL_SCRIPT_PATH"
    echo "[run.sh] eval divergences=$EVAL_DIVERGENCES"
    if [[ -n "$EVAL_MAX_STEP" ]]; then
        DIVERGENCES="$EVAL_DIVERGENCES" MAX_STEP="$EVAL_MAX_STEP" bash "$EVAL_SCRIPT_PATH"
    else
        DIVERGENCES="$EVAL_DIVERGENCES" bash "$EVAL_SCRIPT_PATH"
    fi
else
    echo "[run.sh] skipping eval because RUN_EVAL=$RUN_EVAL"
fi
