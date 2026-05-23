#!/usr/bin/env bash
# Run scripts/run_opsd_1b_hkaift.sh for the improved f-divergences.
#
# Usage:
#   bash scripts/run_opsd_1b_hkaift_improved.sh [--smoke]
#
# Env overrides are forwarded to run_opsd_1b_hkaift.sh, for example:
#   MODEL=/path/to/Qwen3-1.7B OUT=/path/to/runs bash scripts/run_opsd_1b_hkaift_improved.sh
#
# To run only one of the improved divergences:
#   DIVERGENCES="improved_jsd" bash scripts/run_opsd_1b_hkaift_improved.sh

set -euo pipefail

SMOKE="${1:-}"
if [[ -n "$SMOKE" && "$SMOKE" != "--smoke" ]]; then
    echo "error: unknown arg '$SMOKE' (only '--smoke' is supported)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r -a DIVERGENCE_LIST <<< "${DIVERGENCES:-improved_forward_kl improved_jsd}"

if (( ${#DIVERGENCE_LIST[@]} == 0 )); then
    echo "error: DIVERGENCES did not contain any entries" >&2
    exit 1
fi

for div in "${DIVERGENCE_LIST[@]}"; do
    case "$div" in
        improved_forward_kl|improved_jsd) ;;
        *)
            echo "error: unknown or unsupported divergence_type '$div'" >&2
            echo "       this wrapper supports: improved_forward_kl improved_jsd" >&2
            exit 1
            ;;
    esac

    echo "[run_opsd_1b_hkaift_improved] starting divergence_type=$div"
    if [[ -n "$SMOKE" ]]; then
        bash "$SCRIPT_DIR/run_opsd_1b_hkaift.sh" "$div" "$SMOKE"
    else
        bash "$SCRIPT_DIR/run_opsd_1b_hkaift.sh" "$div"
    fi
    echo "[run_opsd_1b_hkaift_improved] finished divergence_type=$div"
done
