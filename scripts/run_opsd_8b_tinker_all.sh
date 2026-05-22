#!/usr/bin/env bash
# Run scripts/run_opsd_8b_tinker.sh for all supported f-divergences.
#
# Usage:
#   bash scripts/run_opsd_8b_tinker_all.sh [--smoke]
#
# Env overrides are forwarded to run_opsd_8b_tinker.sh, for example:
#   MODEL=/path/to/Qwen3-8B OUT=/path/to/runs bash scripts/run_opsd_8b_tinker_all.sh
#
# To run only a subset:
#   DIVERGENCES="reverse_kl jsd" bash scripts/run_opsd_8b_tinker_all.sh

set -euo pipefail

SMOKE="${1:-}"
if [[ -n "$SMOKE" && "$SMOKE" != "--smoke" ]]; then
    echo "error: unknown arg '$SMOKE' (only '--smoke' is supported)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r -a DIVERGENCE_LIST <<< "${DIVERGENCES:-reverse_kl forward_kl jsd improved_forward_kl improved_reverse_kl improved_jsd}"

if (( ${#DIVERGENCE_LIST[@]} == 0 )); then
    echo "error: DIVERGENCES did not contain any entries" >&2
    exit 1
fi

for div in "${DIVERGENCE_LIST[@]}"; do
    case "$div" in
        reverse_kl|forward_kl|jsd|improved_forward_kl|improved_reverse_kl|improved_jsd) ;;
        *)
            echo "error: unknown divergence_type '$div'" >&2
            echo "       must be one of: reverse_kl forward_kl jsd improved_forward_kl improved_reverse_kl improved_jsd" >&2
            exit 1
            ;;
    esac

    echo "[run_opsd_8b_tinker_all] starting divergence_type=$div"
    if [[ -n "$SMOKE" ]]; then
        bash "$SCRIPT_DIR/run_opsd_8b_tinker.sh" "$div" "$SMOKE"
    else
        bash "$SCRIPT_DIR/run_opsd_8b_tinker.sh" "$div"
    fi
    echo "[run_opsd_8b_tinker_all] finished divergence_type=$div"
done
