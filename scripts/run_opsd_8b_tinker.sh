#!/usr/bin/env bash
# Run OPSD on Qwen3-8B (thinking mode) with the Tinker-style sampled-token
# loss for any of the 5 supported f-divergences.
#
# Usage:
#   bash scripts/run_opsd_8b_tinker.sh <divergence_type> [--smoke]
#
#   divergence_type: reverse_kl | forward_kl | jsd | improved_forward_kl | improved_jsd
#   --smoke: short run for pipeline validation
#
# Env overrides:
#   MODEL  (default: /data0/shared/Qwen3-8B)  -- HF id or local path
#   OUT    (default: /data0/siyanz/opsd)      -- output dir for checkpoints

set -euo pipefail

DIV="${1:?usage: $0 <divergence_type> [--smoke]}"
SMOKE="${2:-}"

case "$DIV" in
    reverse_kl|forward_kl|jsd|improved_forward_kl|improved_jsd) ;;
    *)
        echo "error: unknown divergence_type '$DIV'" >&2
        echo "       must be one of: reverse_kl forward_kl jsd improved_forward_kl improved_jsd" >&2
        exit 1
        ;;
esac

MODEL="${MODEL:-/data0/shared/Qwen3-8B}"
OUT="${OUT:-/data0/siyanz/opsd}"

if [[ "$SMOKE" == "--smoke" ]]; then
    RUN_CONFIG="smoke_8b_${DIV}"
    STEP_ARGS=(--max_steps 4 --save_steps 2 --logging_steps 1 --max_completion_length 256)
elif [[ -n "$SMOKE" ]]; then
    echo "error: unknown second arg '$SMOKE' (only '--smoke' is supported)" >&2
    exit 1
else
    RUN_CONFIG="qwen38b_tinker_${DIV}"
    STEP_ARGS=(--num_train_epochs 1 --save_steps 25 --logging_steps 2 --max_completion_length 1024)
fi

echo "[run_opsd_8b_tinker] MODEL=$MODEL"
echo "[run_opsd_8b_tinker] OUT=$OUT"
echo "[run_opsd_8b_tinker] divergence_type=$DIV  run_config=$RUN_CONFIG"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path "$MODEL" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir "$OUT" \
    --run_config "$RUN_CONFIG" \
    "${STEP_ARGS[@]}" \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --use_tinker_loss \
    --divergence_type "$DIV" \
    --jsd_token_clip 0.06 \
    --wandb_project OPSD
