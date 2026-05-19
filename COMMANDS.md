# OPSD Replication Commands

Commands to replicate OPSD on **Qwen3-1.7B** with the Tinker-style sampled-token
loss for all 5 f-divergences:

1. `reverse_kl` — original (bit-exact with prior implementation)
2. `forward_kl`
3. `jsd`
4. `improved_forward_kl` — bias-corrected exact-PG variant
5. `improved_jsd` — bias-corrected exact-PG variant

All commands assume 4×H100 (or similar) and the `opsd` conda env from `environment.yml`.
Adjust `--model_name_or_path`, `--output_dir`, and `CUDA_VISIBLE_DEVICES` for your setup.

> The 5 runs are differentiated only by `--divergence_type` and `--run_config`.
> Everything else mirrors `scripts/run_opsd_1b.sh`.

---

## 0. Setup

```bash
conda env create -f environment.yml
conda activate opsd
pip install flash-attn==2.8.3 --no-build-isolation
```

Shared variables used below:

```bash
export MODEL=/data0/shared/Qwen3-1.7B
export OUT=/data0/siyanz/opsd
```

---

## 1. Reverse KL (original Tinker-style loss)

```bash
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path $MODEL \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir $OUT \
    --run_config qwen31b_tinker_reverse_kl \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
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
    --divergence_type reverse_kl \
    --jsd_token_clip 0.05 \
    --wandb_project OPSD
```

## 2. Forward KL

```bash
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path $MODEL \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir $OUT \
    --run_config qwen31b_tinker_forward_kl \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
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
    --divergence_type forward_kl \
    --jsd_token_clip 0.05 \
    --wandb_project OPSD
```

## 3. JSD

```bash
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path $MODEL \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir $OUT \
    --run_config qwen31b_tinker_jsd \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
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
    --divergence_type jsd \
    --jsd_token_clip 0.05 \
    --wandb_project OPSD
```

## 4. Improved Forward KL (exact-PG)

```bash
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path $MODEL \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir $OUT \
    --run_config qwen31b_tinker_improved_forward_kl \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
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
    --divergence_type improved_forward_kl \
    --jsd_token_clip 0.05 \
    --wandb_project OPSD
```

## 5. Improved JSD (exact-PG)

```bash
accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path $MODEL \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir $OUT \
    --run_config qwen31b_tinker_improved_jsd \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
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
    --divergence_type improved_jsd \
    --jsd_token_clip 0.05 \
    --wandb_project OPSD
```

---

## 6. Evaluation

For each `$RUN_CONFIG` above, evaluate the base model once and each saved checkpoint
on AIME24 / AIME25 / HMMT25:

```bash
cd eval

BASE_MODEL=/data0/shared/Qwen3-1.7B

# Base model (run once across all variants)
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset aime24 \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 4

# Loop over variants and checkpoints
for variant in reverse_kl forward_kl jsd improved_forward_kl improved_jsd; do
    EXP_DIR=/data0/siyanz/opsd/qwen31b_tinker_${variant}
    for step in 25 50 75 100; do
        for ds in aime24 aime25 hmmt25; do
            NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
                --base_model "$BASE_MODEL" \
                --dataset "$ds" \
                --val_n 12 \
                --temperature 1.0 \
                --tensor_parallel_size 4 \
                --checkpoint_dir "$EXP_DIR/checkpoint-$step"
        done
    done
done
```

Evaluation settings match the README: temperature=1.0, thinking mode, max new tokens=38912, top-p=none, top-k disabled, min-p=0, num samples=12.

---

## 7. Unit tests for the new loss helpers

```bash
pytest tests/ -q
```

Covers `_compute_neg_g_u`, `_tinker_loss_from_logprobs`, and a bit-exact
regression guard for `divergence_type="reverse_kl"` against the pre-refactor
formula.
