# OPSD Replication Commands

Commands to replicate OPSD on **Qwen3-1.7B** with the Tinker-style sampled-token
loss for all 6 f-divergences:

1. `reverse_kl` — original (bit-exact with prior implementation)
2. `forward_kl`
3. `jsd`
4. `improved_forward_kl` — bias-corrected exact-PG variant
5. `improved_reverse_kl` — bias-corrected exact-PG variant
6. `improved_jsd` — bias-corrected exact-PG variant

All 6 variants share the same hyper-parameters; they differ only in
`--divergence_type` and `--run_config`. We expose them through parameterized scripts:

| Script | Model | Mode | Notes |
|---|---|---|---|
| `scripts/run_opsd_1b_tinker.sh` | Qwen3-1.7B | thinking | 4 GPUs, jsd_token_clip 0.05 |
| `scripts/run_opsd_4b_tinker.sh` | Qwen3-4B | thinking | 8 GPUs, jsd_token_clip 0.05 |
| `scripts/run_opsd_4b_nonthink_tinker.sh` | Qwen3-4B | non-thinking | 4 GPUs, jsd_token_clip 1e-6 |
| `scripts/run_opsd_8b_tinker.sh` | Qwen3-8B | thinking | 8 GPUs, jsd_token_clip 0.06 |
| `scripts/run_opsd_8b_nonthink_tinker.sh` | Qwen3-8B | non-thinking | 8 GPUs, jsd_token_clip 1e-7 |

All take `<divergence_type> [--smoke]` as args.

---

## 0. Setup

```bash
conda env create -f environment.yml
conda activate opsd
pip install flash-attn==2.8.3 --no-build-isolation
```

One-time logins:

```bash
hf auth login       # for model + dataset downloads
wandb login         # or set WANDB_MODE=disabled to skip
```

Shared variables (override the script defaults):

```bash
export MODEL=Qwen/Qwen3-1.7B            # HF id; auto-downloads on first use
export OUT=/home/$USER/opsd_runs        # checkpoints land here
mkdir -p $OUT
```

---

## 1. Smoke test (≈2–3 min)

Verify the pipeline end-to-end before committing to a 15-min run:

```bash
bash scripts/run_opsd_1b_tinker.sh reverse_kl --smoke
```

`--smoke` overrides the training schedule to `max_steps=4`, `save_steps=2`,
`max_completion_length=256`, and prefixes the run name with `smoke_`.

Pass criteria:
- no tracebacks
- `$OUT/smoke_reverse_kl_*/checkpoint-{2,4}` exist
- loss printed each step
- vLLM init + a generation actually happened

---

## 2. Full training — all 6 divergences

Each run takes ~15 min on 4×H100, saves checkpoints at steps 25/50/75/100.

```bash
bash scripts/run_opsd_1b_hkaift.sh reverse_kl
bash scripts/run_opsd_1b_hkaift.sh forward_kl
bash scripts/run_opsd_1b_hkaift.sh jsd
bash scripts/run_opsd_1b_hkaift.sh improved_forward_kl
bash scripts/run_opsd_1b_hkaift.sh improved_reverse_kl
bash scripts/run_opsd_1b_hkaift.sh improved_jsd
```

Run them serially in `tmux`/`screen` so they survive SSH disconnects:

```bash
tmux new -s opsd
for div in reverse_kl forward_kl jsd improved_forward_kl improved_reverse_kl improved_jsd; do
    bash scripts/run_opsd_1b_tinker.sh $div
done
# Ctrl+B then D to detach; `tmux attach -t opsd` to reattach
```

Checkpoints land in `$OUT/qwen31b_tinker_<divergence>/checkpoint-{25,50,75,100}`.

---

## 3. Evaluation

Evaluate the base model once, then each checkpoint of each variant on AIME24 / AIME25 / HMMT25:

```bash
cd eval

# Base model (run once)
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
    --base_model "$MODEL" \
    --dataset aime24 \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 4

# Each variant × checkpoint × dataset
for variant in reverse_kl forward_kl jsd improved_forward_kl improved_reverse_kl improved_jsd; do
    EXP_DIR=$OUT/qwen31b_tinker_${variant}
    for step in 25 50 75 100; do
        for ds in aime24 aime25 hmmt25; do
            NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
                --base_model "$MODEL" \
                --dataset "$ds" \
                --val_n 12 \
                --temperature 1.0 \
                --tensor_parallel_size 4 \
                --checkpoint_dir "$EXP_DIR/checkpoint-$step"
        done
    done
done
```

Evaluation settings match the README: temperature=1.0, thinking mode,
max new tokens=38912, top-p=none, top-k disabled, min-p=0, num samples=12.

---

## 4. Unit tests for the new loss helpers

```bash
pytest tests/ -q
```

Covers `_compute_neg_g_u`, `_tinker_loss_from_logprobs`, and a bit-exact
regression guard for `divergence_type="reverse_kl"` against the pre-refactor
formula.

---

## Notes

- **Datasets** auto-download from HuggingFace: training uses
  `siyanzhao/Openthoughts_math_30k_opsd`; evaluation uses
  `HuggingFaceH4/aime_2024`, `yentinglin/aime_2025`, `MathArena/hmmt_feb_2025`.
- **Model**: `MODEL=Qwen/Qwen3-1.7B` downloads via HF on first run.
  Use a local path (e.g. `/data0/shared/Qwen3-1.7B`) if your nodes have no internet.
- **Port conflict**: if `--main_process_port 12949` is taken, edit the script.
- **WandB**: runs log to project `OPSD` with run name `qwen31b_tinker_<divergence>`.
  Set `WANDB_MODE=disabled` to skip.
