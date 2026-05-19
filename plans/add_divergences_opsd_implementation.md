# Add 4 New f-Divergence Per-Token Advantages to OPSD's Tinker-Style Loss

Plan for extending the sampled-token (Thinking-Machines-style) loss in
`opsd_trainer.py` so the per-token advantage can use 5 different f-divergences
instead of only Reverse KL. Mirrors the design in
`plans/add_divergences_implementation.md` (which targeted the `tinker_cookbook`
package) but adapted to this repo's `OPSDTrainer.compute_loss`.

## Goal

The current `use_thinking_machines_loss=True` branch hard-codes Reverse KL:

```python
# opsd_trainer.py:714, 726
advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()
loss = -(advantage * student_log_probs_sampled_masked).mean()
```

i.e. per-token advantage `= log u` where `u = p_teacher / p_student`, and
`loss = -advantage * log π_student`. We generalise the advantage to
`-g(u)` for any f-divergence with generator `g(u)`, keeping the same
policy-gradient outer form.

## Math

Let `log_u = log p_teacher − log q_student` (already scaled by `1/temperature`,
see "Temperature" note below) and `u = exp(log_u)`. Five supported divergences:

| `divergence_type`     | `g(u)`                              | `-g(u)` used as advantage                        |
| --------------------- | ----------------------------------- | ------------------------------------------------ |
| `reverse_kl`          | `-ln u`                             | `log_u`                                          |
| `forward_kl`          | `u ln u`                            | `-u * log_u`                                     |
| `jsd`                 | `0.5 [u ln u - (1+u) ln((1+u)/2)]`  | `-0.5 * (u*log_u - (u+1)*(log1p(u) - log 2))`    |
| `improved_forward_kl` | `1 - u`                             | `u - 1`                                          |
| `improved_jsd`        | `-0.5 ln((1+u)/2)`                  | `0.5 * (log1p(u) - log 2)`                       |

> **Note on "`g(u)`" in this table.** Here `g` is just the symbol we wire into
> the code so `-g(u)` is the per-token advantage. For `reverse_kl`,
> `forward_kl`, `jsd`, `g` *is* the f-divergence generator (i.e.
> `D_f(p‖q) = E_q[g(u)]`). For `improved_forward_kl` and `improved_jsd`, `g`
> is *not* a divergence generator — it's the bias-corrected PG integrand
> `(f − u f')` of forward KL / JSD respectively, packaged into the same slot.
> See the policy-gradient section below for the algebra. This distinction
> matters for the metric naming (you cannot just log `mean(g(u))` and call
> it "the divergence" for the improved variants).

### Behavior at `u = 1` (student matches teacher)

`g(1) = 0` for all five entries above, so the per-token advantage `-g(u)`
vanishes when student == teacher — once the student matches, there's no
gradient signal. `improved_forward_kl` uses `g(u) = 1 - u` (rather than the
zero-mean-equivalent `g(u) = -u`) precisely to get this zero-at-match
property without changing the gradient up to baseline.

### Policy-gradient interpretation

With advantage `A = -g(u)` (detached) and loss `L = -A · log π_student`, the
gradient is `∇θ L = -A · ∇θ log π_student`. The exact f-divergence
score-function gradient under sampling from `q_student` is

```
∇θ D_f(p_teacher || q_θ) = E_{q_θ}[(f(u) − u f'(u)) · ∇θ log q_θ],
```

so the *exact* per-token advantage is `A_exact = -(f − u f')` (any constant
baseline can be added, since `E_{q_θ}[c · ∇θ log q_θ] = 0`). The five options
split cleanly into "exact PG" and "biased MC-of-integrand":

| `divergence_type`     | Per-token A used here                 | Target D_f          | Status                                                  |
| --------------------- | ------------------------------------- | ------------------- | ------------------------------------------------------- |
| `reverse_kl`          | `log u`                               | `D_KL(q‖p)`         | **Exact PG** (matches `-(f − u f') = log u − 1` up to baseline `+1`) |
| `improved_forward_kl` | `u − 1`                               | `D_KL(p‖q)`         | **Exact PG** for forward KL `f(u) = u log u` (matches `-(f − u f') = u` minus baseline `1`) |
| `improved_jsd`        | `0.5 (log1p(u) − log 2)`              | `D_JSD(p, q)`       | **Exact PG** for JSD `f` (algebra: `f − u f' = -0.5 log((1+u)/2)`) |
| `forward_kl`          | `-u log u`                            | `D_KL(p‖q)`         | **Biased**: uses `-f(u)` (the integrand of `D_f = E_q[f]`) rather than `-(f − u f')`. |
| `jsd`                 | `-0.5 (u log u − (1+u)(log1p(u) − log 2))` | `D_JSD(p, q)` | **Biased**: same `-f(u)` substitution as above.        |

In other words, the `improved_*` names are *exactly* the bias-corrected
forward-KL and JSD score-function gradients (up to a zero-mean baseline); the
unprefixed `forward_kl` and `jsd` are the looser "MC the divergence integrand"
estimators we inherit from the `tinker_cookbook` convention. `reverse_kl`
happens to coincide for both definitions because the IS factor `u` does not
appear in its f-generator (`f(u) = -log u`).

This matters for users tuning: prefer `improved_forward_kl` / `improved_jsd`
when you want the unbiased PG; keep `forward_kl` / `jsd` if you want
parity with the tinker recipe. The docstring on `divergence_type` should
state this explicitly.

## Numerical stability (same strategy as tinker plan, with one OPSD-specific addition)

- **Upcast to float32 inside the helper for non-`reverse_kl` paths.** In the
  `tinker_cookbook` codepath, `sampled_logprobs` and `teacher_logprobs` are
  already float32 by the time they reach `_compute_neg_g_u`. In OPSD, `log_u`
  is computed from `F.log_softmax` outputs of a bf16/fp16 model and inherits
  that dtype. `exp`, `log1p`, and `u * log_u` on bf16 lose precision fast in
  the tails (bf16 mantissa is 7 bits). Cast at the helper boundary:
  `log_u = log_u.float()` before any `exp`/`log1p` op for the four
  non-`reverse_kl` variants. Skip the cast for `reverse_kl` so it stays
  bit-exact with the prior code.
- **Compute `log_u` first** in log-space (subtraction of log-probs), then
  `u = exp(log_u_clamped)`.
- **Clamp `log_u` to `[-10, 10]` before `exp()`**. Without it, an early
  mismatch (`log_u = 30`) gives `u ≈ 1e13` and `u·log_u` saturates to Inf.
  Clamp keeps `u ∈ [4.5e-5, 22026]`, all safe in fp32.
- **Use `torch.log1p(u)`** for the `ln((1+u)/2)` terms in JSD variants.
- **`reverse_kl` left unclamped** so behavior remains bit-exact with the
  current code (which doesn't `exp` `log_u`).
- For the four non-reverse-KL variants, the implementation will use the
  **clamped** `log_u` in the multiplicative slots as well (i.e.
  `-u * log_u_clamped`, not `-u * log_u`). This is the "clipped estimator"
  consistency choice from the tinker plan; flagged here too for parity.

### Temperature subtlety (OPSD-specific)

This repo applies `self.temperature` before softmax for both student and
teacher (`opsd_trainer.py:649, 692`):

```python
student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
```

So `log_u` is the log-density-ratio of **temperature-scaled** distributions,
not the raw model outputs. This is consistent with how the existing
reverse-KL loss already behaves, but worth noting: tuning `temperature`
changes the effective sharpness of `u` and therefore the dynamic range of
`-g(u)`. No behavior change for `reverse_kl`; for the new divergences,
`temperature` will interact with the clamp window in `_compute_neg_g_u`.

## Files to change

### 1. `opsd_trainer.py`

**Imports**: add `import math` at the top (next to `import torch`,
`import torch.nn.functional as F`).

**Module-level constant**: add near other module-level definitions
(close to the existing imports/`EMAUpdateCallback`):

```python
DIVERGENCE_TYPES = (
    "reverse_kl",
    "forward_kl",
    "jsd",
    "improved_forward_kl",
    "improved_jsd",
)
_LOG2 = math.log(2.0)
```

**New private helper** (place above `OPSDTrainer` or as a `@staticmethod`
inside it — whichever matches the file's style; helper is stateless):

```python
def _compute_neg_g_u(log_u: torch.Tensor, divergence_type: str) -> torch.Tensor:
    """Return -g(u) per token for the chosen f-divergence.

    log_u: tensor of log p_teacher - log q_student (any shape).
    divergence_type: one of DIVERGENCE_TYPES.
    """
    if divergence_type == "reverse_kl":
        # Bit-exact with the prior implementation; no dtype cast.
        return log_u

    # Upcast for the exp/log1p/multiplication ops below. OPSD log_u may be
    # bf16/fp16, where the tails of u·log_u lose precision fast. We do NOT
    # downcast on return: the advantage is detached (no autograd cost for
    # the dtype mismatch), and `fp32_advantage * bf16_log_pi_student`
    # promotes to fp32 via PyTorch dtype-promotion, giving the loss the
    # full fp32 precision of the per-token signal. Downcasting back to
    # bf16 here would re-round and largely undo the upcast.
    log_u_clamped = torch.clamp(log_u.float(), min=-10.0, max=10.0)
    u = torch.exp(log_u_clamped)

    if divergence_type == "forward_kl":
        return -u * log_u_clamped
    if divergence_type == "jsd":
        return -0.5 * (u * log_u_clamped - (u + 1.0) * (torch.log1p(u) - _LOG2))
    if divergence_type == "improved_forward_kl":
        return u - 1.0
    if divergence_type == "improved_jsd":
        return 0.5 * (torch.log1p(u) - _LOG2)
    raise ValueError(
        f"Unknown divergence_type={divergence_type!r}. "
        f"Expected one of {DIVERGENCE_TYPES}."
    )
```

> **Dtype contract.** For `reverse_kl`, the helper returns `log_u` unchanged
> (matches the caller's dtype, bit-exact). For the other four, it returns a
> **fp32** tensor regardless of input dtype. The caller in `compute_loss`
> doesn't need to do anything — `(advantage_fp32 * student_log_probs_bf16)`
> promotes to fp32 in eager mode, the `.mean()` is fp32, and gradients flow
> through `student_log_probs` in its native dtype. Note this in the test
> for dtype behavior: assert the *returned* tensor is fp32 for non-reverse
> variants and matches `log_u.dtype` for reverse_kl.

**`OPSDTrainer.__init__` signature** (`opsd_trainer.py:123-147`): add a new
keyword arg right next to `use_thinking_machines_loss`:

```python
divergence_type: str = "reverse_kl",
```

Store it on `self` (next to `self.use_thinking_machines_loss = ...` around
line 186) and validate:

```python
if divergence_type not in DIVERGENCE_TYPES:
    raise ValueError(
        f"divergence_type={divergence_type!r} not in {DIVERGENCE_TYPES}"
    )
self.divergence_type = divergence_type
```

Also: warn (or no-op) when `use_thinking_machines_loss=False` and
`divergence_type != "reverse_kl"` — the divergence knob only affects the
sampled-token loss, not the supervised JSD branch. Recommended: log a
warning at init when this combination is set.

**`compute_loss` body** (`opsd_trainer.py:705-733`): replace the advantage
line and update the surrounding comment block.

Before (line 706-732):

```python
if self.use_thinking_machines_loss:
    # Thinking Machines uses RL-style policy gradient:
    # Advantage = log π_teacher(x) - log π_student(x)
    # Loss = -E[Advantage * log π_student(x)]
    ...
    advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()

    if shifted_labels is not None:
        mask = shifted_labels != -100
        advantage = advantage[mask]
        student_log_probs_sampled_masked = student_log_probs_sampled[mask]
    else:
        student_log_probs_sampled_masked = student_log_probs_sampled

    loss = -(advantage * student_log_probs_sampled_masked).mean()
```

After:

```python
if self.use_thinking_machines_loss:
    # RL-style policy gradient with an f-divergence per-token advantage.
    # Advantage = -g(u),  u = p_teacher / q_student  (see _compute_neg_g_u).
    # Loss = -E[Advantage * log π_student(x)]
    #
    # CRITICAL: advantage is detached. For divergence_type="reverse_kl",
    # this is bit-exact with the previous implementation.

    log_u = teacher_log_probs_sampled - student_log_probs_sampled
    advantage = _compute_neg_g_u(log_u, self.divergence_type).detach()

    if shifted_labels is not None:
        mask = shifted_labels != -100
        advantage = advantage[mask]
        log_u_masked = log_u[mask].detach()
        student_log_probs_sampled_masked = student_log_probs_sampled[mask]
    else:
        log_u_masked = log_u.detach()
        student_log_probs_sampled_masked = student_log_probs_sampled

    loss = -(advantage * student_log_probs_sampled_masked).mean()
```

Keep `log_u_masked` around so we can log:

- **`teacher_kl`** — back-compat dashboard metric. `mean(-log_u)` over
  masked tokens, the k1 reverse-KL estimator. **Always logged**, regardless
  of the active `divergence_type` (so existing dashboards keep working).
- **`teacher_pg_signal_mean/{divergence_type}`** — *signed* mean of
  `A = -g(u)` over masked tokens. Useful as a diagnostic, **not** as a
  magnitude. By construction the signed mean cancels out for
  `improved_forward_kl` (`E_q[u − 1] ≡ 0`, this is the zero-mean baseline
  that makes it the exact PG); for `improved_jsd` it concentrates near 0
  near convergence; for `forward_kl` and `jsd` it's a (signed) MC estimate
  of `-D_f`. Read this as "is my baseline behaving" rather than "how big
  is my update."
- **`teacher_pg_signal_abs_mean/{divergence_type}`** — `mean(|A|)` over
  masked tokens. This is the actual per-token signal *magnitude*. Won't
  cancel and is meaningful for every divergence type. Use this on
  dashboards when you want to see "is the student still being pushed."
- **`teacher_pg_signal_std/{divergence_type}`** — `std(A)` over masked
  tokens. The PG variance contribution; useful for spotting blown-up
  updates from extreme `u` values escaping the clamp.
- **`teacher_div_f/{divergence_type}`** — `mean(f(u))` over masked tokens.
  Only logged for the variants where this is a genuine MC estimator of the
  divergence: `reverse_kl` → `mean(-log_u)` = D_KL(q‖p) (already covered
  by `teacher_kl`; skip to avoid duplication), `forward_kl` →
  `mean(u log u)` = D_KL(p‖q), `jsd` →
  `mean(0.5 (u log u − (1+u)(log1p u − log 2)))` = D_JSD. **Skip** for
  `improved_forward_kl` and `improved_jsd` — their `g(u)` is not a
  divergence integrand.

So a JSD run logs `teacher_kl`, the three `teacher_pg_signal_{mean,abs_mean,std}/jsd`
keys, and `teacher_div_f/jsd`; an `improved_jsd` run logs `teacher_kl` plus
the three `teacher_pg_signal_*` keys (and skips `teacher_div_f/*`). Document
all five metric semantics in the trainer docstring so dashboard readers don't
conflate them.

Add these to `self._metrics["train"]` (or wherever per-step metrics live).
Check the existing logging pattern around the `_metrics` dict in `__init__`
and follow it; the JSD branch already logs per-step values that get averaged
on `log()`.

### 2. `opsd_train.py`

**`CustomScriptArguments`** (`opsd_train.py:24-108`): add a new field next to
`use_tinker_loss`:

```python
divergence_type: str = field(
    default="reverse_kl",
    metadata={
        "help": "f-divergence used for the per-token advantage when "
        "use_tinker_loss=True. One of: reverse_kl (default, matches prior "
        "behavior), forward_kl, jsd, improved_forward_kl, improved_jsd. "
        "Ignored when use_tinker_loss=False (the full-vocab JSD branch is "
        "unaffected)."
    },
)
```

**Trainer construction** (`opsd_train.py:269-285`): forward it:

```python
trainer = OPSDTrainer(
    ...,
    use_thinking_machines_loss=script_args.use_tinker_loss,
    divergence_type=script_args.divergence_type,
    ...,
)
```

**WandB config** (`opsd_train.py:172-198`): add to the logged config dict so
runs are tagged with the divergence:

```python
"divergence_type": script_args.divergence_type if script_args.use_tinker_loss else None,
```

**`run_config` autoname** (`opsd_train.py:139-155`): consider appending
`divergence_type` to the auto-generated name (e.g. `..._div-jsd`) so wandb
runs and output dirs are self-describing. Optional — skip if it bloats names.

## Backwards compatibility

- Default `divergence_type="reverse_kl"` preserves existing behavior.
- For `reverse_kl`, `_compute_neg_g_u(log_u, "reverse_kl") == log_u`, so the
  loss is bit-exact with the prior implementation (no `exp`/`clamp`/`log1p`
  on this path).
- `use_thinking_machines_loss=False` (the supervised generalized-JSD branch)
  is untouched.
- WandB key `teacher_kl` is preserved with its existing definition.

## How to use

CLI (matches the existing `scripts/` invocation style):

```bash
accelerate launch --config_file accelerate.yaml opsd_train.py \
    --use_tinker_loss True \
    --divergence_type jsd \
    --model_name_or_path Qwen/Qwen3-1.7B \
    ...
```

Programmatically:

```python
trainer = OPSDTrainer(
    ...,
    use_thinking_machines_loss=True,
    divergence_type="forward_kl",  # or "jsd", "improved_forward_kl", "improved_jsd"
)
```

## Unit tests (in scope for this pass)

The helper is small, stateless, and easy to test — and the exact-vs-biased
split plus dtype/clamping behavior are exactly the regression surfaces we'd
miss without tests. Add `tests/test_compute_neg_g_u.py` (create `tests/` if
absent) covering:

1. **Shape preservation.** For each `name in DIVERGENCE_TYPES`,
   `_compute_neg_g_u(torch.randn(B, T), name).shape == (B, T)`.
2. **Zero-at-match.** For each name, `_compute_neg_g_u(torch.zeros(B, T), name)`
   is all zeros (within fp32 atol `1e-6`). Confirms the "no gradient when
   student matches teacher" property documented above.
3. **Reverse-KL bit-exactness.** `_compute_neg_g_u(log_u, "reverse_kl")` is
   `log_u` exactly (same tensor, no dtype change). Regression guard against
   any future refactor accidentally upcasting this path.
4. **Improved-variant exact-PG sanity.** On a fixed `log_u` (no clamp
   triggered), check `improved_forward_kl` equals `exp(log_u) − 1` and
   `improved_jsd` equals `0.5 * (log1p(exp(log_u)) − log 2)` (i.e. the
   formulas in the table) to a tight tolerance. Documents what "improved"
   means and locks in the algebra.
5. **Dtype contract.** Pass a bf16 `log_u` (e.g.
   `torch.full((4,), 5.0, dtype=torch.bfloat16)`); assert:
   - `_compute_neg_g_u(log_u, "reverse_kl").dtype == torch.bfloat16` (returns
     the input tensor unchanged).
   - For each of the four non-reverse names: returned dtype is `torch.float32`
     (the helper deliberately does not downcast back to bf16 — see "Dtype
     contract" note in the helper section).
   - Values for the non-reverse variants match an fp32 reference computation
     to `atol=1e-5`, *much* tighter than what a bf16-roundtrip would allow
     (which is `~5e-2` for `u·log_u` at `log_u = 5`). This is the test that
     would fail if someone added a `.to(log_u.dtype)` on the way out.
6. **Clamping.** At `log_u = 20.0`, the four non-reverse variants return the
   value computed from `log_u_clamped = 10.0`, not from 20. Documents the
   clipped-estimator behavior so future readers don't "fix" the clamp without
   intent.
7. **Validator.** `_compute_neg_g_u(torch.zeros(4), "kl_divergence_typo")`
   raises `ValueError` mentioning `DIVERGENCE_TYPES`.

A second, smaller test file `tests/test_opsd_trainer_loss_regression.py` (or
adding to an existing trainer test, if one exists) should cover the integration
guard:

8. **Reverse-KL loss regression.** Construct a tiny `OPSDTrainer` (or call
   the loss-relevant slice as a free function if `compute_loss` is hard to
   instantiate), feed a fixed `(teacher_log_probs_sampled,
   student_log_probs_sampled, shifted_labels)` tuple, and assert
   `divergence_type="reverse_kl"` produces the same scalar loss as a
   hand-coded recomputation of the old formula on the same inputs (atol
   `0` — bit-exact). This is the strongest guarantee that the refactor is
   inert for default users.

If `compute_loss` is too entangled to test in isolation (it expects
`OPSDTrainer.__init__` to have run with `model`, `args`, etc.), factor the
post-forward arithmetic into a small free function
`_tinker_loss_from_logprobs(student_logp, teacher_logp, shifted_labels,
divergence_type)` and have `compute_loss` call it — then test the free
function. Worth doing on its own merits (improves testability of the entire
sampled-token branch).

### Test runner and dependency

The repo currently has no test harness: there is no `tests/` directory, no
`pytest.ini` / `pyproject.toml` config, and `environment.yml:10` does not list
`pytest`. Pick one of:

- **(Recommended) Add `pytest` to `environment.yml`.** One-line change:

  ```yaml
  # environment.yml
  - pip:
      - ...
      - pytest==8.3.4   # any recent 8.x is fine; pin for env reproducibility
  ```

  Tests use the standard pytest idioms (`@pytest.mark.parametrize` over
  `DIVERGENCE_TYPES`, `pytest.approx` / `torch.testing.assert_close` for
  tolerance asserts). Run with:

  ```bash
  pytest tests/ -q
  ```

  Adding pytest as a dev dependency does not affect the training environment
  (it's pip-only and only used at test time).

- **Stdlib `unittest` fallback.** If you want to avoid touching
  `environment.yml`, write the tests as `unittest.TestCase` subclasses
  (parametrize manually via subTest loops over `DIVERGENCE_TYPES`). Run with:

  ```bash
  python -m unittest discover -s tests -v
  ```

  Slightly more boilerplate, no dependency change.

Either way, add a one-line note to `README.md` under a new "Running tests"
heading so the test command is discoverable. Skip CI wiring in this pass —
this is a research repo without GitHub Actions configured today, and adding
CI is out of scope.

## Out of scope / follow-ups

- **No `kl_penalty_coef`.** Unlike the tinker_cookbook recipe, OPSD's loss
  scales the per-token signal only by `1/N_tokens` via `.mean()` — there's
  no outer coefficient knob. If divergences with different natural scales
  (e.g. JSD bounded by `log 2`) need rescaling for stability, add a
  `divergence_coef: float = 1.0` arg later; do not do it in this pass.
- **No supervised-loss extension.** `generalized_jsd_loss` already covers
  full-vocab JSD/forward/reverse KL via the `beta` parameter; the four new
  divergences here apply specifically to the sampled-token RL-style branch.
- **Temperature interaction.** As noted above, `temperature ≠ 1` rescales
  `log_u`. Worth a small ablation: does `divergence_type=jsd` with `T=1`
  behave differently than with the current default `T`?
- **Metric naming consistency between train/eval.** Decide whether
  the `teacher_pg_signal_{mean,abs_mean,std}/{type}` and
  `teacher_div_f/{type}` keys should also be emitted in eval, not just train. The current trainer's `_metrics` dict
  has `"train"` and `"eval"` keys (see `opsd_trainer.py:274`); follow the
  existing pattern.

## Implementation status

**Not started.** This is a forward-looking plan; no code edits yet. Next
step: open a `feature/f-divergences` branch off `main` on the fork
(`haoxian-chen/OPSD`) and implement in this order:

1. Factor the post-forward arithmetic in `compute_loss` into a free function
   `_tinker_loss_from_logprobs(...)` (testability prerequisite for step 5).
2. Add `_compute_neg_g_u` helper and `DIVERGENCE_TYPES` / `_LOG2` constants.
3. Wire `divergence_type` through `OPSDTrainer.__init__` (validate, store,
   warn on `use_thinking_machines_loss=False` combo).
4. Update `compute_loss` to call the new helper and log the five metrics
   (`teacher_kl`, `teacher_pg_signal_{mean,abs_mean,std}/{type}`, and
   `teacher_div_f/{type}` for the unprefixed variants only).
5. Add `pytest==8.3.4` to `environment.yml`'s pip block, create `tests/`
   directory, write `tests/test_compute_neg_g_u.py` (tests 1–7 above) and
   the loss regression test (test 8). Verify with `pytest tests/ -q`.
   These gate the PR.
6. Add `divergence_type` to `CustomScriptArguments` in `opsd_train.py` and
   thread to the trainer; extend the WandB config dict.
7. Add a short "Running tests" section to `README.md` with the `pytest`
   command.
8. Open PR against `haoxian-chen/OPSD:main` for review.
