"""Sampled-token (Tinker-style) f-divergence loss helpers for OPSD.

Kept in its own module (with only `math` and `torch` as dependencies) so it
can be unit-tested without importing the full trainer stack (which pulls in
accelerate, transformers, trl, vllm, etc.). `opsd_trainer.OPSDTrainer.compute_loss`
imports `_tinker_loss_from_logprobs` from here.

See `plans/add_divergences_opsd_implementation.md` for math and design notes.
"""
from __future__ import annotations

import math

import torch

DIVERGENCE_TYPES = (
    "reverse_kl",
    "forward_kl",
    "jsd",
    "improved_reverse_kl",
    "improved_forward_kl",
    "improved_jsd",
)
_LOG2 = math.log(2.0)


def _compute_neg_g_u(log_u: torch.Tensor, divergence_type: str) -> torch.Tensor:
    """Return the per-token policy-gradient advantage A = -g(u) for the chosen f-divergence.

    u = p_teacher / q_student;  log_u = log p_teacher - log q_student.

    Dtype contract:
      - "reverse_kl" returns log_u unchanged (bit-exact with the prior code path).
      - All other variants return a float32 tensor regardless of input dtype.
        OPSD log_u typically arrives in bf16/fp16; we upcast for exp/log1p/
        multiplication and deliberately do NOT downcast on return, so the
        detached advantage retains full precision when multiplied against
        the (possibly bf16) student log-probs in the loss expression.
    """
    if divergence_type == "reverse_kl":
        return log_u
    if divergence_type == "improved_reverse_kl":
        return log_u - 1.0  

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


def _tinker_loss_from_logprobs(
    student_log_probs_sampled: torch.Tensor,
    teacher_log_probs_sampled: torch.Tensor,
    shifted_labels: torch.Tensor | None,
    divergence_type: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the sampled-token (Tinker-style) RL distillation loss.

    Loss = -E[A * log pi_student], where A = -g(u) is the per-token
    advantage (detached) returned by _compute_neg_g_u for the chosen
    f-divergence. For divergence_type="reverse_kl" this is bit-exact with
    the prior implementation that used `advantage = log_u`.

    Returns:
        (loss, metrics) where metrics is a dict of scalar floats:
          - teacher_kl: mean(-log_u) over masked tokens (k1 reverse-KL estimator,
            independent of divergence_type, for dashboard back-compat).
          - teacher_pg_signal_mean/{type}: signed mean of A (diagnostic; cancels
            by construction for improved_forward_kl).
          - teacher_pg_signal_abs_mean/{type}: mean(|A|) -- the real magnitude.
          - teacher_pg_signal_std/{type}: std(A) -- PG variance.
          - teacher_div_f/{type}: mean(g(u)) -- only emitted for forward_kl and
            jsd, where g IS the f-divergence generator and the mean is a genuine
            MC estimate of the divergence value. Skipped for the improved
            variants (where g is a PG control-variate form, not a divergence
            integrand) and for reverse_kl (already covered by teacher_kl).
    """
    if divergence_type not in DIVERGENCE_TYPES:
        raise ValueError(
            f"divergence_type={divergence_type!r} not in {DIVERGENCE_TYPES}"
        )

    log_u = teacher_log_probs_sampled - student_log_probs_sampled
    advantage = _compute_neg_g_u(log_u, divergence_type).detach()

    if shifted_labels is not None:
        mask = shifted_labels != -100
        advantage_masked = advantage[mask]
        log_u_masked = log_u[mask].detach()
        student_log_probs_masked = student_log_probs_sampled[mask]
    else:
        advantage_masked = advantage
        log_u_masked = log_u.detach()
        student_log_probs_masked = student_log_probs_sampled

    loss = -(advantage_masked * student_log_probs_masked).mean()

    metrics: dict[str, float] = {}
    if log_u_masked.numel() > 0:
        metrics["teacher_kl"] = (-log_u_masked).mean().item()

        adv_f32 = advantage_masked.float()
        metrics[f"teacher_pg_signal_mean/{divergence_type}"] = adv_f32.mean().item()
        metrics[f"teacher_pg_signal_abs_mean/{divergence_type}"] = adv_f32.abs().mean().item()
        if adv_f32.numel() > 1:
            metrics[f"teacher_pg_signal_std/{divergence_type}"] = adv_f32.std().item()

        if divergence_type in ("forward_kl", "jsd"):
            log_u_f32 = log_u_masked.float()
            log_u_clamped = torch.clamp(log_u_f32, min=-10.0, max=10.0)
            u = torch.exp(log_u_clamped)
            if divergence_type == "forward_kl":
                f_u = u * log_u_clamped
            else:
                f_u = 0.5 * (u * log_u_clamped - (u + 1.0) * (torch.log1p(u) - _LOG2))
            metrics[f"teacher_div_f/{divergence_type}"] = f_u.mean().item()

    return loss, metrics
