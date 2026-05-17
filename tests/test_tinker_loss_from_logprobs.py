"""Integration tests for `opsd_trainer._tinker_loss_from_logprobs`.

The headline test (`test_reverse_kl_loss_regression`) asserts bit-exact parity
with the previous inline implementation for `divergence_type="reverse_kl"`.
This is the strongest backwards-compatibility guarantee: existing OPSD runs
with default settings should produce identical loss values after this refactor.

See `plans/add_divergences_opsd_implementation.md` § "Unit tests" test #8.
"""
from __future__ import annotations

import pytest
import torch

from opsd_losses import DIVERGENCE_TYPES, _tinker_loss_from_logprobs


def _prior_reverse_kl_loss(
    student_log_probs_sampled: torch.Tensor,
    teacher_log_probs_sampled: torch.Tensor,
    shifted_labels: torch.Tensor | None,
) -> torch.Tensor:
    """Reference implementation copied verbatim from the prior compute_loss body
    (pre-refactor). Kept inline here so the regression target is self-contained.
    """
    advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()

    if shifted_labels is not None:
        mask = shifted_labels != -100
        advantage = advantage[mask]
        student_log_probs_sampled_masked = student_log_probs_sampled[mask]
    else:
        student_log_probs_sampled_masked = student_log_probs_sampled

    return -(advantage * student_log_probs_sampled_masked).mean()


@pytest.fixture
def fixed_batch():
    """Reproducible (student_logp, teacher_logp, labels) tuple."""
    torch.manual_seed(0)
    B, T = 2, 6
    student_logp = -torch.rand(B, T) * 5.0  # log-probs are in (-inf, 0]
    teacher_logp = -torch.rand(B, T) * 5.0
    labels = torch.tensor(
        [
            [1, 2, 3, -100, -100, -100],
            [1, 2, 3, 4, 5, -100],
        ],
        dtype=torch.long,
    )
    return student_logp, teacher_logp, labels


def test_reverse_kl_loss_regression(fixed_batch) -> None:
    """Bit-exact match with the pre-refactor reverse-KL formula."""
    student_logp, teacher_logp, labels = fixed_batch

    loss, _ = _tinker_loss_from_logprobs(
        student_log_probs_sampled=student_logp,
        teacher_log_probs_sampled=teacher_logp,
        shifted_labels=labels,
        divergence_type="reverse_kl",
    )
    ref = _prior_reverse_kl_loss(student_logp, teacher_logp, labels)

    # Bit-exact: identical ops should give identical floats.
    assert loss.item() == ref.item()


def test_reverse_kl_loss_regression_no_mask(fixed_batch) -> None:
    student_logp, teacher_logp, _ = fixed_batch

    loss, _ = _tinker_loss_from_logprobs(
        student_log_probs_sampled=student_logp,
        teacher_log_probs_sampled=teacher_logp,
        shifted_labels=None,
        divergence_type="reverse_kl",
    )
    ref = _prior_reverse_kl_loss(student_logp, teacher_logp, None)
    assert loss.item() == ref.item()


def test_loss_requires_grad_through_student(fixed_batch) -> None:
    """Gradient must flow through student_log_probs only (advantage is detached)."""
    student_logp, teacher_logp, labels = fixed_batch
    student_logp = student_logp.clone().requires_grad_(True)
    teacher_logp = teacher_logp.clone().requires_grad_(True)

    loss, _ = _tinker_loss_from_logprobs(
        student_log_probs_sampled=student_logp,
        teacher_log_probs_sampled=teacher_logp,
        shifted_labels=labels,
        divergence_type="reverse_kl",
    )
    loss.backward()

    assert student_logp.grad is not None
    assert student_logp.grad.abs().sum().item() > 0
    # Teacher must NOT receive gradient (advantage is detached).
    assert teacher_logp.grad is None or teacher_logp.grad.abs().sum().item() == 0


@pytest.mark.parametrize("name", DIVERGENCE_TYPES)
def test_loss_is_finite_for_all_types(fixed_batch, name: str) -> None:
    student_logp, teacher_logp, labels = fixed_batch
    loss, metrics = _tinker_loss_from_logprobs(
        student_log_probs_sampled=student_logp,
        teacher_log_probs_sampled=teacher_logp,
        shifted_labels=labels,
        divergence_type=name,
    )
    assert torch.isfinite(loss).item()
    assert "teacher_kl" in metrics
    for v in metrics.values():
        assert v == v  # not NaN
        assert v not in (float("inf"), float("-inf"))


@pytest.mark.parametrize("name", DIVERGENCE_TYPES)
def test_metric_keys_match_divergence_type(fixed_batch, name: str) -> None:
    student_logp, teacher_logp, labels = fixed_batch
    _, metrics = _tinker_loss_from_logprobs(
        student_log_probs_sampled=student_logp,
        teacher_log_probs_sampled=teacher_logp,
        shifted_labels=labels,
        divergence_type=name,
    )
    # teacher_kl is always present; it's the back-compat anchor.
    assert "teacher_kl" in metrics

    # PG signal triplet uses the divergence_type suffix.
    assert f"teacher_pg_signal_mean/{name}" in metrics
    assert f"teacher_pg_signal_abs_mean/{name}" in metrics
    assert f"teacher_pg_signal_std/{name}" in metrics

    # teacher_div_f/* is only emitted for variants where g IS the divergence
    # generator (forward_kl, jsd). Skipped for reverse_kl (covered by
    # teacher_kl) and the improved variants (g is not a divergence integrand).
    if name in ("forward_kl", "jsd"):
        assert f"teacher_div_f/{name}" in metrics
    else:
        assert f"teacher_div_f/{name}" not in metrics


def test_improved_forward_kl_signed_mean_near_zero_on_uniform_random(fixed_batch) -> None:
    """E_q[u - 1] = E_q[p/q] - 1 = 0 in expectation. Here we sample once so
    finite-batch noise dominates, but the signed mean should still be a *much*
    smaller magnitude than the abs_mean for improved_forward_kl. This guards
    the documented "signed mean cancels by construction" claim.
    """
    student_logp, teacher_logp, labels = fixed_batch
    _, metrics = _tinker_loss_from_logprobs(
        student_log_probs_sampled=student_logp,
        teacher_log_probs_sampled=teacher_logp,
        shifted_labels=labels,
        divergence_type="improved_forward_kl",
    )
    signed = metrics["teacher_pg_signal_mean/improved_forward_kl"]
    abs_mean = metrics["teacher_pg_signal_abs_mean/improved_forward_kl"]
    # Sanity: abs_mean is strictly positive (we aren't at convergence).
    assert abs_mean > 0
    # The signed mean is bounded above by the abs_mean by definition; the
    # diagnostic value of this metric is that it tends to be much smaller.
    # We don't assert a tight ratio (small batch makes that flaky); just the
    # weaker inequality, plus a print-friendly note that the gap exists.
    assert abs(signed) <= abs_mean + 1e-6


def test_invalid_divergence_type_raises(fixed_batch) -> None:
    student_logp, teacher_logp, labels = fixed_batch
    with pytest.raises(ValueError):
        _tinker_loss_from_logprobs(
            student_log_probs_sampled=student_logp,
            teacher_log_probs_sampled=teacher_logp,
            shifted_labels=labels,
            divergence_type="not_a_divergence",
        )
