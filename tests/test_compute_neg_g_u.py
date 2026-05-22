"""Tests for `opsd_trainer._compute_neg_g_u` — the per-token f-divergence
advantage helper used by the Tinker-style sampled-token loss.

See `plans/add_divergences_opsd_implementation.md` § "Unit tests".
"""
from __future__ import annotations

import math

import pytest
import torch

from opsd_losses import DIVERGENCE_TYPES, _LOG2, _compute_neg_g_u

LOG2_REF = math.log(2.0)
NON_REVERSE_TYPES = tuple(t for t in DIVERGENCE_TYPES if t != "reverse_kl")
ZERO_AT_MATCH_TYPES = ("reverse_kl", "forward_kl", "jsd", "improved_jsd")
EXP_USING_TYPES = ("forward_kl", "jsd", "improved_forward_kl", "improved_jsd")


# --- Test 1: shape preservation ----------------------------------------------
@pytest.mark.parametrize("name", DIVERGENCE_TYPES)
def test_shape_preserved(name: str) -> None:
    log_u = torch.randn(4, 7)
    out = _compute_neg_g_u(log_u, name)
    assert out.shape == log_u.shape


# --- Test 2: behavior at match ----------------------------------------------
@pytest.mark.parametrize("name", ZERO_AT_MATCH_TYPES)
def test_zero_at_match(name: str) -> None:
    """A = -g(u) vanishes at u = 1 for the zero-shifted variants."""
    log_u = torch.zeros(3, 5)
    out = _compute_neg_g_u(log_u, name)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_improved_forward_kl_match_value() -> None:
    log_u = torch.zeros(3, 5)
    out = _compute_neg_g_u(log_u, "improved_forward_kl")
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)


def test_improved_reverse_kl_match_value() -> None:
    log_u = torch.zeros(3, 5)
    out = _compute_neg_g_u(log_u, "improved_reverse_kl")
    assert torch.allclose(out, -torch.ones_like(out), atol=1e-6)


# --- Test 3: reverse_kl is identity (bit-exact, no dtype change) ------------
def test_reverse_kl_returns_input_tensor() -> None:
    log_u = torch.randn(8, dtype=torch.float32)
    out = _compute_neg_g_u(log_u, "reverse_kl")
    # Same object (no copy/clamp/exp on this path).
    assert out is log_u
    assert out.dtype == torch.float32


def test_reverse_kl_bf16_returns_input_unchanged() -> None:
    log_u = torch.tensor([0.0, 1.5, -2.0], dtype=torch.bfloat16)
    out = _compute_neg_g_u(log_u, "reverse_kl")
    assert out is log_u
    assert out.dtype == torch.bfloat16


# --- Test 4: improved-variant exact-PG algebra -------------------------------
def test_improved_forward_kl_matches_exp() -> None:
    """A = u = exp(log_u) (no clamp triggered)."""
    log_u = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.5], dtype=torch.float32)
    out = _compute_neg_g_u(log_u, "improved_forward_kl")
    expected = torch.exp(log_u)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_improved_reverse_kl_matches_log_u_minus_one() -> None:
    """A = log_u - 1 (unclamped)."""
    log_u = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.5], dtype=torch.float32)
    out = _compute_neg_g_u(log_u, "improved_reverse_kl")
    expected = log_u - 1.0
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_improved_jsd_matches_log1p_form() -> None:
    """A = 0.5 * (log1p(u) - log 2)."""
    log_u = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.5], dtype=torch.float32)
    out = _compute_neg_g_u(log_u, "improved_jsd")
    u = torch.exp(log_u)
    expected = 0.5 * (torch.log1p(u) - LOG2_REF)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_forward_kl_matches_minus_u_log_u() -> None:
    log_u = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.5], dtype=torch.float32)
    out = _compute_neg_g_u(log_u, "forward_kl")
    u = torch.exp(log_u)
    expected = -u * log_u
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_jsd_matches_formula() -> None:
    log_u = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.5], dtype=torch.float32)
    out = _compute_neg_g_u(log_u, "jsd")
    u = torch.exp(log_u)
    expected = -0.5 * (u * log_u - (u + 1.0) * (torch.log1p(u) - LOG2_REF))
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


# --- Test 5: dtype contract --------------------------------------------------
@pytest.mark.parametrize("name", NON_REVERSE_TYPES)
def test_non_reverse_returns_fp32_for_bf16_input(name: str) -> None:
    """The helper deliberately does NOT downcast: bf16 in → fp32 out."""
    log_u_bf16 = torch.tensor([-2.0, 0.0, 2.0, 5.0], dtype=torch.bfloat16)
    out = _compute_neg_g_u(log_u_bf16, name)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("name", NON_REVERSE_TYPES)
def test_non_reverse_bf16_input_matches_fp32_reference(name: str) -> None:
    """Returned fp32 values agree with an fp32-from-the-start reference much
    more tightly than a bf16-roundtrip would.
    """
    log_u_fp32 = torch.tensor([-2.0, 0.0, 2.0, 5.0], dtype=torch.float32)
    log_u_bf16 = log_u_fp32.to(torch.bfloat16)

    # Reference: same call but starting from fp32.
    ref = _compute_neg_g_u(log_u_fp32, name)
    out = _compute_neg_g_u(log_u_bf16, name)

    # Differences here come from log_u_bf16 itself being a rounded version of
    # log_u_fp32 (bf16 mantissa is 7 bits), NOT from the helper's internal
    # arithmetic. Tolerance is set well below a bf16-roundtrip of the final
    # value (which would be `~5e-2` for forward_kl at log_u=5).
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


# --- Test 6: clamping --------------------------------------------------------
@pytest.mark.parametrize("name", EXP_USING_TYPES)
def test_clamping_at_extreme_log_u(name: str) -> None:
    """At log_u = 20 (outside [-10, 10]), helper uses log_u_clamped = 10."""
    log_u_extreme = torch.tensor([20.0], dtype=torch.float32)
    log_u_at_clamp = torch.tensor([10.0], dtype=torch.float32)

    out_extreme = _compute_neg_g_u(log_u_extreme, name)
    out_at_clamp = _compute_neg_g_u(log_u_at_clamp, name)

    torch.testing.assert_close(out_extreme, out_at_clamp, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", EXP_USING_TYPES)
def test_clamping_at_extreme_negative_log_u(name: str) -> None:
    log_u_extreme = torch.tensor([-20.0], dtype=torch.float32)
    log_u_at_clamp = torch.tensor([-10.0], dtype=torch.float32)

    out_extreme = _compute_neg_g_u(log_u_extreme, name)
    out_at_clamp = _compute_neg_g_u(log_u_at_clamp, name)

    torch.testing.assert_close(out_extreme, out_at_clamp, atol=1e-6, rtol=1e-6)


def test_improved_reverse_kl_is_unclamped() -> None:
    log_u = torch.tensor([20.0, -20.0], dtype=torch.float32)
    out = _compute_neg_g_u(log_u, "improved_reverse_kl")
    torch.testing.assert_close(out, torch.tensor([19.0, -21.0]))


# --- Test 7: validator -------------------------------------------------------
def test_unknown_divergence_type_raises() -> None:
    log_u = torch.zeros(4)
    with pytest.raises(ValueError) as exc_info:
        _compute_neg_g_u(log_u, "kl_divergence_typo")
    msg = str(exc_info.value)
    assert "kl_divergence_typo" in msg
    # Mentions the valid set so the user knows what to pick.
    for name in DIVERGENCE_TYPES:
        assert name in msg


# --- Sanity: _LOG2 constant matches math.log(2) ------------------------------
def test_log2_constant() -> None:
    assert _LOG2 == pytest.approx(math.log(2.0), rel=0.0, abs=0.0)
