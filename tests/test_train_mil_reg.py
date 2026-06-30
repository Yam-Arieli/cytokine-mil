"""
Unit tests for the §33 attention-collapse interventions in train_mil:
the attention-entropy penalty helper and the cell-type exclusion mask.
"""

import math

import pytest
import torch

from cytokine_mil.training.train_mil import _apply_exclude_mask, _attn_peakedness


# --- _attn_peakedness: 1 - H(a)/log(N) -------------------------------------

def test_peakedness_uniform_is_zero():
    n = 8
    a = torch.full((n,), 1.0 / n)
    assert float(_attn_peakedness(a)) == pytest.approx(0.0, abs=1e-6)


def test_peakedness_onehot_is_one():
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # H(one-hot)=0 -> peakedness = 1 (clamp keeps it finite/≈1)
    assert float(_attn_peakedness(a)) == pytest.approx(1.0, abs=1e-3)


def test_peakedness_monotonic_and_bounded():
    n = 16
    uniform = torch.full((n,), 1.0 / n)
    peaked = torch.full((n,), 0.01 / (n - 1))
    peaked[0] = 0.99
    pu, pp = float(_attn_peakedness(uniform)), float(_attn_peakedness(peaked))
    assert 0.0 <= pu < pp <= 1.0


def test_peakedness_singleton():
    assert float(_attn_peakedness(torch.tensor([1.0]))) == 0.0


def test_peakedness_differentiable():
    a = torch.tensor([0.5, 0.3, 0.2], requires_grad=True)
    _attn_peakedness(a).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


# --- _apply_exclude_mask ----------------------------------------------------

def test_exclude_drops_right_rows():
    X = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    cts = ["CD4_T", "B_cell", "CD4_T", "pDC"]
    out = _apply_exclude_mask(X, cts, {"B_cell", "pDC"})
    assert out.shape == (2, 3)
    assert torch.allclose(out, X[torch.tensor([True, False, True, False])])


def test_exclude_noop_when_all_kept():
    X = torch.zeros(3, 2)
    out = _apply_exclude_mask(X, ["NK", "NK", "CD14_Mono"], {"pDC"})
    assert out.shape == (3, 2)


def test_exclude_noop_when_all_dropped():
    X = torch.zeros(3, 2)
    out = _apply_exclude_mask(X, ["pDC", "pDC", "pDC"], {"pDC"})
    assert out.shape == (3, 2)  # never drop everything


def test_exclude_noop_on_misaligned_labels():
    X = torch.zeros(3, 2)
    out = _apply_exclude_mask(X, ["pDC", "NK"], {"pDC"})  # len mismatch
    assert out.shape == (3, 2)
