"""Tests for cascadir.signatures — Integrated Gradients correctness."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cascadir import build_pseudotubes, integrated_gradients, preprocess
from cascadir.exceptions import SignatureError
from cascadir.signatures import derive_signature


class _LinearBag(nn.Module):
    """Bag model whose positive logit is linear in X: logit0 = sum(X * w)."""

    def __init__(self, w: torch.Tensor) -> None:
        super().__init__()
        self.w = w

    def forward(self, X: torch.Tensor):
        s = (X * self.w).sum()
        logits = torch.stack([s, torch.zeros_like(s)])
        return logits, None, None


def test_ig_completeness():
    """IG should satisfy completeness: sum(IG) == f(X) - f(baseline) for a linear f."""
    g = 8
    torch.manual_seed(0)
    w = torch.randn(g)
    model = _LinearBag(w)
    X = torch.randn(5, g)
    baseline = torch.zeros(5, g)
    ig = integrated_gradients(
        model, X, target_class=0, baseline=baseline, n_steps=50
    )
    f0 = lambda Z: (Z * w).sum()  # noqa: E731
    assert torch.allclose(ig.sum(), f0(X) - f0(baseline), atol=1e-4)
    assert ig.shape == X.shape


def test_derive_signature_missing_condition_raises(synthetic_adata):
    proc = preprocess(synthetic_adata, assume="raw")
    ts = build_pseudotubes(
        proc,
        condition_col="cytokine",
        donor_col="donor",
        celltype_col="cell_type",
        control_label="PBS",
        n_per_cell_type=15,
        min_cells=8,
        n_tubes=2,
        seed=0,
    )
    # condition not present -> SignatureError before the model is ever used
    with pytest.raises(SignatureError):
        derive_signature(model=None, tube_set=ts, condition="NotAThing", device="cpu")
