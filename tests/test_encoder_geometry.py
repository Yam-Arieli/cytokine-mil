"""Positive and negative controls for the gene->embedding geometry statistic.

A flat result from `probe_encoder_gene_geometry.py` would be a real finding -- three of the
last four sweeps came back flat -- but only if the statistic can detect a collapse when one
is genuinely present. These tests plant a known-rank bottleneck and check it is recovered,
then check a full-rank encoder is not reported as collapsed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cascadir.models import InstanceEncoder  # noqa: E402
from probe_encoder_gene_geometry import (  # noqa: E402
    geometry_stats, infer_dims, jacobian_matrix, onehot_matrix, spearman, w1_matrix,
)

G, EMBED, HIDDEN = 60, 16, (24, 20)
PLANTED_RANK = 3


def _encoder(seed: int = 0):
    torch.manual_seed(seed)
    return InstanceEncoder(input_dim=G, embed_dim=EMBED, n_cell_types=4,
                           hidden_dims=HIDDEN).eval()


def _plant_rank(enc, rank: int = PLANTED_RANK):
    """Force the input projection to rank `rank` with equal singular values.

    Equal singular values make the participation ratio land on `rank` exactly, so the
    assertion below is about the statistic and not about spectrum shape.
    """
    g = torch.Generator().manual_seed(7)
    U, _ = torch.linalg.qr(torch.randn(HIDDEN[0], rank, generator=g))
    V, _ = torch.linalg.qr(torch.randn(G, rank, generator=g))
    with torch.no_grad():
        enc.input_proj[0].weight.copy_(U @ V.T)
    return enc


def test_infer_dims_roundtrip():
    enc = _encoder()
    dims = infer_dims(enc.state_dict())
    assert dims == {"input_dim": G, "hidden_dims": HIDDEN, "embed_dim": EMBED,
                    "n_cell_types": 4}


@pytest.mark.parametrize("rank", [3, 8])
def test_planted_bottleneck_is_recovered_by_w1_probe(rank):
    """POSITIVE CONTROL: a rank-r input projection must read as PR ~= r."""
    enc = _plant_rank(_encoder(), rank)
    pr = geometry_stats(w1_matrix(enc))[0]["raw_pr"]
    assert rank - 0.05 <= pr <= rank + 0.05, f"planted rank {rank} read as {pr}"


@pytest.mark.parametrize("rank", [3, 8])
def test_planted_bottleneck_propagates_to_the_jacobian(rank):
    """dE/dx factors through W1, so a rank-r W1 caps the Jacobian's gene-space rank."""
    enc = _plant_rank(_encoder(), rank)
    pr = geometry_stats(jacobian_matrix(enc, torch.rand(G)), want_cos=False)[0]["raw_pr"]
    assert pr <= rank + 0.05, f"rank-{rank} W1 but Jacobian PR = {pr}"


def test_statistic_is_graded_and_untrained_is_not_called_collapsed():
    """The metric must ORDER bottlenecks, not just detect the planted one.

    Note the absolute scale is deliberately not asserted: an untrained encoder is a product
    of several random matrices, whose spectrum concentrates on its own, so even a genuinely
    full-rank map reads well below its rank. That is why the real run carries a matched
    untrained control -- 'PR = 47 of 512' means nothing without it.
    """
    x = torch.rand(G)

    def jac_pr(enc):
        return geometry_stats(jacobian_matrix(enc, x), want_cos=False)[0]["raw_pr"]

    pr3 = jac_pr(_plant_rank(_encoder(), 3))
    pr8 = jac_pr(_plant_rank(_encoder(), 8))
    pr_free = jac_pr(_encoder())
    assert pr3 < pr8 < pr_free, f"not monotone in planted rank: {pr3}, {pr8}, {pr_free}"
    assert pr_free > 1.3 * pr3, f"untrained ({pr_free}) barely above rank-3 ({pr3})"
    assert geometry_stats(w1_matrix(_encoder()))[0]["raw_pr"] > 3 * PLANTED_RANK


def test_onehot_subtracts_the_shared_bias():
    """E(0) must be removed, or every gene inherits the same offset."""
    enc = _encoder()
    Z, h0 = onehot_matrix(enc, "cpu", batch=16)
    assert Z.shape == (G, EMBED) and h0.shape == (EMBED,)
    with torch.no_grad():
        e5 = torch.zeros(1, G)
        e5[0, 5] = 1.0
        expect = (enc(e5)[0] - enc(torch.zeros(1, G))[0]).numpy()
    assert np.allclose(Z[5], expect, atol=1e-5)


def test_geometry_stats_shapes_and_bounds():
    enc = _encoder()
    stats, per_gene = geometry_stats(w1_matrix(enc))
    assert per_gene["norm"].shape == (G,) and per_gene["cos_u1"].shape == (G,)
    assert 0.0 <= stats["mean_abs_cos"] <= 1.0
    assert 0.0 <= stats["raw_var_top1"] <= stats["raw_var_top5"] <= 1.0
    assert 0.0 <= stats["norm_gini"] <= 1.0


@pytest.mark.parametrize("rho_expected", [1.0, -1.0])
def test_spearman_endpoints(rho_expected):
    a = np.arange(20, dtype=float)
    b = a * rho_expected
    assert spearman(a, b) == pytest.approx(rho_expected)
