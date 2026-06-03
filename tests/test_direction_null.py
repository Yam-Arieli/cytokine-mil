"""
Unit tests for cytokine_mil.analysis.direction_null (CLAUDE.md §27.2).

Covers:
  * observed cross_median exactly matches pathway_audit.directional_asymmetry_test
  * antisymmetry: swapping (A,B) + (S_a,S_b) flips the sign
  * p_emp in [0,1]; empty signature -> NaN result
  * bh_fdr against a hand-computed example; storey_pi0 limits
"""

import numpy as np
import pytest

from cytokine_mil.analysis.direction_null import (
    direction_permutation_test,
    bh_fdr,
    storey_pi0,
)
from cytokine_mil.analysis.pathway_audit import directional_asymmetry_test


def _toy_cells(seed=0, n_genes=20, n_cells=15):
    """Two cell types, cytokines A/B/PBS, deterministic random expression.

    Plant a mild A->B asymmetry: A-cells get a bump on the S_b gene block
    (idx_b) so cross_asym should lean positive (A engages B's signature).
    """
    rng = np.random.default_rng(seed)
    idx_a = np.arange(0, 5)
    idx_b = np.arange(5, 10)
    cells_by_pair = {}
    for T in ("T0", "T1"):
        cA = rng.gamma(1.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        cB = rng.gamma(1.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        cP = rng.gamma(1.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        cA[:, idx_b] += 0.8  # A-cells carry B's program (cascade signal)
        cells_by_pair[("A", T)] = cA
        cells_by_pair[("B", T)] = cB
        cells_by_pair[("PBS", T)] = cP
    return cells_by_pair, idx_a, idx_b


def test_observed_matches_pathway_audit():
    cells, idx_a, idx_b = _toy_cells(seed=1)
    # reference: §24 directional_asymmetry_test, cross_asym = sA_PB_norm - sB_PA_norm
    df = directional_asymmetry_test(
        cells_by_pair=cells,
        pathway_idx_dict={"A": idx_a, "B": idx_b},
        A="A", B="B", P_A="A", P_B="B",
        pbs_label="PBS", min_cells=10,
    )
    ref_cross = float(np.median((df["sA_PB_norm"] - df["sB_PA_norm"]).to_numpy()))

    res = direction_permutation_test(
        cells, idx_a, idx_b, "A", "B", pbs_label="PBS", min_cells=10, n_perm=0,
    )
    assert res["dir_n_celltypes"] == 2
    assert np.isclose(res["dir_observed_cross_median"], ref_cross, atol=1e-6)
    # planted A->B asymmetry => positive cross_asym
    assert res["dir_observed_cross_median"] > 0


def test_antisymmetry():
    cells, idx_a, idx_b = _toy_cells(seed=2)
    ab = direction_permutation_test(
        cells, idx_a, idx_b, "A", "B", min_cells=10, n_perm=0,
    )["dir_observed_cross_median"]
    ba = direction_permutation_test(
        cells, idx_b, idx_a, "B", "A", min_cells=10, n_perm=0,
    )["dir_observed_cross_median"]
    assert np.isclose(ab, -ba, atol=1e-6)


def test_pvalue_in_range_and_planted_signal_significant():
    cells, idx_a, idx_b = _toy_cells(seed=3)
    rng = np.random.default_rng(123)
    res = direction_permutation_test(
        cells, idx_a, idx_b, "A", "B", min_cells=10, n_perm=500, rng=rng,
    )
    assert 0.0 <= res["dir_p_emp"] <= 1.0
    assert res["dir_n_perms"] == 500
    # the planted +0.8 bump on idx_b in A-cells is a strong asymmetry
    assert res["dir_p_emp"] < 0.05


def test_empty_signature_returns_nan():
    cells, idx_a, idx_b = _toy_cells(seed=4)
    res = direction_permutation_test(
        cells, np.array([], dtype=int), idx_b, "A", "B", n_perm=100,
    )
    assert np.isnan(res["dir_observed_cross_median"])
    assert res["dir_n_perms"] == 0


def test_no_qualifying_celltype_returns_nan():
    cells, idx_a, idx_b = _toy_cells(seed=5, n_cells=3)  # < min_cells
    res = direction_permutation_test(
        cells, idx_a, idx_b, "A", "B", min_cells=10, n_perm=100,
    )
    assert res["dir_n_celltypes"] == 0
    assert np.isnan(res["dir_observed_cross_median"])


def test_bh_fdr_hand_example():
    # p = c/5 * rank gives all q == c when ranks are 1..5 with equal spacing
    p = np.array([0.04, 0.01, 0.03, 0.05, 0.02])
    q = bh_fdr(p)
    assert np.allclose(q, 0.05, atol=1e-9)
    assert q.shape == p.shape


def test_bh_fdr_nan_excluded_and_monotone():
    p = np.array([0.001, 0.2, np.nan, 0.04, 0.5])
    q = bh_fdr(p)
    assert np.isnan(q[2])                     # NaN stays NaN
    # m = 4 valid; monotone in p among the valid ones
    valid = ~np.isnan(p)
    order = np.argsort(p[valid])
    qv = q[valid][order]
    assert np.all(np.diff(qv) >= -1e-12)
    assert np.all((qv >= 0) & (qv <= 1))


def test_storey_pi0_limits():
    rng = np.random.default_rng(0)
    unif = rng.uniform(0, 1, size=2000)
    assert storey_pi0(unif, lam=0.5) > 0.85          # ~all null
    tiny = np.full(200, 0.001)
    assert storey_pi0(tiny, lam=0.5) < 0.05          # ~all signal
    assert np.isnan(storey_pi0(np.array([np.nan, np.nan])))
