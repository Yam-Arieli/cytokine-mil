"""
Unit tests for cytokine_mil.analysis.signature_coupling (CLAUDE.md §28).

Covers:
  * engagement_per_celltype E[t,i,j] matches pathway_audit.directional_asymmetry_test
    (sA_PB_norm / sB_PA_norm) when all cell types are fully present
  * coupling = M[a,b]+M[b,a] (symmetric); cross_asym = M[a,b]-M[b,a] (antisymmetric)
  * coupling_null_p in [0,1]; planted strong coupling beats the random-gene null
"""

import numpy as np

from cytokine_mil.analysis.signature_coupling import (
    engagement_per_celltype,
    cross_engagement_matrix,
    coupling_direction,
)
from cytokine_mil.analysis.pathway_audit import directional_asymmetry_test


def _toy(seed=0, n_genes=40, n_cells=20):
    """A/B/PBS over 2 fully-present cell types; A-cells carry B's signature block
    (a planted A->B-style coupling) so M[A,B] > 0 and coupling is strong."""
    rng = np.random.default_rng(seed)
    idx_a = np.arange(0, 6)
    idx_b = np.arange(6, 12)
    cells = {}
    for T in ("T0", "T1"):
        cA = rng.gamma(1.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        cB = rng.gamma(1.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        cP = rng.gamma(1.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        cA[:, idx_b] += 0.7          # A engages B's signature -> coupling + a_to_b
        cB[:, idx_b] += 0.7          # B engages its own signature (direct)
        cells[("A", T)] = cA
        cells[("B", T)] = cB
        cells[("PBS", T)] = cP
    return cells, {"A": idx_a, "B": idx_b}, idx_a, idx_b


def test_engagement_matches_directional_asymmetry():
    cells, sig, idx_a, idx_b = _toy(seed=1)
    cyts, cts, E = engagement_per_celltype(cells, sig, pbs_label="PBS", min_cells=5)
    assert cyts == ["A", "B"]
    M = cross_engagement_matrix(E)
    ia, ib = cyts.index("A"), cyts.index("B")

    df = directional_asymmetry_test(
        cells_by_pair=cells, pathway_idx_dict={"A": idx_a, "B": idx_b},
        A="A", B="B", P_A="A", P_B="B", pbs_label="PBS", min_cells=5,
    )
    ref_ab = float(np.median(df["sA_PB_norm"].to_numpy()))   # s(A,S_B)-pbs
    ref_ba = float(np.median(df["sB_PA_norm"].to_numpy()))   # s(B,S_A)-pbs
    assert np.isclose(M[ia, ib], ref_ab, atol=1e-6)
    assert np.isclose(M[ib, ia], ref_ba, atol=1e-6)
    # antisymmetric direction matches cross_asym
    assert np.isclose(M[ia, ib] - M[ib, ia],
                      float(np.median((df["sA_PB_norm"] - df["sB_PA_norm"]).to_numpy())),
                      atol=1e-6)


def test_coupling_direction_signs_and_null():
    cells, sig, _, _ = _toy(seed=2)
    rng = np.random.default_rng(7)
    rows = coupling_direction(
        cells, sig, pbs_label="PBS", min_cells=5, n_perm=200, rng=rng,
    )
    assert len(rows) == 1
    r = rows[0]
    assert (r["axis_a"], r["axis_b"]) == ("A", "B")
    # coupling = m_ab + m_ba ; cross_asym = m_ab - m_ba
    assert np.isclose(r["coupling"], r["m_ab"] + r["m_ba"], atol=1e-9)
    assert np.isclose(r["cross_asym"], r["m_ab"] - r["m_ba"], atol=1e-9)
    # planted shared B-block => positive, strong coupling that beats the null
    assert r["coupling"] > 0
    assert 0.0 <= r["coupling_null_p"] <= 1.0
    assert r["coupling_null_p"] < 0.05
    assert r["n_celltypes"] == 2


def test_unrelated_pair_weak_coupling():
    # No planted overlap -> coupling near zero, should NOT beat the null
    rng = np.random.default_rng(3)
    n_genes = 40
    cells = {}
    idx_a, idx_b = np.arange(0, 6), np.arange(6, 12)
    for T in ("T0", "T1"):
        for c in ("A", "B", "PBS"):
            cells[(c, T)] = rng.gamma(1.0, 1.0, size=(25, n_genes)).astype(np.float32)
    rows = coupling_direction(
        cells, {"A": idx_a, "B": idx_b}, pbs_label="PBS", min_cells=5,
        n_perm=200, rng=np.random.default_rng(11),
    )
    assert rows[0]["coupling_null_p"] > 0.05   # not specifically coupled
