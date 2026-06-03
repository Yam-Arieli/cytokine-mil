"""Signature-space coupling: unified with cross_asym; donor-level path runs."""

import numpy as np
import pytest

from cascadir.config import CrossAsymConfig
from cascadir.cross_asym import aggregate_direction, directional_asymmetry_test
from cascadir.signature_coupling import cross_engagement_matrix, signature_coupling
from cascadir.types import Signature


def _toy(seed=0, n_genes=20, n_cells=40):
    """A, B, PBS over 2 cell types; A's cells carry B's signature (planted coupling)."""
    rng = np.random.default_rng(seed)
    idx_a, idx_b = np.arange(0, 5), np.arange(5, 10)
    cbp = {}
    for T in ("T0", "T1"):
        cA = rng.gamma(1.0, 1.0, size=(n_cells, n_genes)).astype(np.float64)
        cB = rng.gamma(1.0, 1.0, size=(n_cells, n_genes)).astype(np.float64)
        cP = rng.gamma(1.0, 1.0, size=(n_cells, n_genes)).astype(np.float64)
        cA[:, idx_b] += 0.7   # A engages B's signature -> coupling + a_to_b
        cbp[("A", T)] = cA
        cbp[("B", T)] = cB
        cbp[("PBS", T)] = cP
    genes = tuple(f"g{i}" for i in range(n_genes))
    sig_idx = {"A": idx_a, "B": idx_b}
    sigs = {
        "A": Signature("A", tuple(genes[i] for i in idx_a), (1.0,) * 5, 5),
        "B": Signature("B", tuple(genes[i] for i in idx_b), (1.0,) * 5, 5),
    }
    return cbp, sig_idx, sigs, genes


def test_M_matches_directional_asymmetry_test():
    cbp, sig_idx, _sigs, _g = _toy(seed=1)
    conds, M = cross_engagement_matrix(cbp, sig_idx, control_label="PBS", min_cells=10)
    assert conds == ["A", "B"]
    i, j = 0, 1
    df = directional_asymmetry_test(cbp, sig_idx, "A", "B", control_label="PBS", min_cells=10)
    ref_ab = float(np.median(df["sA_PB_norm"].to_numpy()))  # = M[A,B]
    ref_ba = float(np.median(df["sB_PA_norm"].to_numpy()))  # = M[B,A]
    assert np.isclose(M[i, j], ref_ab, atol=1e-9)
    assert np.isclose(M[j, i], ref_ba, atol=1e-9)


def test_coupling_and_cross_asym_decompose_M():
    cbp, _sig_idx, sigs, genes = _toy(seed=2)
    df = signature_coupling(cbp, sigs, genes, config=CrossAsymConfig(n_null_perms=0))
    row = df.iloc[0]
    assert np.isclose(row["coupling"], row["m_ab"] + row["m_ba"], atol=1e-9)
    assert np.isclose(row["cross_asym"], row["m_ab"] - row["m_ba"], atol=1e-9)
    # planted A->B coupling => positive coupling, positive cross_asym (A is alphabetically first)
    assert row["coupling"] > 0
    assert row["cross_asym"] > 0


def test_cross_asym_matches_direction_aggregate():
    cbp, sig_idx, sigs, genes = _toy(seed=3)
    df = signature_coupling(cbp, sigs, genes)
    sc_cross = float(df.iloc[0]["cross_asym"])
    dat = directional_asymmetry_test(cbp, sig_idx, "A", "B", control_label="PBS", min_cells=10)
    direction_median = aggregate_direction(dat, column="cross_asym")[0]
    assert np.isclose(sc_cross, direction_median, atol=1e-9)


def test_null_pvalue_in_range():
    cbp, _sig_idx, sigs, genes = _toy(seed=4)
    df = signature_coupling(cbp, sigs, genes)  # default config -> n_null_perms=100
    p = df.iloc[0]["coupling_null_p"]
    assert np.isnan(p) or (0.0 <= p <= 1.0)


def test_donor_level_path_runs():
    cbp1, _si, sigs, genes = _toy(seed=5)
    cbp2, _si2, _sigs2, _g2 = _toy(seed=6)
    df = signature_coupling(
        cbp1, sigs, genes, cells_by_pair_per_donor={"d1": cbp1, "d2": cbp2}
    )
    for col in ("donor_coupling_mean", "donor_consensus", "donor_sign_p", "n_donors", "coupled"):
        assert col in df.columns
    assert int(df.iloc[0]["n_donors"]) == 2
