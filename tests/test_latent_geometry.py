"""
Tests for the refined PBS-RC latent geometry pipeline.

Exercises the per-donor bias + Wilcoxon significance path that replaces the
deprecated antisymmetric `bias - bias` subtraction.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pytest
import torch

from cytokine_mil.analysis.latent_geometry import (
    compute_directional_bias_per_donor,
    test_directional_significance as run_significance_tests,
    build_latent_cascade_graph_from_calls,
)
from cytokine_mil.analysis.pbs_rc import (
    compute_pbs_centroids_per_cell_type,
    make_pbs_relative_fn,
    precompute_transform_means,
)


class _FakeLabelEncoder:
    def __init__(self, cytokines: List[str]):
        self.cytokines = cytokines
        self._label_to_idx = {c: i for i, c in enumerate(cytokines)}
        self._idx_to_label = {i: c for c, i in self._label_to_idx.items()}

    def encode(self, name: str) -> int:
        return self._label_to_idx[name]


def _make_cache(
    cytokine_means: Dict[str, np.ndarray],
    cell_type_offsets: Dict[str, np.ndarray],
    donors: List[str],
    label_encoder,
    pbs_baseline: np.ndarray,
    cells_per_type: int = 8,
    extra_per_donor_jitter: float = 0.0,
    seed: int = 0,
) -> List[dict]:
    """
    Build a synthetic in-memory cache with the same shape as build_cache() output.

    Each tube contains `cells_per_type` cells of every cell type. A cell of
    type T inside a cytokine-A tube has embedding:
        h = pbs_baseline + cell_type_offsets[T] + cytokine_means[A]
            + small jitter
    so PBS-RC subtraction (per cell type) leaves cytokine_means[A] + jitter.
    """
    rng = np.random.default_rng(seed)
    cell_types = list(cell_type_offsets.keys())
    cache = []
    for donor in donors:
        donor_jitter = rng.normal(0, extra_per_donor_jitter,
                                  size=pbs_baseline.shape).astype(np.float64)
        for cyt, mu_cyt in cytokine_means.items():
            ct_labels = []
            rows = []
            for ct in cell_types:
                for _ in range(cells_per_type):
                    base = (pbs_baseline + cell_type_offsets[ct] + mu_cyt
                            + donor_jitter)
                    rows.append(base + rng.normal(0, 1e-3, size=base.shape))
                    ct_labels.append(ct)
            H = np.stack(rows).astype(np.float32)
            cache.append({
                "H": torch.from_numpy(H),
                "label": label_encoder.encode(cyt),
                "cell_types": ct_labels,
                "donor": donor,
            })
    return cache


# ---------------------------------------------------------------------------
# pbs_rc primitives
# ---------------------------------------------------------------------------

def test_pbs_centroids_zero_pbs_in_pbs_rc_space():
    """After PBS-RC subtraction, PBS cells should map to ~0 per cell type."""
    embed_dim = 6
    cytokine_means = {
        "PBS": np.zeros(embed_dim),
        "IL-2": np.array([1.0, 0, 0, 0, 0, 0]),
    }
    cell_type_offsets = {
        "NK": np.array([0, 2.0, 0, 0, 0, 0]),
        "CD4_T": np.array([0, 0, 2.0, 0, 0, 0]),
    }
    pbs_baseline = np.array([5.0, 5.0, 5.0, 0, 0, 0])
    le = _FakeLabelEncoder(["PBS", "IL-2"])
    cache = _make_cache(
        cytokine_means, cell_type_offsets, donors=["D1", "D2", "D3"],
        label_encoder=le, pbs_baseline=pbs_baseline, cells_per_type=10,
    )

    pbs_ct = compute_pbs_centroids_per_cell_type(cache, le, train_donors=["D1", "D2"])
    pbs_fn = make_pbs_relative_fn(pbs_ct)

    # For each PBS tube, after subtraction the per-cell-type mean should be ~0.
    for entry in cache:
        if le._idx_to_label[entry["label"]] != "PBS":
            continue
        if entry["donor"] not in {"D1", "D2"}:
            continue
        H_pbs = pbs_fn(entry["H"].numpy().astype(np.float64),
                       np.array(entry["cell_types"]))
        for ct in set(entry["cell_types"]):
            mask = np.array(entry["cell_types"]) == ct
            mean = H_pbs[mask].mean(axis=0)
            assert np.linalg.norm(mean) < 0.05, f"PBS-RC mean not ~0 for {ct}"


def test_precompute_transform_means_train_donor_filter():
    """train_donors should restrict which entries contribute to the means."""
    le = _FakeLabelEncoder(["PBS"])
    embed_dim = 4
    cache = [
        {"H": torch.zeros(2, embed_dim), "label": 0,
         "cell_types": ["NK", "NK"], "donor": "D1"},
        {"H": torch.full((2, embed_dim), 100.0), "label": 0,
         "cell_types": ["NK", "NK"], "donor": "D2"},
    ]
    _, pbs = precompute_transform_means(cache, le, train_donors=["D1"])
    assert np.allclose(pbs["NK"], np.zeros(embed_dim))
    _, pbs_all = precompute_transform_means(cache, le, train_donors=None)
    assert np.allclose(pbs_all["NK"], np.full(embed_dim, 50.0))


# ---------------------------------------------------------------------------
# Per-donor bias structure
# ---------------------------------------------------------------------------

def test_per_donor_bias_shapes_and_keys():
    embed_dim = 6
    cytokine_means = {
        "PBS": np.zeros(embed_dim),
        "A":   np.array([1.0, 0, 0, 0, 0, 0]),
        "B":   np.array([0, 1.0, 0, 0, 0, 0]),
    }
    cell_type_offsets = {
        "T1": np.array([0, 0, 1.0, 0, 0, 0]),
        "T2": np.array([0, 0, 0, 1.0, 0, 0]),
    }
    pbs_baseline = np.array([10.0, 10.0, 0, 0, 0, 0])
    le = _FakeLabelEncoder(["PBS", "A", "B"])
    train = ["D1", "D2", "D3", "D4"]
    cache = _make_cache(cytokine_means, cell_type_offsets, donors=train + ["D5"],
                        label_encoder=le, pbs_baseline=pbs_baseline)
    pbs_ct = compute_pbs_centroids_per_cell_type(cache, le, train_donors=train)

    out = compute_directional_bias_per_donor(
        cache, le, pbs_ct, train_donors=train, direction_mode="global",
    )
    assert sorted(out["donors"]) == train
    # b_per_donor is keyed by (cytokine, cell_type)
    assert ("A", "T1") in out["b_per_donor"]
    assert set(out["b_per_donor"][("A", "T1")].keys()) == set(train)
    # PBS-RC centroids should reflect cytokine_means (≈) since PBS baseline is removed.
    assert out["centroids"]["A"][0] > 0.5


# ---------------------------------------------------------------------------
# Wilcoxon path: synthetic per-donor scores
# ---------------------------------------------------------------------------

def test_wilcoxon_significant_for_all_positive_donors():
    """All-positive per-donor scores should yield a significant forward p."""
    pytest.importorskip("scipy")
    from cytokine_mil.analysis.latent_geometry import _one_sided_wilcoxon_greater
    from scipy.stats import wilcoxon

    p_pos = _one_sided_wilcoxon_greater(np.array([0.4, 0.5, 0.3, 0.6, 0.7, 0.8]),
                                        wilcoxon)
    p_neg = _one_sided_wilcoxon_greater(np.array([-0.4, -0.5, -0.3, -0.6, -0.7]),
                                        wilcoxon)
    p_zero = _one_sided_wilcoxon_greater(np.zeros(8), wilcoxon)
    p_sym = _one_sided_wilcoxon_greater(np.array([0.5, -0.5, 0.4, -0.4]), wilcoxon)
    assert p_pos < 0.05
    assert p_neg > 0.5
    assert p_zero == 1.0
    assert p_sym > 0.1


# ---------------------------------------------------------------------------
# End-to-end significance + cascade calls
# ---------------------------------------------------------------------------

def test_cascade_call_when_one_direction_dominates():
    """
    Construct a synthetic cascade A -> B via relay T1: in A-tubes, T1 is shifted
    toward B's centroid. In B-tubes, T1 sits at its normal B-response (no shift
    toward A). The forward test for (A, B) should fire while the reverse for
    (B, A) should not.
    """
    pytest.importorskip("scipy")
    embed_dim = 6
    cytokine_means = {
        "PBS": np.zeros(embed_dim),
        "A":   np.array([1.0, 0, 0, 0, 0, 0]),
        "B":   np.array([0, 1.0, 0, 0, 0, 0]),
    }
    cell_type_offsets = {
        "T1": np.zeros(embed_dim),
        "T2": np.zeros(embed_dim),
    }
    pbs_baseline = np.array([5.0, 5.0, 0, 0, 0, 0])
    le = _FakeLabelEncoder(["PBS", "A", "B"])
    donors = [f"D{i}" for i in range(8)]
    cache = _make_cache(cytokine_means, cell_type_offsets, donors=donors,
                        label_encoder=le, pbs_baseline=pbs_baseline,
                        cells_per_type=12, extra_per_donor_jitter=0.05, seed=11)

    # Inject the cascade: in every A-tube, push T1's mean toward B's centroid
    # by +0.6 along axis 1 (the B direction).
    for entry in cache:
        if le._idx_to_label[entry["label"]] != "A":
            continue
        H = entry["H"].numpy().astype(np.float64)
        ct = np.array(entry["cell_types"])
        H[ct == "T1", 1] += 0.6
        entry["H"] = torch.from_numpy(H.astype(np.float32))

    pbs_ct = compute_pbs_centroids_per_cell_type(cache, le, train_donors=donors)
    bias = compute_directional_bias_per_donor(
        cache, le, pbs_ct, train_donors=donors, direction_mode="global",
    )
    sig = run_significance_tests(bias, le, alpha=0.05)

    fwd_call = sig["cascade_call"].get(("A", "B"))
    assert fwd_call == "A->B", f"expected A->B cascade, got {fwd_call}"
    assert sig["relay_T"].get(("A", "B")) == "T1"
    # The reverse direction should NOT also fire — that would mean shared.
    rev_call = sig["cascade_call"].get(("B", "A"))
    assert rev_call != "A->B"

    # Cascade graph should contain the A -> B edge.
    G = build_latent_cascade_graph_from_calls(sig, le)
    assert ("A", "B") in G.edges
    assert G.edges[("A", "B")]["relay_T"] == "T1"


def test_no_cascade_when_no_signal():
    """Pure noise donor scores should not yield any cascade call."""
    pytest.importorskip("scipy")
    embed_dim = 5
    cytokine_means = {
        "PBS": np.zeros(embed_dim),
        "A":   np.array([1.0, 0, 0, 0, 0]),
        "B":   np.array([0, 1.0, 0, 0, 0]),
    }
    cell_type_offsets = {"T1": np.zeros(embed_dim), "T2": np.zeros(embed_dim)}
    pbs_baseline = np.array([3.0, 3.0, 0, 0, 0])
    le = _FakeLabelEncoder(["PBS", "A", "B"])
    donors = [f"D{i}" for i in range(8)]
    cache = _make_cache(cytokine_means, cell_type_offsets, donors=donors,
                        label_encoder=le, pbs_baseline=pbs_baseline,
                        cells_per_type=10, extra_per_donor_jitter=0.05, seed=3)
    pbs_ct = compute_pbs_centroids_per_cell_type(cache, le, train_donors=donors)
    bias = compute_directional_bias_per_donor(
        cache, le, pbs_ct, train_donors=donors, direction_mode="global",
    )
    sig = run_significance_tests(bias, le, alpha=0.05)
    assert sig["cascade_call"].get(("A", "B")) in {"none", "shared"}


def test_forward_and_reverse_are_independent_not_antisymmetric():
    """
    Regression test: the new b_fwd and b_rev are NOT algebraic negatives of
    each other (the deprecated `bias - bias` was antisymmetric by construction).
    """
    pytest.importorskip("scipy")
    embed_dim = 5
    cytokine_means = {
        "PBS": np.zeros(embed_dim),
        "A":   np.array([1.0, 0, 0, 0, 0]),
        "B":   np.array([0, 1.0, 0, 0, 0]),
    }
    # T2 is a non-relay type; its presence lets µ_A ≠ µ_{A,T1} so the
    # centroid subtraction can isolate T1's specific cascade component.
    cell_type_offsets = {"T1": np.zeros(embed_dim), "T2": np.zeros(embed_dim)}
    pbs_baseline = np.array([3.0, 3.0, 0, 0, 0])
    le = _FakeLabelEncoder(["PBS", "A", "B"])
    donors = [f"D{i}" for i in range(6)]
    cache = _make_cache(cytokine_means, cell_type_offsets, donors=donors,
                        label_encoder=le, pbs_baseline=pbs_baseline,
                        cells_per_type=10, extra_per_donor_jitter=0.1, seed=2)

    # Asymmetric injection: only A-tubes get a T1 shift toward B.
    for entry in cache:
        if le._idx_to_label[entry["label"]] != "A":
            continue
        H = entry["H"].numpy().astype(np.float64)
        ct = np.array(entry["cell_types"])
        H[ct == "T1", 1] += 0.5
        entry["H"] = torch.from_numpy(H.astype(np.float32))

    pbs_ct = compute_pbs_centroids_per_cell_type(cache, le, train_donors=donors)
    bias = compute_directional_bias_per_donor(
        cache, le, pbs_ct, train_donors=donors, direction_mode="global",
    )
    sig = run_significance_tests(bias, le, alpha=0.05)
    # Forward: T1 in A-tubes projected toward B direction.
    # Reverse: T1 in B-tubes projected toward A direction (= b_fwd[(B, A, T1)]).
    fwd = sig["b_fwd"].get(("A", "B", "T1"))
    rev = sig["b_fwd"].get(("B", "A", "T1"))
    assert fwd is not None and rev is not None
    # fwd should be strongly positive (cascade injected into A-tubes only).
    # rev should hover near zero (no injection in B-tubes).
    # They are NOT negatives of each other (old antisymmetric formula would force -rev=fwd).
    assert fwd.mean() > 0.1
    assert abs(rev.mean()) < fwd.mean()
    assert not np.allclose(fwd, -rev)
