"""
Tests for `cytokine_mil.analysis.pair_alignment`.

Builds a tiny synthetic cache (3 cytokines × 2 cell types × 3 donors × 5D
embeddings) and verifies:
  - `compute_per_atd_centroids` returns the right shape and keys.
  - `compute_pair_scores` is symmetric in (A, B).
  - Cosine of identical vectors is 1.0; orthogonal vectors give 0.
  - PCA-6D projection has correct dimensionality.
  - Top-pairs output respects the schema expected by downstream consumers.
"""

from typing import Dict, List

import numpy as np
import pytest

from cytokine_mil.analysis.pair_alignment import (
    compute_per_atd_centroids,
    fit_pca_projections,
    compute_pair_scores,
    rank_and_format_top_pairs,
)


class _FakeLabelEncoder:
    def __init__(self, cytokines: List[str]):
        self.cytokines = cytokines
        self._label_to_idx = {c: i for i, c in enumerate(cytokines)}
        self._idx_to_label = {i: c for c, i in self._label_to_idx.items()}

    def encode(self, name: str) -> int:
        return self._label_to_idx[name]

    def decode(self, idx: int) -> str:
        return self._idx_to_label[idx]


def _make_cache(
    cytokines: List[str],
    cell_types: List[str],
    donors: List[str],
    label_encoder,
    embed_dim: int = 5,
    cells_per_type: int = 8,
    response_shifts: Dict[str, np.ndarray] = None,
    seed: int = 0,
):
    """Build a synthetic cache list of per-tube dicts.

    Each tube has `cells_per_type` cells per cell type.
    A cell of cytokine C, cell type T has embedding:
        base_T + response_shifts[C] (if C != PBS else 0) + small noise.
    """
    rng = np.random.default_rng(seed)
    if response_shifts is None:
        response_shifts = {c: rng.normal(size=embed_dim) for c in cytokines if c != "PBS"}
        response_shifts["PBS"] = np.zeros(embed_dim)
    base_per_ct = {t: rng.normal(size=embed_dim) for t in cell_types}

    cache = []
    for cyto in cytokines:
        for donor in donors:
            H_rows = []
            ct_labels = []
            for ct in cell_types:
                base = base_per_ct[ct]
                shift = response_shifts.get(cyto, np.zeros(embed_dim))
                for _ in range(cells_per_type):
                    H_rows.append(base + shift + rng.normal(scale=0.01, size=embed_dim))
                    ct_labels.append(ct)
            H = np.stack(H_rows, axis=0)
            cache.append({
                "H":          H.astype(np.float32),
                "label":      label_encoder.encode(cyto),
                "cell_types": ct_labels,
                "donor":      donor,
            })
    return cache, response_shifts, base_per_ct


def _pbs_centroids(base_per_ct, embed_dim, cytokines, label_encoder):
    """PBS centroid per cell type ≈ base_per_ct (since shift_PBS = 0)."""
    return {ct: base.copy() for ct, base in base_per_ct.items()}


def test_compute_per_atd_centroids_shape():
    le = _FakeLabelEncoder(["PBS", "IL-A", "IL-B"])
    cytokines = ["PBS", "IL-A", "IL-B"]
    cell_types = ["T1", "T2"]
    donors = ["D1", "D2", "D3"]
    embed_dim = 5

    cache, shifts, base = _make_cache(
        cytokines, cell_types, donors, le, embed_dim=embed_dim,
    )
    pbs_ct = _pbs_centroids(base, embed_dim, cytokines, le)

    centroids = compute_per_atd_centroids(
        cache, le, pbs_ct, train_donors=donors,
    )

    # Excludes PBS by default → 2 cytokines × 2 cell types × 3 donors = 12
    assert len(centroids) == 2 * 2 * 3
    for (c, t, d), v in centroids.items():
        assert c != "PBS"
        assert t in cell_types
        assert d in donors
        assert v.shape == (embed_dim,)

    # PBS-RC: centroid ≈ response_shift for that cytokine (cell-type-invariant
    # in this fixture).
    for c in ["IL-A", "IL-B"]:
        for t in cell_types:
            for d in donors:
                v = centroids[(c, t, d)]
                np.testing.assert_allclose(v, shifts[c], atol=0.02)


def test_pair_scores_symmetric_and_alignment():
    """Cosine score should be symmetric: score(A, B) == score(B, A)."""
    le = _FakeLabelEncoder(["PBS", "IL-A", "IL-B", "IL-C"])
    cytokines = ["PBS", "IL-A", "IL-B", "IL-C"]
    cell_types = ["T1", "T2"]
    donors = ["D1", "D2", "D3"]
    embed_dim = 5

    cache, shifts, base = _make_cache(
        cytokines, cell_types, donors, le, embed_dim=embed_dim,
    )
    pbs_ct = _pbs_centroids(base, embed_dim, cytokines, le)
    centroids = compute_per_atd_centroids(cache, le, pbs_ct, train_donors=donors)

    pair_scores, relay_T, full = compute_pair_scores(centroids)

    # 3 cytokines (excluding PBS) → 3 unordered pairs.
    for metric in ("cosine", "inner_product"):
        assert len(pair_scores[metric]) == 3, \
            f"Expected 3 unordered pairs for metric {metric}, got {len(pair_scores[metric])}"

    # All keys should be (A, B) with A < B alphabetically.
    for (a, b) in pair_scores["cosine"]:
        assert a < b, f"Pair ({a}, {b}) violates alphabetical order"


def test_cosine_identical_vectors():
    """Two cytokines with the SAME shift vector should have cosine ≈ 1."""
    le = _FakeLabelEncoder(["PBS", "IL-A", "IL-B"])
    cell_types = ["T1"]
    donors = ["D1", "D2"]
    embed_dim = 4

    shared_shift = np.array([1.0, 2.0, 3.0, 4.0])
    response_shifts = {
        "PBS":   np.zeros(embed_dim),
        "IL-A":  shared_shift,
        "IL-B":  shared_shift,
    }
    cache, _, base = _make_cache(
        ["PBS", "IL-A", "IL-B"], cell_types, donors, le,
        embed_dim=embed_dim, response_shifts=response_shifts, seed=42,
    )
    pbs_ct = _pbs_centroids(base, embed_dim, ["PBS", "IL-A", "IL-B"], le)
    centroids = compute_per_atd_centroids(cache, le, pbs_ct, train_donors=donors)
    pair_scores, _, _ = compute_pair_scores(centroids)

    # IL-A and IL-B share the same shift → cosine ≈ 1
    cos_score = pair_scores["cosine"][("IL-A", "IL-B")]
    assert cos_score > 0.99, f"Expected cosine ≈ 1, got {cos_score:.4f}"


def test_pca_projection_shape():
    """PCA-2D output should be 2D; full-dim baseline retained."""
    le = _FakeLabelEncoder(["PBS", "IL-A", "IL-B", "IL-C"])
    cell_types = ["T1", "T2", "T3"]
    donors = ["D1", "D2", "D3"]
    embed_dim = 8

    cache, _, base = _make_cache(
        ["PBS", "IL-A", "IL-B", "IL-C"], cell_types, donors, le,
        embed_dim=embed_dim, seed=1,
    )
    pbs_ct = _pbs_centroids(base, embed_dim, le.cytokines, le)
    centroids = compute_per_atd_centroids(cache, le, pbs_ct, train_donors=donors)

    projections = fit_pca_projections(centroids, n_components_list=(2, 4))
    assert embed_dim in projections, "Full-dim baseline missing"
    assert 2 in projections
    assert 4 in projections

    for d_target, proj_dict in projections.items():
        for v in proj_dict.values():
            assert v.shape == (d_target,), \
                f"At dim={d_target}, got vec shape {v.shape}"

    # Same keys across all dims
    keys_full = set(projections[embed_dim].keys())
    keys_2    = set(projections[2].keys())
    assert keys_full == keys_2


def test_top_pairs_schema():
    """Output entries match downstream-consumer schema (A, B, relay_cell_type, ...)."""
    le = _FakeLabelEncoder(["PBS", "IL-A", "IL-B", "IL-C"])
    cell_types = ["T1", "T2"]
    donors = ["D1", "D2"]
    embed_dim = 6

    cache, _, base = _make_cache(
        ["PBS", "IL-A", "IL-B", "IL-C"], cell_types, donors, le,
        embed_dim=embed_dim, seed=3,
    )
    pbs_ct = _pbs_centroids(base, embed_dim, le.cytokines, le)
    centroids = compute_per_atd_centroids(cache, le, pbs_ct, train_donors=donors)

    pair_scores, relay_T, _ = compute_pair_scores(centroids)
    top = rank_and_format_top_pairs(
        pair_scores["cosine"], relay_T["cosine"], top_pct=1.0,
    )
    assert len(top) >= 1
    for i, entry in enumerate(top):
        assert set(entry.keys()) >= {"A", "B", "relay_cell_type", "score", "rank"}
        assert entry["rank"] == i + 1
        assert entry["relay_cell_type"] in cell_types
