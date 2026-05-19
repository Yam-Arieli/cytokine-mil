"""
Alignment-based cytokine pair detection (no directionality).

Hypothesis: two cytokines that engage the same cellular program produce
*aligned* shifts in latent space within their relay cell type. Compute the
inner product / cosine similarity between PBS-relative centroids per cell
type per donor, then aggregate to find pairs with strongly aligned response
profiles.

This is the symmetric counterpart to the directional PBS-RC analysis in
`latent_geometry.py`. It does NOT call cascade direction — directionality
is left to downstream ablation.

Pipeline:
    cache  →  per-(C, T, d) PBS-RC centroids       (compute_per_atd_centroids)
           →  PCA projections at multiple dims     (fit_pca_projections)
           →  pair alignment scores per (metric, dim, cell_type, donor)
                                                    (compute_pair_scores)
           →  ranked top pairs                     (rank_and_format_top_pairs)

Pairs are unordered: each (A, B) with A < B alphabetically is scored once.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA


METRIC_DESCRIPTIONS = {
    "cosine": (
        "Cosine similarity of PBS-RC centroids: "
        "cos_d(A, B, T) = <v_A,T,d, v_B,T,d> / (||v_A,T,d|| * ||v_B,T,d||); "
        "pair score = max_T mean_d cos_d"
    ),
    "inner_product": (
        "Raw inner product of PBS-RC centroids: "
        "ip_d(A, B, T) = <v_A,T,d, v_B,T,d>; "
        "pair score = max_T mean_d ip_d"
    ),
}


def compute_per_atd_centroids(
    cache: list,
    label_encoder,
    pbs_ct_means: Dict[str, np.ndarray],
    train_donors: Sequence[str],
    exclude_cytokines: Iterable[str] = ("PBS",),
) -> Dict[Tuple[str, str, str], np.ndarray]:
    """Compute per-(cytokine, cell_type, donor) PBS-RC centroid vectors.

    For each tube of cytokine C, donor d, the encoder embeddings of cells of
    each cell type T are averaged. These are then pooled (mean) across tubes
    belonging to the same (C, T, d) triple. PBS-relative: subtract `µ_{PBS,T}`
    (the pooled PBS centroid for cell type T across training donors).

    Cell types absent from `pbs_ct_means` are skipped.

    Args:
        cache:           List of per-tube dicts from `build_cache` with keys
                         "H" (torch tensor (N, D)), "label" (int cytokine idx),
                         "cell_types" (list[str]), "donor" (str).
        label_encoder:   CytokineLabel — for label → name decoding.
        pbs_ct_means:    {cell_type_str: pbs_centroid (D,)} from
                         `compute_pbs_centroids`.
        train_donors:    Donor names to include. Tubes from other donors are
                         skipped (covers the val-donor case).
        exclude_cytokines: Cytokine names to exclude (default: PBS).

    Returns:
        Dict mapping (cytokine_name, cell_type, donor) -> np.ndarray of shape (D,).
    """
    train_donors = set(train_donors)
    exclude_cytokines = set(exclude_cytokines)

    # Accumulate per-tube means: {(C, T, d): list of (sum_vec, n_cells)}
    sums: Dict[Tuple[str, str, str], np.ndarray] = {}
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for entry in cache:
        donor = entry["donor"]
        if donor not in train_donors:
            continue
        cyto_name = label_encoder.decode(entry["label"])
        if cyto_name in exclude_cytokines:
            continue

        H = entry["H"].numpy() if hasattr(entry["H"], "numpy") else np.asarray(entry["H"])
        ct_labels = np.asarray(entry["cell_types"])

        for ct in np.unique(ct_labels):
            if ct not in pbs_ct_means:
                continue
            mask = ct_labels == ct
            ct_sum = H[mask].sum(axis=0)
            n = int(mask.sum())
            key = (cyto_name, ct, donor)
            if key in sums:
                sums[key] = sums[key] + ct_sum
            else:
                sums[key] = ct_sum.copy()
            counts[key] += n

    # Mean within each (C, T, d), then subtract PBS centroid for that cell type.
    centroids: Dict[Tuple[str, str, str], np.ndarray] = {}
    for key, s in sums.items():
        n = counts[key]
        if n == 0:
            continue
        mu = s / n
        ct = key[1]
        centroids[key] = mu - pbs_ct_means[ct]

    return centroids


def fit_pca_projections(
    atd_centroids: Dict[Tuple[str, str, str], np.ndarray],
    n_components_list: Sequence[int] = (6, 24),
    random_state: int = 0,
) -> Dict[int, Dict[Tuple[str, str, str], np.ndarray]]:
    """Fit one global PCA per requested dimensionality.

    The original full-dim centroids are returned under the key `D` (the
    embedding dimensionality detected from the input vectors), so callers
    can iterate over a single dict-of-dicts.

    Args:
        atd_centroids:     Output of `compute_per_atd_centroids`.
        n_components_list: PCA dimensionalities to compute.
        random_state:      PCA random seed.

    Returns:
        {n_components: {(C, T, d) -> projected_vector}}.
        Always includes the full-dim identity projection under key `D`.
    """
    keys = sorted(atd_centroids.keys())
    if not keys:
        return {}
    X = np.stack([atd_centroids[k] for k in keys], axis=0)
    full_dim = X.shape[1]

    projections: Dict[int, Dict[Tuple[str, str, str], np.ndarray]] = {}
    # Identity: full-dim no-PCA baseline.
    projections[full_dim] = {k: atd_centroids[k] for k in keys}

    for n in n_components_list:
        if n >= full_dim:
            # Skip — would duplicate the identity projection.
            continue
        pca = PCA(n_components=n, random_state=random_state)
        Z = pca.fit_transform(X)
        projections[n] = {k: Z[i] for i, k in enumerate(keys)}

    return projections


def _pair_score_single(
    vA: np.ndarray, vB: np.ndarray, metric: str
) -> float:
    if metric == "cosine":
        nA = np.linalg.norm(vA)
        nB = np.linalg.norm(vB)
        if nA == 0.0 or nB == 0.0:
            return 0.0
        return float(np.dot(vA, vB) / (nA * nB))
    if metric == "inner_product":
        return float(np.dot(vA, vB))
    raise ValueError(f"Unknown metric: {metric}")


def compute_pair_scores(
    centroids: Dict[Tuple[str, str, str], np.ndarray],
    metrics: Sequence[str] = ("cosine", "inner_product"),
) -> Tuple[
    Dict[str, Dict[Tuple[str, str], float]],
    Dict[str, Dict[Tuple[str, str], str]],
    Dict[str, Dict[Tuple[str, str, str], Dict[str, np.ndarray]]],
]:
    """Score every unordered cytokine pair under each metric.

    For each metric, each pair (A, B) with A < B alphabetically and A ≠ B,
    each cell type T, each donor d where both v_{A,T,d} and v_{B,T,d} exist:
        metric_d(A, B, T) is computed.

    Donors are aggregated by mean. Pair-level score = max over cell types
    (relay cell type = argmax). Cell types lacking any donor with both
    centroids are skipped.

    Args:
        centroids: {(C, T, d) -> vector}.
        metrics:   metric names from METRIC_DESCRIPTIONS.

    Returns:
        pair_scores: {metric -> {(A, B) -> pair_score}}.
        relay_T:     {metric -> {(A, B) -> best_cell_type}}.
        full_table:  {metric -> {(A, B, T) -> {"mean_d": float,
                                               "donor_scores": np.ndarray}}}.
    """
    # Index centroids by (cyto, ct) -> {donor: vec}
    by_cyto_ct: Dict[Tuple[str, str], Dict[str, np.ndarray]] = defaultdict(dict)
    for (c, t, d), v in centroids.items():
        by_cyto_ct[(c, t)][d] = v

    # Catalog of all cytokines and cell types present
    cytokines = sorted({c for (c, _) in by_cyto_ct})
    cell_types = sorted({t for (_, t) in by_cyto_ct})

    pair_scores: Dict[str, Dict[Tuple[str, str], float]] = {m: {} for m in metrics}
    relay_T: Dict[str, Dict[Tuple[str, str], str]] = {m: {} for m in metrics}
    full_table: Dict[str, Dict[Tuple[str, str, str], Dict[str, np.ndarray]]] = {
        m: {} for m in metrics
    }

    for i, a in enumerate(cytokines):
        for b in cytokines[i + 1:]:
            for metric in metrics:
                best_score = -np.inf
                best_ct = None
                any_data = False
                for t in cell_types:
                    da = by_cyto_ct.get((a, t), {})
                    db = by_cyto_ct.get((b, t), {})
                    shared_donors = sorted(set(da) & set(db))
                    if not shared_donors:
                        continue
                    donor_scores = np.array([
                        _pair_score_single(da[d], db[d], metric)
                        for d in shared_donors
                    ], dtype=np.float64)
                    mean_d = float(donor_scores.mean())
                    full_table[metric][(a, b, t)] = {
                        "mean_d": mean_d,
                        "donor_scores": donor_scores,
                        "donors": shared_donors,
                    }
                    any_data = True
                    if mean_d > best_score:
                        best_score = mean_d
                        best_ct = t
                if any_data:
                    pair_scores[metric][(a, b)] = best_score
                    relay_T[metric][(a, b)] = best_ct

    return pair_scores, relay_T, full_table


def rank_and_format_top_pairs(
    pair_scores: Dict[Tuple[str, str], float],
    relay_T: Dict[Tuple[str, str], str],
    top_pct: float = 0.10,
) -> List[dict]:
    """Sort pairs by score descending; return top `top_pct` formatted for JSON.

    Each entry has keys: A, B, relay_cell_type, score, rank.
    """
    items = sorted(pair_scores.items(), key=lambda kv: -kv[1])
    n_total = len(items)
    n_top = max(1, int(round(n_total * top_pct)))
    top = items[:n_top]
    out = []
    for rank, ((a, b), score) in enumerate(top, 1):
        out.append({
            "A":               a,
            "B":               b,
            "relay_cell_type": relay_T.get((a, b)),
            "score":           float(score),
            "rank":            rank,
        })
    return out
