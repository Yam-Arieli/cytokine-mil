"""
PBS-Relative Centroid (PBS-RC) transform.

The encoder organises cells primarily by cell type. Subtracting the per-cell-type
PBS centroid removes the resting-state baseline, so cells that have not changed
state from PBS map to ~0 and cytokine-induced shifts become measurable directly.

This module owns the PBS-RC primitive used by the latent geometry analysis. It
is also re-exported from `scripts/run_experiment_geo.py` for backwards
compatibility with existing pipelines.
"""

from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def precompute_transform_means(
    cache: list,
    label_encoder,
    train_donors: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute per-cell-type embedding means needed for PBS-RC and h_residual.

    Args:
        cache: list of dicts with keys "H" (N, D) torch tensor, "label" (int),
               "cell_types" (list[str], length N), and optionally "donor" (str).
        label_encoder: object with `_idx_to_label` mapping int -> cytokine name.
        train_donors: if given, only entries with `entry["donor"] in train_donors`
                      contribute to the means. If None, all entries contribute.

    Returns:
        global_ct_means: {cell_type -> mean h_i across ALL conditions}
        pbs_ct_means:    {cell_type -> mean h_i across PBS tubes only}
    """
    if not cache:
        return {}, {}
    embed_dim = cache[0]["H"].shape[1]

    ct_sum = defaultdict(lambda: np.zeros(embed_dim, dtype=np.float64))
    ct_count = defaultdict(float)
    pbs_sum = defaultdict(lambda: np.zeros(embed_dim, dtype=np.float64))
    pbs_count = defaultdict(float)

    train_set = set(train_donors) if train_donors is not None else None

    for entry in cache:
        if train_set is not None and entry.get("donor") not in train_set:
            continue
        H_np = entry["H"].numpy().astype(np.float64)
        ct_labels = entry["cell_types"]
        cytokine = label_encoder._idx_to_label[entry["label"]]

        for i, ct in enumerate(ct_labels):
            ct_sum[ct] += H_np[i]
            ct_count[ct] += 1.0
            if cytokine == "PBS":
                pbs_sum[ct] += H_np[i]
                pbs_count[ct] += 1.0

    global_ct_means = {ct: ct_sum[ct] / ct_count[ct] for ct in ct_sum}
    pbs_ct_means = {
        ct: pbs_sum[ct] / pbs_count[ct] for ct in pbs_sum if pbs_count[ct] > 0
    }
    return global_ct_means, pbs_ct_means


def make_pbs_relative_fn(pbs_ct_means: Dict[str, np.ndarray]):
    """
    Return a transform h_i -> h_i - μ_{PBS, cell_type(i)}.

    Cells whose cell type has no PBS representation are left unchanged.
    """
    def fn(H_np: np.ndarray, ct_labels: np.ndarray) -> np.ndarray:
        result = H_np.copy()
        for i, ct in enumerate(ct_labels):
            if ct in pbs_ct_means:
                result[i] -= pbs_ct_means[ct]
        return result
    fn.__name__ = "pbs_relative"
    return fn


def make_hresidual_fn(global_ct_means: Dict[str, np.ndarray]):
    """
    Return a transform h_i -> h_i - μ_{global, cell_type(i)}.

    Removes the cell-type component shared across all cytokine conditions.
    """
    def fn(H_np: np.ndarray, ct_labels: np.ndarray) -> np.ndarray:
        result = H_np.copy()
        for i, ct in enumerate(ct_labels):
            if ct in global_ct_means:
                result[i] -= global_ct_means[ct]
        return result
    fn.__name__ = "h_residual"
    return fn


def compute_pbs_centroids_per_cell_type(
    cache: list,
    label_encoder,
    train_donors: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience: return only the PBS per-cell-type centroid dict.

    Equivalent to `precompute_transform_means(...)[1]`.
    """
    _, pbs = precompute_transform_means(cache, label_encoder, train_donors=train_donors)
    return pbs
