"""
Dynamics analysis helpers: compute and aggregate per-tube learning metrics.

All metrics are designed to be computed from the dynamics dict returned by
train_mil(), without requiring the model or raw data.

Aggregation must always proceed to donor level first (median across pseudo-tubes
per donor) before any cross-cytokine comparison. Effective N = 12 donors.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-tube metrics
# ---------------------------------------------------------------------------

def compute_entropy(attention_weights: torch.Tensor) -> float:
    """
    Shannon entropy of attention weights (nats).

    Low entropy -> focused, targeted pathway.
    High entropy -> broadly distributed, pleiotropic response.

    Args:
        attention_weights: (N,) tensor of non-negative weights summing to 1.
    Returns:
        Scalar entropy value.
    """
    a = attention_weights.clamp(min=1e-10)
    return float(-(a * a.log()).sum())


def compute_instance_confidence(
    attention: torch.Tensor, p_correct: float
) -> torch.Tensor:
    """
    Instance-level confidence: C_i = a_i * P(Y_correct).

    Args:
        attention: (N,) attention weights.
        p_correct: bag-level correct class probability.
    Returns:
        (N,) per-cell confidence values.
    """
    return attention * p_correct


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_to_donor_level(records: List[Dict]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Aggregate per-tube p_correct trajectories to donor level.

    For each (cytokine, donor) pair, takes the median p_correct trajectory
    across pseudo-tubes from that donor. This reduces within-donor correlation
    and yields effective N = n_unique_donors independent measurements.

    Args:
        records: List of per-tube dicts from train_mil() dynamics output.
            Each record must contain 'cytokine', 'donor', 'p_correct_trajectory'.
    Returns:
        donor_trajectories: {cytokine -> {donor -> np.array(n_logged_epochs)}}
    """
    raw: Dict[str, Dict[str, List[List[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in records:
        raw[rec["cytokine"]][rec["donor"]].append(rec["p_correct_trajectory"])

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for cytokine, donors in raw.items():
        result[cytokine] = {}
        for donor, trajectories in donors.items():
            result[cytokine][donor] = np.median(np.array(trajectories), axis=0)
    return result


def group_confidence_by_cell_type(
    confidences: np.ndarray,
    cell_type_labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Group per-cell instance confidences by cell type.

    Args:
        confidences: (N,) array of instance confidence values C_i.
        cell_type_labels: (N,) array of cell type strings.
    Returns:
        dict mapping cell_type -> np.array of confidence values for those cells.
    """
    result: Dict[str, List[float]] = defaultdict(list)
    for conf, ct in zip(confidences, cell_type_labels):
        result[str(ct)].append(float(conf))
    return {ct: np.array(vals) for ct, vals in result.items()}


# ---------------------------------------------------------------------------
# Learnability ranking
# ---------------------------------------------------------------------------

def rank_cytokines_by_learnability(
    donor_trajectories: Dict[str, Dict[str, np.ndarray]],
    exclude: Optional[List[str]] = None,
) -> List[tuple]:
    """
    Rank cytokines by learnability (area under the donor-level learning curve).

    Higher AUC -> learned earlier / more easily.
    PBS is excluded from biological interpretation but can be included for
    sanity checking by not passing it in `exclude`.

    Args:
        donor_trajectories: Output of aggregate_to_donor_level().
        exclude: Cytokine names to exclude from ranking (e.g., ['PBS']).
    Returns:
        List of (cytokine_name, mean_auc) tuples sorted descending by AUC.
    """
    exclude_set = set(exclude or [])
    scores = []
    for cytokine, donors in donor_trajectories.items():
        if cytokine in exclude_set:
            continue
        aucs = [np.trapz(traj) for traj in donors.values()]
        scores.append((cytokine, float(np.mean(aucs))))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def compute_cytokine_entropy_summary(records: List[Dict]) -> Dict[str, Dict]:
    """
    Summarise final-epoch attention entropy per cytokine.

    Returns:
        {cytokine -> {'mean': float, 'std': float, 'per_donor_median': dict}}
    """
    raw: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec["entropy_trajectory"]:
            final_entropy = rec["entropy_trajectory"][-1]
            raw[rec["cytokine"]][rec["donor"]].append(final_entropy)

    summary = {}
    for cytokine, donors in raw.items():
        donor_medians = {donor: float(np.median(vals)) for donor, vals in donors.items()}
        all_medians = list(donor_medians.values())
        summary[cytokine] = {
            "mean": float(np.mean(all_medians)),
            "std": float(np.std(all_medians)),
            "per_donor_median": donor_medians,
        }
    return summary


def build_cell_type_confidence_matrix(
    records: List[Dict],
    cell_type_obs: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build a matrix of mean instance confidence per (cytokine, cell_type).

    Requires cell_type_obs: a dict mapping tube_path -> (N,) cell_type array.
    Cell type information is re-introduced here for post-hoc analysis only —
    it was never seen by the model during training.

    Args:
        records: List of per-tube dicts from train_mil() output.
        cell_type_obs: dict mapping tube_path -> cell_type array (N,).
    Returns:
        {cytokine -> {cell_type -> np.array of mean confidences per donor}}
    """
    if cell_type_obs is None:
        return {}

    result: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in records:
        conf = rec.get("instance_confidence_mean")
        if conf is None:
            continue
        ct_labels = cell_type_obs.get(rec["tube_path"])
        if ct_labels is None:
            continue
        grouped = group_confidence_by_cell_type(conf, ct_labels)
        for ct, vals in grouped.items():
            result[rec["cytokine"]][ct].append(float(np.mean(vals)))

    return {
        cyt: {ct: np.array(vals) for ct, vals in cts.items()}
        for cyt, cts in result.items()
    }
