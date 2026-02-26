"""
Dynamics analysis helpers: compute and aggregate per-tube learning metrics.

All metrics are designed to be computed from the dynamics dict returned by
train_mil(), without requiring the model or raw data.

Aggregation must always proceed to donor level first (median across pseudo-tubes
per donor) before any cross-cytokine comparison. Effective N = 12 donors.

Every public function that produces a ranking or summary returns a dict that
includes a 'metric_description' key stating exactly what was computed.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-tube metrics (used in analysis or called from train_mil)
# ---------------------------------------------------------------------------

def compute_entropy(attention_weights: torch.Tensor) -> float:
    """
    Shannon entropy of attention weights (nats).
    H(attention_weights) = -sum_i a_i * log(a_i).

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
    Instance-level confidence: C_i(t) = a_i(t) * P(t)(Y_correct).

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

def aggregate_to_donor_level(
    records: List[Dict],
    trajectory_key: str = "p_correct_trajectory",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Aggregate per-tube scalar trajectories to donor level.

    For each (cytokine, donor) pair, takes the median trajectory across
    pseudo-tubes from that donor. This reduces within-donor correlation
    and yields effective N = n_unique_donors independent measurements.

    Args:
        records: List of per-tube dicts from train_mil() dynamics output.
            Each record must contain 'cytokine', 'donor', and the trajectory
            specified by trajectory_key.
        trajectory_key: Key of the scalar trajectory to aggregate.
            Must be a per-tube field containing a list or 1-D array of floats
            with length n_logged_epochs.
            Supported: 'p_correct_trajectory', 'entropy_trajectory'.
    Returns:
        donor_trajectories: {cytokine -> {donor -> np.array(n_logged_epochs)}}
    """
    raw: Dict[str, Dict[str, List[List[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in records:
        traj = rec.get(trajectory_key)
        if traj is not None:
            raw[rec["cytokine"]][rec["donor"]].append(list(traj))

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
) -> Dict:
    """
    Rank cytokines by learnability (area under the donor-level learning curve).

    Higher AUC -> learned earlier / more easily.
    PBS is excluded from biological interpretation but can be included for
    sanity checking by not passing it in `exclude`.

    Args:
        donor_trajectories: Output of aggregate_to_donor_level() on
            'p_correct_trajectory'.
        exclude: Cytokine names to exclude from ranking (e.g., ['PBS']).
    Returns:
        dict with keys:
            'ranking': list of (cytokine_name, mean_auc) tuples, sorted
                descending by mean_auc.
            'metric_description': exact description of what was computed.
    """
    metric_description = (
        "AUC of mean p_correct_trajectory across pseudo-tubes, aggregated to donor level "
        "(median across pseudo-tubes per donor, then mean across donors)"
    )
    exclude_set = set(exclude or [])
    scores = []
    for cytokine, donors in donor_trajectories.items():
        if cytokine in exclude_set:
            continue
        aucs = [np.trapz(traj) for traj in donors.values()]
        scores.append((cytokine, float(np.mean(aucs))))
    ranking = sorted(scores, key=lambda x: x[1], reverse=True)
    return {"ranking": ranking, "metric_description": metric_description}


# ---------------------------------------------------------------------------
# Attention entropy summary
# ---------------------------------------------------------------------------

def compute_cytokine_entropy_summary(records: List[Dict]) -> Dict:
    """
    Summarise attention entropy per cytokine.

    Uses the mean across ALL logged epochs (not just the final epoch), then
    aggregates to donor level.

    Metric: mean across epochs and pseudo-tubes of
    H(attention_weights) = -sum_i a_i * log(a_i),
    aggregated to donor level (median across pseudo-tubes per donor,
    then mean across donors).

    Args:
        records: List of per-tube dicts from train_mil() dynamics output.
            Each record must contain 'cytokine', 'donor', 'entropy_trajectory'.
    Returns:
        dict with keys:
            'summary': {cytokine -> {'mean_entropy', 'std_entropy',
                                     'per_donor_median'}}
            'metric_description': exact description of what was computed.
    """
    metric_description = (
        "mean across epochs and pseudo-tubes of "
        "H(attention_weights) = -sum_i a_i * log(a_i), "
        "aggregated to donor level (median across pseudo-tubes per donor, "
        "then mean across donors)"
    )
    raw: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        traj = rec.get("entropy_trajectory")
        if traj:
            mean_entropy = float(np.mean(traj))  # mean over all logged epochs
            raw[rec["cytokine"]][rec["donor"]].append(mean_entropy)

    summary = {}
    for cytokine, donors in raw.items():
        donor_medians = {donor: float(np.median(vals)) for donor, vals in donors.items()}
        all_medians = list(donor_medians.values())
        summary[cytokine] = {
            "mean_entropy": float(np.mean(all_medians)),
            "std_entropy": float(np.std(all_medians)),
            "per_donor_median": donor_medians,
        }
    return {"summary": summary, "metric_description": metric_description}


# ---------------------------------------------------------------------------
# Confusion entropy summary
# ---------------------------------------------------------------------------

def compute_confusion_entropy_summary(
    confusion_entropy_trajectory: Dict[str, np.ndarray],
    exclude: Optional[List[str]] = None,
) -> Dict:
    """
    Rank cytokines by confusion entropy AUC.

    Low AUC -> confusion concentrated on a few similar cytokines (similarity
    confound). High AUC -> confusion spread randomly (genuine difficulty).

    Metric: AUC of H_confusion(C,t) = -sum_{k!=C} q_k(t) * log(q_k(t)),
    where q_k(t) = ȳ_{C,k}(t) / sum_{j!=C} ȳ_{C,j}(t)
    and ȳ_C(t) is the mean softmax output across all pseudo-tubes of cytokine C.

    Args:
        confusion_entropy_trajectory: 'confusion_entropy_trajectory' from
            the train_mil() dynamics dict. Maps cytokine_name ->
            np.array(n_logged_epochs).
        exclude: Cytokine names to exclude (e.g., ['PBS']).
    Returns:
        dict with keys:
            'ranking': list of (cytokine_name, auc) tuples, sorted descending.
            'metric_description': exact description of what was computed.
    """
    metric_description = (
        "AUC of H_confusion(C,t) = -sum_{k!=C} q_k(t) * log(q_k(t)), "
        "where q_k(t) is the renormalized off-diagonal mean softmax score "
        "across all pseudo-tubes of cytokine C at epoch t. "
        "Low = confusion concentrated on similar cytokines (similarity confound). "
        "High = confusion spread randomly (genuine difficulty)."
    )
    exclude_set = set(exclude or [])
    scores = []
    for cytokine, traj in confusion_entropy_trajectory.items():
        if cytokine in exclude_set:
            continue
        scores.append((cytokine, float(np.trapz(traj))))
    ranking = sorted(scores, key=lambda x: x[1], reverse=True)
    return {"ranking": ranking, "metric_description": metric_description}


# ---------------------------------------------------------------------------
# Instance confidence: cell-type cascade profiles
# ---------------------------------------------------------------------------

def build_cell_type_confidence_matrix(
    records: List[Dict],
    cell_type_obs: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """
    Build AUC-based cell-type cascade profiles per cytokine.

    For each (cytokine, cell_type) pair:
      1. Per tube: mean C_i(t) = a_i(t) * P(t)(Y_correct) across cells of
         that type -> (n_logged_epochs,) trajectory.
      2. Per donor: median trajectory across pseudo-tubes of that donor ->
         (n_logged_epochs,).
      3. AUC of the donor-level trajectory, then mean AUC across donors.

    Cell type information is re-introduced here for post-hoc analysis only —
    it was never seen by the model during training.

    Metric: AUC of mean C_i(t) = a_i(t) * P(t)(Y_correct), averaged across
    cells of the same type within each pseudo-tube, then aggregated to donor
    level (median per donor), then mean AUC across donors.

    Args:
        records: List of per-tube dicts from train_mil() output.
            Each record must contain 'instance_confidence_trajectory':
            np.ndarray of shape (n_cells, n_logged_epochs).
        cell_type_obs: dict mapping tube_path -> (N,) array of cell type strings.
    Returns:
        dict with keys:
            'profiles': {cytokine -> {cell_type -> {
                'mean_donor_auc': float,
                'std_donor_auc': float,
                'per_donor_auc': list[float],
            }}}
            'metric_description': exact description of what was computed.
    """
    metric_description = (
        "AUC of mean C_i(t) = a_i(t) * P(t)(Y_correct), averaged across cells "
        "of the same type within each pseudo-tube, then aggregated to donor level "
        "(median across pseudo-tubes per donor), then mean AUC across donors"
    )
    if cell_type_obs is None:
        return {"profiles": {}, "metric_description": metric_description}

    # {cytokine -> {cell_type -> {donor -> [per-tube mean trajectory, ...]}}}
    raw: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for rec in records:
        conf = rec.get("instance_confidence_trajectory")
        if conf is None:
            continue
        ct_labels = cell_type_obs.get(rec["tube_path"])
        if ct_labels is None:
            continue
        # conf: (n_cells, n_logged_epochs)
        for ct in np.unique(ct_labels):
            mask = ct_labels == ct
            mean_traj = conf[mask].mean(axis=0)  # (n_logged_epochs,)
            raw[rec["cytokine"]][str(ct)][rec["donor"]].append(mean_traj)

    profiles: Dict[str, Dict[str, Dict]] = {}
    for cytokine, cell_types in raw.items():
        profiles[cytokine] = {}
        for ct, donors in cell_types.items():
            donor_aucs = []
            for _donor, trajs in donors.items():
                donor_traj = np.median(np.stack(trajs), axis=0)
                donor_aucs.append(float(np.trapz(donor_traj)))
            profiles[cytokine][ct] = {
                "mean_donor_auc": float(np.mean(donor_aucs)),
                "std_donor_auc": float(np.std(donor_aucs)),
                "per_donor_auc": donor_aucs,
            }
    return {"profiles": profiles, "metric_description": metric_description}
