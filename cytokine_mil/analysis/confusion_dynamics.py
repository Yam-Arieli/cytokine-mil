"""
Cascade Inference via Confusion Dynamics.

Analyzes the structured confusion pattern between cytokines during AB-MIL training
to infer cytokine cascade relationships.

Core hypothesis: if cascade A→B exists, A-tubes contain a weak B-like signal.
The trained classifier will assign asymmetrically higher softmax mass to class B
when processing A-tubes than vice versa. This asymmetry, especially when it emerges
late in training (after direct signatures are learned), is the cascade direction signal.

Main outputs:
    - C(A, B, t): K×K×T confusion trajectory tensor
    - Asym(A→B): K×K asymmetry matrix (late-epoch mean difference)
    - Cascade graph: directed edges where asymmetry is significant after FDR correction

Functions:
    compute_confusion_trajectory
    aggregate_confusion_to_donor_level
    compute_asymmetry_score
    compute_temporal_profile
    extract_cell_type_attention_for_confusion
    build_cascade_graph
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import anndata
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_confusion_trajectory(
    records: List[dict],
    label_encoder,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the K×K×T confusion trajectory tensor from per-tube softmax trajectories.

    C[A, B, t] = mean over tubes with true label A of softmax_trajectory[B, t]

    Args:
        records: list of per-tube dicts from train_mil(), each must contain
                 'softmax_trajectory' of shape (K, n_logged_epochs).
        label_encoder: CytokineLabel or BinaryLabel with .encode() and .cytokines.

    Returns:
        confusion: np.ndarray of shape (K, K, T).
                   Row = true class, Col = predicted class.
        cytokine_names: list of length K mapping index → cytokine name.
    """
    cytokine_names = list(label_encoder.cytokines)

    # Group indices by true label.
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        label = label_encoder.encode(rec["cytokine"])
        groups[label].append(i)

    # Determine K and T from the first valid softmax_trajectory.
    # K is the number of model output classes (= softmax dimension),
    # which must equal label_encoder.n_classes() in practice.
    K = None
    T = None
    for rec in records:
        sm = rec.get("softmax_trajectory")
        if sm is not None:
            K, T = sm.shape[0], sm.shape[1]
            break
    if K is None or T is None:
        raise ValueError("No softmax_trajectory found in records.")

    confusion = np.zeros((K, K, T), dtype=np.float32)
    for a_idx in range(K):
        indices = groups.get(a_idx, [])
        if not indices:
            continue
        # Stack: (n_tubes, K, T)
        softmaxes = np.stack(
            [records[i]["softmax_trajectory"] for i in indices], axis=0
        )
        confusion[a_idx] = softmaxes.mean(axis=0)  # (K, T)

    return confusion, cytokine_names


def aggregate_confusion_to_donor_level(
    records: List[dict],
    label_encoder,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute donor-level confusion trajectories.

    For each (cytokine, donor) pair: median across pseudo-tubes → one (K, T) trajectory.
    Statistical comparisons should use this to respect effective N = n_donors.

    Args:
        records: per-tube dicts with 'softmax_trajectory', 'cytokine', 'donor'.
        label_encoder: encoder with .encode() and .cytokines.

    Returns:
        donor_confusion: np.ndarray of shape (n_donors, K, K, T).
                         donor_confusion[d, A, :, t] = median over donor-d A-tubes
                         of softmax[:, t].
        cytokine_names: list of length K.
        donor_names: sorted list of donor names (index d → donor).
    """
    cytokine_names = list(label_encoder.cytokines)
    K = len(cytokine_names)

    donor_names = sorted({rec["donor"] for rec in records})
    D = len(donor_names)
    donor_idx = {d: i for i, d in enumerate(donor_names)}

    T = None
    for rec in records:
        sm = rec.get("softmax_trajectory")
        if sm is not None:
            T = sm.shape[1]
            break
    if T is None:
        raise ValueError("No softmax_trajectory found in records.")

    # Group by (donor, cytokine).
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        d = donor_idx[rec["donor"]]
        a = label_encoder.encode(rec["cytokine"])
        groups[(d, a)].append(i)

    donor_confusion = np.zeros((D, K, K, T), dtype=np.float32)
    for (d, a), indices in groups.items():
        softmaxes = np.stack(
            [records[i]["softmax_trajectory"] for i in indices], axis=0
        )  # (n_tubes, K, T)
        donor_confusion[d, a] = np.median(softmaxes, axis=0)  # (K, T)

    return donor_confusion, cytokine_names, donor_names


def compute_asymmetry_score(
    confusion_trajectory: np.ndarray,
    late_epoch_fraction: float = 0.3,
) -> np.ndarray:
    """
    Compute the directed asymmetry score for each ordered pair (A, B).

    Asym[A, B] = mean_{t in late epochs}[ C[A,B,t] ] - mean_{t in late epochs}[ C[B,A,t] ]

    Positive Asym[A, B] → A confuses toward B more than B toward A → evidence for A→B cascade.

    The matrix is antisymmetric by construction: Asym[A,B] == -Asym[B,A].

    Args:
        confusion_trajectory: np.ndarray of shape (K, K, T).
        late_epoch_fraction: fraction of final epochs to use (default 0.3 = last 30%).

    Returns:
        asymmetry: np.ndarray of shape (K, K). Diagonal is zero.
    """
    T = confusion_trajectory.shape[2]
    n_late = max(1, int(T * late_epoch_fraction))
    late = confusion_trajectory[:, :, -n_late:]  # (K, K, n_late)
    late_mean = late.mean(axis=2)               # (K, K)
    asymmetry = late_mean - late_mean.T
    np.fill_diagonal(asymmetry, 0.0)
    return asymmetry


def compute_temporal_profile(
    confusion_trajectory: np.ndarray,
    a_idx: int,
    b_idx: int,
    early_threshold: float = 0.3,
    late_threshold: float = 0.7,
    onset_fraction: float = 0.05,
) -> dict:
    """
    Characterize the temporal profile of confusion from cytokine A toward cytokine B.

    Args:
        confusion_trajectory: np.ndarray of shape (K, K, T).
        a_idx: row index (true label = A).
        b_idx: col index (predicted class = B).
        early_threshold: peak before this fraction of T → 'early' (shared pathway).
        late_threshold: peak after this fraction of T → 'late' (cascade).
        onset_fraction: onset defined as first t where C > onset_fraction * max(C).

    Returns:
        dict with keys:
            'trajectory': np.ndarray of shape (T,)
            'peak_epoch': int, argmax of trajectory
            'onset_epoch': int or None
            'profile_type': str, one of 'early', 'mid', 'late'
            'peak_fraction': float, peak_epoch / T
            'max_value': float
    """
    traj = confusion_trajectory[a_idx, b_idx, :]  # (T,)
    T = len(traj)
    max_val = float(traj.max())
    peak_epoch = int(traj.argmax())
    peak_fraction = peak_epoch / max(T - 1, 1)

    if peak_fraction < early_threshold:
        profile_type = "early"
    elif peak_fraction > late_threshold:
        profile_type = "late"
    else:
        profile_type = "mid"

    onset_epoch = None
    if max_val > 0:
        threshold = onset_fraction * max_val
        above = np.where(traj >= threshold)[0]
        if len(above) > 0:
            onset_epoch = int(above[0])

    return {
        "trajectory": traj,
        "peak_epoch": peak_epoch,
        "onset_epoch": onset_epoch,
        "profile_type": profile_type,
        "peak_fraction": peak_fraction,
        "max_value": max_val,
    }


@torch.no_grad()
def extract_cell_type_attention_for_confusion(
    model,
    dataset,
    label_encoder,
    true_cyt: str,
    confused_cyt: str,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Identify which cell types carry the attention when the model confuses A with B.

    For each pseudo-tube with true label `true_cyt`:
      1. Run forward pass to get attention weights a_i and softmax output.
      2. Weight per-cell attention by softmax[confused_cyt_idx]: a_i * P(predict B | tube).
      3. Load cell-type labels from the tube .h5ad (tube.obs["cell_type"]).
      4. Group weighted attention by cell type, average within tube.
    Final output: mean across all A-tubes of per-cell-type weighted attention.

    High score for cell type X → X carries the B-signal in A-tubes → X is the relay.

    Args:
        model: CytokineABMIL (eval mode).
        dataset: PseudoTubeDataset with .get_entries() and __getitem__.
        label_encoder: encoder with .encode().
        true_cyt: name of cytokine A (true label).
        confused_cyt: name of cytokine B (confused label).
        device: torch device.

    Returns:
        dict mapping cell_type → float (mean attention weighted by P(predict B)).
        Also includes 'metric_description' key.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    true_idx = label_encoder.encode(true_cyt)
    confused_idx = label_encoder.encode(confused_cyt)
    entries = dataset.get_entries()

    cell_type_scores: Dict[str, List[float]] = defaultdict(list)

    for ds_idx, entry in enumerate(entries):
        if label_encoder.encode(entry["cytokine"]) != true_idx:
            continue

        X, label, _donor, _cyt_name = dataset[ds_idx]
        X = X.to(device)

        from cytokine_mil.models.cytokine_abmil_v2 import CytokineABMIL_V2
        if isinstance(model, CytokineABMIL_V2):
            y_hat, a_SA, _a_CA, _H = model(X)
            a = a_SA
        else:
            y_hat, a, _H = model(X)

        probs = F.softmax(y_hat, dim=0)
        p_confused = probs[confused_idx].item()

        # Weighted attention: a_i * P(predict B).
        weighted = (a * p_confused).cpu().numpy()  # (N,)

        # Load cell-type annotations from the h5ad.
        tube = anndata.read_h5ad(entry["path"])
        if "cell_type" not in tube.obs.columns:
            continue
        cell_types = tube.obs["cell_type"].values

        if len(cell_types) != len(weighted):
            continue

        # Average weighted attention per cell type in this tube.
        for ct in np.unique(cell_types):
            mask = cell_types == ct
            cell_type_scores[ct].append(float(weighted[mask].mean()))

    # Average across tubes.
    result = {
        ct: float(np.mean(vals))
        for ct, vals in cell_type_scores.items()
    }
    result["metric_description"] = (
        f"Mean attention weight per cell type, weighted by P(predict {confused_cyt}), "
        f"for pseudo-tubes with true label {true_cyt}. "
        f"High score → cell type is a relay in {true_cyt}→{confused_cyt} cascade."
    )
    return result


def build_cascade_graph(
    asymmetry_matrix: np.ndarray,
    label_encoder,
    fdr_alpha: float = 0.05,
    min_asymmetry: float = 0.0,
) -> "nx.DiGraph":
    """
    Build a directed cytokine cascade graph from the asymmetry score matrix.

    Uses Benjamini-Hochberg FDR correction across all K*(K-1) ordered pairs.
    An edge A→B is added when Asym[A,B] > min_asymmetry and is FDR-significant.

    The significance test uses a one-sample permutation approach:
    each asymmetry value is compared against the null distribution of
    |Asym[i,j]| values for i≠j (assuming the null is symmetric confusion).

    Args:
        asymmetry_matrix: np.ndarray of shape (K, K) from compute_asymmetry_score.
        label_encoder: encoder with .cytokines for index→name mapping.
        fdr_alpha: FDR threshold (default 0.05).
        min_asymmetry: minimum raw asymmetry to include (default 0.0).

    Returns:
        nx.DiGraph with nodes = cytokine names, edges A→B with attribute
        'asymmetry' = Asym[A,B].
    """
    import networkx as nx
    from scipy.stats import rankdata

    cytokine_names = list(label_encoder.cytokines)
    K = len(cytokine_names)

    # Collect all upper-triangle raw values for null distribution.
    upper_vals = []
    for a in range(K):
        for b in range(a + 1, K):
            upper_vals.append(abs(asymmetry_matrix[a, b]))
    null_dist = np.array(upper_vals)
    null_dist_sorted = np.sort(null_dist)[::-1]

    # Compute p-values for each ordered pair (A, B) where A≠B.
    pairs = []
    p_values = []
    asym_values = []
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            asym = asymmetry_matrix[a, b]
            if asym <= min_asymmetry:
                continue
            # p-value: fraction of null values >= asym.
            p = float(np.mean(null_dist >= asym))
            pairs.append((a, b))
            p_values.append(p)
            asym_values.append(asym)

    if not pairs:
        return nx.DiGraph()

    # Benjamini-Hochberg FDR correction.
    bh_threshold = _bh_threshold(p_values, fdr_alpha)

    G = nx.DiGraph()
    for name in cytokine_names:
        G.add_node(name)

    for (a, b), p, asym in zip(pairs, p_values, asym_values):
        if p <= bh_threshold:
            G.add_edge(
                cytokine_names[a],
                cytokine_names[b],
                asymmetry=float(asym),
                p_value=float(p),
            )

    return G


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bh_threshold(p_values: List[float], alpha: float) -> float:
    """
    Compute the Benjamini-Hochberg p-value threshold for FDR correction.

    Returns the largest p_i such that p_(i) <= (i/m) * alpha.
    Returns 0.0 if no hypotheses survive.
    """
    m = len(p_values)
    if m == 0:
        return 0.0
    sorted_p = np.sort(p_values)
    ranks = np.arange(1, m + 1)
    bh_vals = sorted_p * m / ranks
    # Find largest rank where p <= (rank/m)*alpha
    passing = np.where(sorted_p <= ranks / m * alpha)[0]
    if len(passing) == 0:
        return 0.0
    critical_rank = passing[-1]
    return float(sorted_p[critical_rank])
