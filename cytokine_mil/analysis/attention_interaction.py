"""
Self-attention cell x cell interaction readouts (CLAUDE.md §34).

The §34 model inserts a self-attention block over cells, so for one cytokine's
tubes we observe a cell-type x cell-type interaction matrix

    M[τ, σ] = mean over τ-cells of the total attention they place on σ-cells
              (row-normalised: Σ_σ M[τ, σ] = 1),

produced by scripts/extract_selfattn_trajectory.py -> interaction_trajectory.pkl.
This module turns that matrix (and its training trajectory) into readouts that the
AB-MIL pooling attention (§33) structurally cannot produce — a directed
"who-influences-whom" cell-type graph, its go/no-go sanity, and a per-cytokine-pair
relay direction statistic.

Pure numpy — consumes the dict structures from interaction_trajectory.pkl. No
torch, no file IO, no model. Every ranking/summary returns a ``metric_description``.

interaction_trajectory.pkl structure:
    interaction:           {cytokine -> {"τ||σ" -> np.array(n_epochs)}}   (donor-mean)
    interaction_per_donor: {cytokine -> {"τ||σ" -> {donor -> np.array}}}
    offdiag:               {cytokine -> np.array(n_epochs)}               (donor-mean cross-type frac)
    epochs, cell_types, cytokines, pair_sep
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PAIR_SEP = "||"


def parse_pair(key: str, sep: str = PAIR_SEP) -> Tuple[str, str]:
    """Split a "τ||σ" interaction key into (source_type, target_type)."""
    tau, sigma = key.split(sep, 1)
    return tau, sigma


# ---------------------------------------------------------------------------
# G0 go/no-go: does self-attention actually use cross-cell-type information?
# ---------------------------------------------------------------------------

def offdiagonal_summary(
    offdiag: Dict[str, np.ndarray],
    epochs: Sequence[int],
    min_frac: float = 0.2,
) -> Dict:
    """
    Summarise cross-cell-type ("off-diagonal") attention mass over training.

    ``offdiag[cyt][t]`` = mean over cells of the fraction of a cell's attention
    placed on OTHER cell types at epoch t (1 = fully cross-type; ~0 = each cell
    attends only within its own type => the SAB collapsed to self/diagonal and
    cells do NOT interact => the §34 premise fails).

    Args:
        offdiag: {cytokine -> np.array(n_epochs)} donor-mean cross-type fraction.
        epochs: checkpoint epochs (ascending).
        min_frac: go/no-go threshold on the final-epoch mean across cytokines.
    Returns:
        dict with 'per_cytokine' {cyt -> {'final','mean','slope'}},
        'mean_final' (across cytokines), 'gate_pass' (mean_final >= min_frac),
        'min_frac', 'metric_description'.
    """
    e = np.asarray(epochs, dtype=np.float64)
    per_cyt: Dict[str, Dict] = {}
    finals: List[float] = []
    for cyt, traj in offdiag.items():
        arr = np.asarray(traj, dtype=np.float64)
        if arr.size == 0:
            per_cyt[cyt] = {"final": 0.0, "mean": 0.0, "slope": 0.0}
            continue
        slope = (float(np.polyfit(e, arr, 1)[0])
                 if arr.size >= 2 and e.size == arr.size and e.std() > 0 else 0.0)
        per_cyt[cyt] = {"final": float(arr[-1]), "mean": float(arr.mean()), "slope": slope}
        finals.append(float(arr[-1]))
    mean_final = float(np.mean(finals)) if finals else 0.0
    return {
        "per_cytokine": per_cyt,
        "mean_final": mean_final,
        "gate_pass": bool(mean_final >= min_frac),
        "min_frac": min_frac,
        "metric_description": (
            "off-diagonal (cross-cell-type) attention fraction = mean over cells of "
            "the attention a cell places on other cell types; go/no-go = final-epoch "
            f"mean across cytokines >= {min_frac} (else cells don't interact)"
        ),
    }


# ---------------------------------------------------------------------------
# Interaction matrix & directed asymmetry (per cytokine)
# ---------------------------------------------------------------------------

def interaction_matrix(
    interaction_cyt: Dict[str, np.ndarray],
    epoch_idx: int = -1,
    cell_types: Optional[Sequence[str]] = None,
    sep: str = PAIR_SEP,
) -> Tuple[np.ndarray, List[str]]:
    """
    Assemble the cell-type x cell-type interaction matrix M for one cytokine at
    one epoch. ``M[i, j]`` = attention type i places on type j (row-normalised).

    Args:
        interaction_cyt: {"τ||σ" -> np.array(n_epochs)} for one cytokine.
        epoch_idx: epoch index into each trajectory (default -1 = final).
        cell_types: fixed ordering; default = sorted set observed in the keys.
        sep: pair separator.
    Returns:
        (M, cell_types) where M is (n_ct, n_ct); missing entries are 0.
    """
    if cell_types is None:
        cts = set()
        for key in interaction_cyt:
            a, b = parse_pair(key, sep)
            cts.add(a); cts.add(b)
        cell_types = sorted(cts)
    idx = {ct: i for i, ct in enumerate(cell_types)}
    M = np.zeros((len(cell_types), len(cell_types)), dtype=np.float64)
    for key, traj in interaction_cyt.items():
        a, b = parse_pair(key, sep)
        if a in idx and b in idx:
            arr = np.asarray(traj, dtype=np.float64)
            if arr.size:
                M[idx[a], idx[b]] = float(arr[epoch_idx])
    return M, list(cell_types)


def asymmetry_matrix(M: np.ndarray) -> np.ndarray:
    """Directed asymmetry Asym[i,j] = M[i,j] - M[j,i] (antisymmetric)."""
    return M - M.T


def directed_pairs(
    interaction_cyt: Dict[str, np.ndarray],
    epoch_idx: int = -1,
    cell_types: Optional[Sequence[str]] = None,
    top_k: int = 10,
    sep: str = PAIR_SEP,
) -> Dict:
    """
    Rank cell-type pairs by directed interaction asymmetry for one cytokine.

    For each unordered pair {i,j}, Asym = M[i,j] - M[j,i]; a positive value means
    type i attends to type j MORE than j attends to i (i "listens to" j -> j is a
    candidate upstream source for i). Returns the strongest |Asym| pairs.

    Returns:
        dict with 'pairs': list of (source_i, target_j, asym) sorted by |asym|
        desc (asym>0 => i attends to j more), and 'metric_description'.
    """
    M, cts = interaction_matrix(interaction_cyt, epoch_idx, cell_types, sep)
    Asym = asymmetry_matrix(M)
    pairs = []
    for i in range(len(cts)):
        for j in range(i + 1, len(cts)):
            pairs.append((cts[i], cts[j], float(Asym[i, j])))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return {
        "pairs": pairs[:top_k],
        "metric_description": (
            "cell-type pairs ranked by |M[i,j] - M[j,i]| (directed self-attention "
            "asymmetry); asym>0 => type i attends to type j more than the reverse"
        ),
    }


# ---------------------------------------------------------------------------
# Per-cytokine-pair relay direction (cross-cytokine, for the 88% benchmark)
# ---------------------------------------------------------------------------

def relay_interaction_direction(
    interaction: Dict[str, Dict[str, np.ndarray]],
    A: str,
    B: str,
    T_A: str,
    T_B: str,
    epoch_idx: int = -1,
    sep: str = PAIR_SEP,
) -> Dict:
    """
    Signed self-attention direction statistic for a cytokine pair (A, B).

    Relay logic (cascade A -> B): in A's tubes the downstream responder cell type
    T_B should "listen to" the upstream responder T_A (it picks up A's signal),
    while in B's own tubes T_A need not listen to T_B. So compare the reciprocal
    cross-type attention across the two cytokine contexts:

        D(A,B) = M^A[T_B, T_A] - M^B[T_A, T_B]

    D > 0 => A upstream (A -> B); D < 0 => B upstream. T_A / T_B are the pooling
    attention-primary cell types of A / B (passed in — data-driven, no prior).

    Args:
        interaction: {cytokine -> {"τ||σ" -> np.array}} donor-mean.
        A, B: cytokine names. T_A, T_B: their primary responder cell types.
        epoch_idx: epoch index (default final).
    Returns:
        dict with 'A','B','T_A','T_B','D','m_A_BtoA','m_B_AtoB','call'
        ('a_to_b'|'b_to_a'|'none'), 'metric_description'.
        D is NaN (call 'none') if either matrix entry is unavailable.
    """
    md = (
        "relay interaction direction D(A,B) = M^A[T_B,T_A] - M^B[T_A,T_B] "
        "(does B's responder attend to A's responder in A's tubes more than the "
        "reverse in B's tubes); D>0 => A->B. T_A/T_B = pooling attention-primary."
    )

    def _entry(cyt: str, src: str, dst: str) -> Optional[float]:
        d = interaction.get(cyt, {})
        arr = d.get(f"{src}{sep}{dst}")
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float64)
        return float(arr[epoch_idx]) if arr.size else None

    m_A = _entry(A, T_B, T_A)   # in A's tubes: T_B attends to T_A
    m_B = _entry(B, T_A, T_B)   # in B's tubes: T_A attends to T_B
    if m_A is None or m_B is None:
        return {"A": A, "B": B, "T_A": T_A, "T_B": T_B, "D": float("nan"),
                "m_A_BtoA": m_A, "m_B_AtoB": m_B, "call": "none",
                "metric_description": md}
    D = m_A - m_B
    call = "a_to_b" if D > 0 else ("b_to_a" if D < 0 else "none")
    return {"A": A, "B": B, "T_A": T_A, "T_B": T_B, "D": float(D),
            "m_A_BtoA": m_A, "m_B_AtoB": m_B, "call": call,
            "metric_description": md}
