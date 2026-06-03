"""
Signature-space coupling (CLAUDE.md §28 / "specific-dimensions" reframe).

MOTIVATION. Latent-geometry Path A (§20) measures cytokine coupling in the
ENCODER EMBEDDING with PBS-RC. PBS-RC removes the *resting* baseline but NOT the
*shared post-activation* program — the immune-response genes almost every
cytokine co-induces. So apparent "coupling" can be dominated by that shared
direction (cytokines look similar because they all activate), not by the
cytokine-SPECIFIC biology. On Oesinghaus, latent-geometry coupling ranks pairs
only weakly like signature-space coupling (Spearman rho ~ 0.29); on Sheu its
gate had no power at all (q=1 everywhere). This module measures coupling
DIRECTLY in the cytokine-specific dimensions — the discovered binary-IG
signatures S_X — bypassing the encoder embedding.

THE CROSS-ENGAGEMENT MATRIX. For cytokines with discovered signatures S_X:

    M[a, b] = s(a, S_b) - s(PBS, S_b)      (a's cells engaging b's signature,
                                            PBS-normalised; median over cell types)

where s(x, S) = mean expression over the genes of S. This is exactly the
`sA_PB_norm` quantity of `pathway_audit.directional_asymmetry_test`, generalised
to every ordered pair. Two readouts fall out of one matrix:

    coupling(a, b)  = M[a, b] + M[b, a]    SYMMETRIC  -> do a and b mutually
                                           engage each other's SPECIFIC programs?
    cross_asym(a,b) = M[a, b] - M[b, a]    ANTISYMMETRIC -> direction (M7/§26);
                                           + => a upstream (a_to_b), a<b canonical.

The "strong enough signal" gate (the user's step) is a GENE-SET NULL: is the
symmetric coupling larger than the engagement of random gene sets of the same
size (drawn disjoint from any observed S_X)? Pairs that clear it are coupled in
SPECIFIC dimensions, not via generic activation. Direction (cross_asym) is then
read only on coupled pairs.

Allowed imports: numpy only (analysis layer; no scipy/matplotlib/pandas here).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-cell-type cross-engagement
# ---------------------------------------------------------------------------

def engagement_per_celltype(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    sig_idx_dict: Dict[str, np.ndarray],
    pbs_label: str = "PBS",
    min_cells: int = 10,
) -> Tuple[List[str], List[str], np.ndarray]:
    """PBS-normalised cross-engagement E[t, i, j] = s(cyt_i, S_{cyt_j}) at cell type t.

    Only cytokines that (a) have a signature in sig_idx_dict AND (b) appear in
    cells_by_pair are kept. A (t, i) entry is finite only if cyt_i AND PBS both
    have >= min_cells at cell type t; otherwise NaN.

    Returns:
        cytokines:   sorted list (row/col order of E).
        cell_types:  list (axis-0 order of E).
        E:           (T, n, n) float64; E[t, i, j] = mean_{i-cells at t}(S_j genes)
                     - mean_{PBS at t}(S_j genes). NaN where unavailable.
    """
    cytokines = sorted(c for c in sig_idx_dict
                       if any(k[0] == c for k in cells_by_pair))
    n = len(cytokines)
    cyt_to_col = {c: j for j, c in enumerate(cytokines)}
    sig_arrs = {c: np.asarray(sig_idx_dict[c], dtype=np.int64) for c in cytokines}

    cell_types = sorted({ct for (_, ct) in cells_by_pair.keys()})
    E = np.full((len(cell_types), n, n), np.nan, dtype=np.float64)

    for t, T in enumerate(cell_types):
        pbs_key = (pbs_label, T)
        if pbs_key not in cells_by_pair:
            continue
        cP = cells_by_pair[pbs_key]
        if len(cP) < min_cells:
            continue
        # PBS mean over each signature's genes (column-indexed by j)
        pbs_score = np.array(
            [float(cP[:, sig_arrs[cytokines[j]]].mean()) for j in range(n)],
            dtype=np.float64,
        )
        for i, a in enumerate(cytokines):
            cA = cells_by_pair.get((a, T))
            if cA is None or len(cA) < min_cells:
                continue
            for j in range(n):
                a_score = float(cA[:, sig_arrs[cytokines[j]]].mean())
                E[t, i, j] = a_score - pbs_score[j]
    return cytokines, cell_types, E


def cross_engagement_matrix(E: np.ndarray) -> np.ndarray:
    """Aggregate E[t,i,j] across cell types -> M[i,j] (nanmedian over t)."""
    with np.errstate(all="ignore"):
        return np.nanmedian(E, axis=0)


# ---------------------------------------------------------------------------
# Coupling + direction, with a gene-set null gate
# ---------------------------------------------------------------------------

def _null_engagement_tensor(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    cytokines: List[str],
    cell_types: List[str],
    set_size: int,
    excluded_indices: Sequence[int],
    n_genes: int,
    pbs_label: str,
    min_cells: int,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null engagement N[k, t, i] = s(cyt_i, R_k) - s(PBS, R_k) at cell type t.

    R_k are n_perm random gene sets of `set_size`, drawn disjoint from any
    observed S_X (excluded_indices). One set per permutation is shared across
    cytokines (the symmetric coupling null asks: do a and b both engage RANDOM
    genes as much as they engage each other's SPECIFIC signatures?).
    """
    pool = np.array([i for i in range(n_genes) if i not in set(excluded_indices)],
                    dtype=np.int64)
    if len(pool) < set_size:
        raise ValueError(
            f"Null pool too small: {len(pool)} non-S_X genes, need {set_size}.")
    n = len(cytokines)
    N = np.full((n_perm, len(cell_types), n), np.nan, dtype=np.float64)
    rand_sets = [rng.choice(pool, size=set_size, replace=False) for _ in range(n_perm)]
    for t, T in enumerate(cell_types):
        cP = cells_by_pair.get((pbs_label, T))
        if cP is None or len(cP) < min_cells:
            continue
        for k, R in enumerate(rand_sets):
            pbs_r = float(cP[:, R].mean())
            for i, a in enumerate(cytokines):
                cA = cells_by_pair.get((a, T))
                if cA is None or len(cA) < min_cells:
                    continue
                N[k, t, i] = float(cA[:, R].mean()) - pbs_r
    return N


def coupling_direction(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    sig_idx_dict: Dict[str, np.ndarray],
    excluded_indices: Optional[Sequence[int]] = None,
    n_genes: Optional[int] = None,
    pbs_label: str = "PBS",
    min_cells: int = 10,
    n_perm: int = 100,
    set_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, object]]:
    """Symmetric coupling + antisymmetric direction per unordered pair, with a
    gene-set-null p-value on the coupling.

    Returns a list of dict rows (one per unordered pair a<b):
        axis_a, axis_b, m_ab, m_ba, coupling, cross_asym,
        coupling_null_p, n_celltypes
    coupling_null_p is NaN if n_perm == 0 or the null could not be built.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    cytokines, cell_types, E = engagement_per_celltype(
        cells_by_pair, sig_idx_dict, pbs_label, min_cells)
    M = cross_engagement_matrix(E)
    n = len(cytokines)

    # per-pair count of cell types where BOTH a and b engagements are finite
    finite_ab = np.isfinite(E)  # (T, n, n)

    null_coupling = None
    if n_perm and n_perm > 0:
        if excluded_indices is None:
            excluded = set()
            for v in sig_idx_dict.values():
                excluded.update(np.asarray(v).tolist())
            excluded_indices = sorted(excluded)
        if n_genes is None:
            # total gene count = width of any cell matrix (NOT max signature index)
            n_genes = 0
            for v in cells_by_pair.values():
                n_genes = int(v.shape[1])
                break
        if set_size is None:
            sizes = [len(np.asarray(sig_idx_dict[c])) for c in cytokines]
            set_size = int(np.median(sizes)) if sizes else 0
        try:
            N = _null_engagement_tensor(
                cells_by_pair, cytokines, cell_types, set_size,
                excluded_indices, n_genes, pbs_label, min_cells, n_perm, rng)
            # null coupling per pair, per perm: nanmedian_t[N[k,t,i] + N[k,t,j]]
            with np.errstate(all="ignore"):
                # shape (n_perm, T, n, n): N[k,t,i] + N[k,t,j]
                pair_sum = N[:, :, :, None] + N[:, :, None, :]
                null_coupling = np.nanmedian(pair_sum, axis=1)  # (n_perm, n, n)
        except Exception:
            null_coupling = None

    rows: List[Dict[str, object]] = []
    for i in range(n):
        for j in range(i + 1, n):
            m_ab = M[i, j]
            m_ba = M[j, i]
            coupling = m_ab + m_ba
            cross = m_ab - m_ba
            n_ct = int(np.sum(finite_ab[:, i, j] & finite_ab[:, j, i]))
            p = float("nan")
            if null_coupling is not None and np.isfinite(coupling):
                nc = null_coupling[:, i, j]
                nc = nc[np.isfinite(nc)]
                if nc.size:
                    p = float(np.mean(nc >= coupling))
            rows.append({
                "axis_a": cytokines[i], "axis_b": cytokines[j],
                "m_ab": float(m_ab), "m_ba": float(m_ba),
                "coupling": float(coupling), "cross_asym": float(cross),
                "coupling_null_p": p, "n_celltypes": n_ct,
            })
    return rows
