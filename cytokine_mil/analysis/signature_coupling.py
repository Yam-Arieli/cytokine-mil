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


# ---------------------------------------------------------------------------
# Donor-level coupling gate (effective N = #donors, not #cells; CLAUDE.md §16)
# ---------------------------------------------------------------------------
# The cell-level gate (`coupling_direction`) is over-powered: pooling thousands
# of cells makes the random-gene null trivially beatable (~77% of pairs "coupled"
# on Oesinghaus). The correct unit of independence is the donor. We compute, PER
# DONOR, the excess of real specific-signature coupling over the donor's own
# random-gene-set baseline, then test whether that excess is consistently > 0
# across donors with a one-sided sign-flip permutation test. This caps power at
# #donors (~10), so the gate can actually discriminate.

def donor_excess_matrix(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    sig_idx_dict: Dict[str, np.ndarray],
    global_cyts: List[str],
    pbs_label: str = "PBS",
    min_cells: int = 10,
    n_perm: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """One donor's (coupling - mean random-gene-set coupling) per pair.

    Returns a (G, G) symmetric matrix aligned to ``global_cyts`` (= sorted
    sig_idx keys); NaN where a cytokine or PBS is absent for this donor. Call
    once per donor and stack the results; cells for the donor can be freed
    afterwards (keeps memory ~1 donor at a time).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    G = len(global_cyts)
    gidx = {c: k for k, c in enumerate(global_cyts)}
    out = np.full((G, G), np.nan, dtype=np.float64)

    cyts, cell_types, E = engagement_per_celltype(
        cells_by_pair, sig_idx_dict, pbs_label, min_cells)
    if not cyts:
        return out
    M = cross_engagement_matrix(E)
    n = len(cyts)

    excluded = sorted({i for v in sig_idx_dict.values()
                       for i in np.asarray(v).tolist()})
    n_genes = 0
    for v in cells_by_pair.values():
        n_genes = int(v.shape[1]); break
    sizes = [len(np.asarray(sig_idx_dict[c])) for c in cyts]
    set_size = int(np.median(sizes)) if sizes else 0

    nullbar = np.zeros((n, n), dtype=np.float64)
    if n_perm and set_size > 0:
        try:
            N = _null_engagement_tensor(
                cells_by_pair, cyts, cell_types, set_size, excluded, n_genes,
                pbs_label, min_cells, n_perm, rng)
            with np.errstate(all="ignore"):
                pair_sum = N[:, :, :, None] + N[:, :, None, :]   # (k,T,n,n)
                null_coupling = np.nanmedian(pair_sum, axis=1)    # (k,n,n)
                nullbar = np.nanmean(null_coupling, axis=0)       # (n,n)
        except Exception:
            nullbar = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            gi, gj = gidx[cyts[i]], gidx[cyts[j]]
            out[gi, gj] = (M[i, j] + M[j, i]) - float(nullbar[i, j])
    return out


def _degree_center(C: np.ndarray) -> np.ndarray:
    """Additive double-centering of a SYMMETRIC coupling matrix (diagonal = NaN).

    R[i,j] = C[i,j] - d_i - d_j + g, where d_i = mean off-diagonal coupling of
    cytokine i (its 'strength'/degree) and g = grand off-diagonal mean. Removes
    each cytokine's overall engagement level (the hub/degree artifact: e.g. IL-15
    couples to everything), leaving pair-SPECIFIC residual coupling. NaNs (absent
    pairs / the diagonal) are excluded from the means.
    """
    with np.errstate(all="ignore"):
        d = np.nanmean(C, axis=1)                       # node strength (diag NaN -> excluded)
        g = float(np.nanmean(C))
    return C - d[:, None] - d[None, :] + g


def donor_residual_coupling_matrix(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    sig_idx_dict: Dict[str, np.ndarray],
    global_cyts: List[str],
    pbs_label: str = "PBS",
    min_cells: int = 10,
) -> np.ndarray:
    """One donor's HUB-CORRECTED (degree-centered) coupling per pair.

    coupling_d[a,b] = M_d[a,b] + M_d[b,a]; then :func:`_degree_center` removes each
    cytokine's overall strength. No random-gene null is needed — the centering
    removes the shared-activation/degree baseline, so a pair's positive residual
    means SPECIFIC engagement beyond what the two cytokines' general levels
    predict. Returns (G, G) aligned to ``global_cyts`` (NaN where absent).
    """
    G = len(global_cyts)
    gidx = {c: k for k, c in enumerate(global_cyts)}
    out = np.full((G, G), np.nan, dtype=np.float64)
    cyts, cell_types, E = engagement_per_celltype(
        cells_by_pair, sig_idx_dict, pbs_label, min_cells)
    if not cyts:
        return out
    M = cross_engagement_matrix(E)
    n = len(cyts)
    C = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = M[i, j] + M[j, i]
    R = _degree_center(C)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(R[i, j]):
                out[gidx[cyts[i]], gidx[cyts[j]]] = R[i, j]
    return out


def cell_coupling_degree(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    sig_idx_dict: Dict[str, np.ndarray],
    pbs_label: str = "PBS",
    min_cells: int = 10,
    n_perm: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, object]]:
    """Cell-level (pooled) coupling, RAW and DEGREE-corrected, each gene-set-null gated.

    For few-donor datasets where the donor-level gate is inapplicable, this tests
    whether the degree correction (hub removal) preserves real coupling while cutting
    the over-call -- at the cell level, where all pairs are testable. The null
    matrices are degree-centered the SAME way as the observed (apples-to-apples), so
    ``null_p_hub`` asks whether the pair-specific residual exceeds random genes.

    Returns rows: axis_a, axis_b, coupling_raw, coupling_hub, cross_asym,
    null_p_raw, null_p_hub.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    cyts, cell_types, E = engagement_per_celltype(
        cells_by_pair, sig_idx_dict, pbs_label, min_cells)
    M = cross_engagement_matrix(E)
    n = len(cyts)
    C = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = M[i, j] + M[j, i]
    C_hub = _degree_center(C)

    null_raw = null_hub = None
    if n_perm and n > 0:
        excluded = sorted({i for v in sig_idx_dict.values()
                           for i in np.asarray(v).tolist()})
        n_genes = next(int(v.shape[1]) for v in cells_by_pair.values())
        sizes = [len(np.asarray(sig_idx_dict[c])) for c in cyts]
        set_size = int(np.median(sizes)) if sizes else 0
        if set_size > 0:
            try:
                N = _null_engagement_tensor(
                    cells_by_pair, cyts, cell_types, set_size, excluded, n_genes,
                    pbs_label, min_cells, n_perm, rng)
                with np.errstate(all="ignore"):
                    pair_sum = N[:, :, :, None] + N[:, :, None, :]
                    null_raw = np.nanmedian(pair_sum, axis=1)        # (k,n,n)
                null_hub = np.full_like(null_raw, np.nan)
                for k in range(null_raw.shape[0]):
                    Ck = null_raw[k].copy()
                    np.fill_diagonal(Ck, np.nan)
                    null_hub[k] = _degree_center(Ck)
            except Exception:
                null_raw = null_hub = None

    rows: List[Dict[str, object]] = []
    for i in range(n):
        for j in range(i + 1, n):
            row = {"axis_a": cyts[i], "axis_b": cyts[j],
                   "coupling_raw": float(C[i, j]), "coupling_hub": float(C_hub[i, j]),
                   "cross_asym": float(M[i, j] - M[j, i]),
                   "null_p_raw": float("nan"), "null_p_hub": float("nan")}
            if null_raw is not None:
                nr = null_raw[:, i, j]; nr = nr[np.isfinite(nr)]
                nh = null_hub[:, i, j]; nh = nh[np.isfinite(nh)]
                if nr.size and np.isfinite(C[i, j]):
                    row["null_p_raw"] = float(np.mean(nr >= C[i, j]))
                if nh.size and np.isfinite(C_hub[i, j]):
                    row["null_p_hub"] = float(np.mean(nh >= C_hub[i, j]))
            rows.append(row)
    return rows


def _signflip_p(vals: np.ndarray, n_signflip: int,
                rng: np.random.Generator) -> Tuple[float, float]:
    """One-sided sign-flip permutation p for H1: mean(vals) > 0.

    Exact enumeration of all 2^n sign vectors when n <= 18 (the observed/identity
    flip is included, so p = mean(t_null >= t_obs)); sampled otherwise.
    """
    vals = np.asarray(vals, dtype=np.float64)
    t_obs = float(vals.mean())
    n = len(vals)
    if n <= 18:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.float64) * 2 - 1
        t_null = (signs * vals).mean(axis=1)
        p = float(np.mean(t_null >= t_obs))
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_signflip, n))
        t_null = (signs * vals).mean(axis=1)
        p = float((np.sum(t_null >= t_obs) + 1) / (n_signflip + 1))
    return p, t_obs


def _bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values (numpy only)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    q = np.empty(m, dtype=np.float64)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * m / (rank + 1))
        q[idx] = prev
    return q


def donor_coupling_test(
    excess_stack: np.ndarray,
    global_cyts: List[str],
    min_donors: int = 5,
    n_signflip: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, object]]:
    """Across-donor sign-flip test on the per-donor excess matrices.

    Args:
        excess_stack: (D, G, G) from stacking :func:`donor_excess_matrix`.
        global_cyts:  cytokine order (axis 1/2 of the stack).
        min_donors:   minimum donors with a finite excess for a pair to be tested
                      (5 lets a unanimous pair reach p<0.05; 4 floors at ~0.06).
    Returns:
        list of rows: axis_a, axis_b, excess_mean, n_donors, p_donor, q_donor
        (BH-FDR over all tested pairs). Sorted by excess_mean descending.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    G = len(global_cyts)
    rows: List[Dict[str, object]] = []
    for i in range(G):
        for j in range(i + 1, G):
            vals = excess_stack[:, i, j]
            vals = vals[np.isfinite(vals)]
            if len(vals) < min_donors:
                continue
            p, t = _signflip_p(vals, n_signflip, rng)
            rows.append({
                "axis_a": global_cyts[i], "axis_b": global_cyts[j],
                "excess_mean": float(t), "n_donors": int(len(vals)),
                "p_donor": float(p),
            })
    if rows:
        q = _bh_fdr([r["p_donor"] for r in rows])
        for r, qq in zip(rows, q):
            r["q_donor"] = float(qq)
    rows.sort(key=lambda r: r["excess_mean"], reverse=True)
    return rows
