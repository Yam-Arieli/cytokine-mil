"""
Direction-permutation null for cross_asym (CLAUDE.md §27.2).

WHY a new null. The §26 random-gene-set null (`run_pipeline_a_bridge_b.py
:_evaluate_null_for_axis`) tests whether the discovered S_X genes are
*cytokine-specific* (vs random activation-responsive HVGs). It does NOT test
whether the *direction* — the a-vs-b cross-engagement asymmetry — is real:
overlap pairs pass it because shared-program genes are themselves
cytokine-specific. The correct null for DIRECTION holds S_a, S_b FIXED and
breaks only the a-vs-b cell-condition label.

cross_asym per cell type T (matches
`pathway_audit.directional_asymmetry_test`'s `sA_PB_norm - sB_PA_norm`):

    cross_asym(T) = (mean_{a-cells} score_on_Sb - pbs_on_Sb)
                  - (mean_{b-cells} score_on_Sa - pbs_on_Sa)

where score_on_Sx(cell) = mean expression over the Sx gene indices, and the
pbs_on_Sx terms are computed from PBS cells (held FIXED, never permuted).
The aggregate observed statistic is the median of cross_asym(T) across cell
types (matching the pipeline's aggregator).

NULL. Within each cell type, pool the a-cells and b-cells and randomly
re-assign which are "pseudo-a" (size n_a) and which are "pseudo-b" (size n_b);
recompute cross_asym(T); aggregate median across cell types per permutation.

RECENTRING. The null has a NONZERO baseline centre by construction: the S_a vs
S_b magnitude offset (pooled mean of score_on_Sb above PBS minus pooled mean of
score_on_Sa above PBS) survives label permutation. That offset is a NUISANCE
(signature magnitude), not direction; the directional effect is
`observed - null_centre`. So the empirical two-sided p recentres by the null
mean:

    p_emp = mean_k( |null_k - null_centre| >= |observed - null_centre| )

The direction CALL (sign) is still the observed cross_median sign; the null
only asks whether the a-vs-b asymmetry is beyond label noise. This tests
"is the directional asymmetry statistically reliable", NOT "is this a cascade"
(existence = Path A) and NOT "is it causal" (wet-lab). Overlap pairs with a
real magnitude asymmetry MAY pass — that is expected and correct.

Allowed imports: numpy only (the pipeline driver's dependency audit forbids
scipy/matplotlib in this layer; BH-FDR and Storey pi0 are implemented by hand).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-cell-type score precomputation
# ---------------------------------------------------------------------------

def _cell_scores(
    cells: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell mean expression over the S_a and S_b gene-index sets.

    Returns (score_on_Sa, score_on_Sb), each shape (n_cells,).
    """
    return (cells[:, idx_a].mean(axis=1).astype(np.float64),
            cells[:, idx_b].mean(axis=1).astype(np.float64))


def _celltype_terms(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    A: str,
    B: str,
    pbs_label: str,
    min_cells: int,
) -> List[Dict[str, object]]:
    """Precompute per-cell scores + PBS baselines for every qualifying cell type.

    A cell type qualifies iff it has >= min_cells in each of A, B, and PBS
    (mirrors `directional_asymmetry_test`'s filter exactly).
    """
    cell_types = sorted({ct for (_, ct) in cells_by_pair.keys()})
    out: List[Dict[str, object]] = []
    for T in cell_types:
        needed = [(A, T), (B, T), (pbs_label, T)]
        if any(k not in cells_by_pair for k in needed):
            continue
        cA = cells_by_pair[(A, T)]
        cB = cells_by_pair[(B, T)]
        cP = cells_by_pair[(pbs_label, T)]
        if len(cA) < min_cells or len(cB) < min_cells or len(cP) < min_cells:
            continue
        a_sa, a_sb = _cell_scores(cA, idx_a, idx_b)
        b_sa, b_sb = _cell_scores(cB, idx_a, idx_b)
        out.append({
            "T": T,
            "a_sa": a_sa, "a_sb": a_sb,
            "b_sa": b_sa, "b_sb": b_sb,
            "pbs_sa": float(cP[:, idx_a].mean()),
            "pbs_sb": float(cP[:, idx_b].mean()),
            "n_a": int(len(cA)), "n_b": int(len(cB)),
        })
    return out


def _observed_cross_per_celltype(terms: List[Dict[str, object]]) -> np.ndarray:
    """cross_asym(T) = (mean_a score_Sb - pbs_Sb) - (mean_b score_Sa - pbs_Sa)."""
    vals = np.empty(len(terms), dtype=np.float64)
    for j, t in enumerate(terms):
        sA_in_PB = float(t["a_sb"].mean())   # a-cells engaging S_b
        sB_in_PA = float(t["b_sa"].mean())   # b-cells engaging S_a
        vals[j] = (sA_in_PB - t["pbs_sb"]) - (sB_in_PA - t["pbs_sa"])
    return vals


# ---------------------------------------------------------------------------
# Public: direction-permutation test for one axis
# ---------------------------------------------------------------------------

def _nan_result() -> Dict[str, float]:
    return {
        "dir_observed_cross_median": float("nan"),
        "dir_null_center": float("nan"),
        "dir_null_q025": float("nan"),
        "dir_null_q975": float("nan"),
        "dir_p_emp": float("nan"),
        "dir_n_perms": 0,
        "dir_n_celltypes": 0,
    }


def direction_permutation_test(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    A: str,
    B: str,
    pbs_label: str = "PBS",
    min_cells: int = 10,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Recentred two-sided permutation p-value for the cross_asym direction.

    Args:
        cells_by_pair: {(cytokine, cell_type) -> (N, G) float array}.
        idx_a: gene indices of S_a (= S_{axis_a}; the P_A column in §24).
        idx_b: gene indices of S_b (= S_{axis_b}; the P_B column in §24).
        A, B:  axis_a, axis_b (order matters; cross_asym is antisymmetric).
        n_perm: permutations (label shuffles). 0 returns observed only.
        rng:   np.random.Generator (default_rng(0) if None).

    Returns a dict with keys:
        dir_observed_cross_median, dir_null_center, dir_null_q025,
        dir_null_q975, dir_p_emp, dir_n_perms, dir_n_celltypes.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    idx_a = np.asarray(idx_a, dtype=np.int64)
    idx_b = np.asarray(idx_b, dtype=np.int64)
    if idx_a.size == 0 or idx_b.size == 0:
        return _nan_result()

    terms = _celltype_terms(cells_by_pair, idx_a, idx_b, A, B, pbs_label, min_cells)
    if not terms:
        return _nan_result()

    obs = _observed_cross_per_celltype(terms)
    observed_median = float(np.median(obs))

    res = _nan_result()
    res["dir_observed_cross_median"] = observed_median
    res["dir_n_celltypes"] = int(len(terms))
    if n_perm <= 0:
        return res

    # Pre-pool per cell type so each permutation is a cheap index+mean.
    pooled: List[Tuple[np.ndarray, np.ndarray, int, int, float, float]] = []
    for t in terms:
        pooled_sa = np.concatenate([t["a_sa"], t["b_sa"]])  # score on S_a, all cells
        pooled_sb = np.concatenate([t["a_sb"], t["b_sb"]])  # score on S_b, all cells
        pooled.append(
            (pooled_sa, pooled_sb, int(t["n_a"]), int(t["n_b"]),
             float(t["pbs_sa"]), float(t["pbs_sb"]))
        )

    null_medians = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        vals = np.empty(len(pooled), dtype=np.float64)
        for j, (psa, psb, n_a, n_b, pbs_sa, pbs_sb) in enumerate(pooled):
            perm = rng.permutation(n_a + n_b)
            pseudo_a = perm[:n_a]
            pseudo_b = perm[n_a:]
            sA_in_PB = float(psb[pseudo_a].mean())
            sB_in_PA = float(psa[pseudo_b].mean())
            vals[j] = (sA_in_PB - pbs_sb) - (sB_in_PA - pbs_sa)
        null_medians[k] = float(np.median(vals))

    null_center = float(np.mean(null_medians))
    dev_obs = observed_median - null_center
    dev_null = null_medians - null_center
    res["dir_null_center"] = null_center
    res["dir_null_q025"] = float(np.quantile(null_medians, 0.025))
    res["dir_null_q975"] = float(np.quantile(null_medians, 0.975))
    res["dir_p_emp"] = float(np.mean(np.abs(dev_null) >= abs(dev_obs)))
    res["dir_n_perms"] = int(n_perm)
    return res


# ---------------------------------------------------------------------------
# Multiple-testing helpers (numpy-only; no scipy in this layer)
# ---------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values. NaN inputs stay NaN and are excluded from m."""
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    mask = ~np.isnan(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    qr = ranked * m / np.arange(1, m + 1, dtype=np.float64)
    # enforce monotone non-decreasing in p (non-increasing scanning from top)
    qr = np.minimum.accumulate(qr[::-1])[::-1]
    qr = np.clip(qr, 0.0, 1.0)
    out = np.empty(m, dtype=np.float64)
    out[order] = qr
    q[mask] = out
    return q


def storey_pi0(pvals: np.ndarray, lam: float = 0.5) -> float:
    """Storey's pi0 (estimated null proportion) at a single lambda. Clamped to <=1."""
    p = np.asarray(pvals, dtype=np.float64)
    p = p[~np.isnan(p)]
    m = p.size
    if m == 0:
        return float("nan")
    pi0 = float((p > lam).sum()) / ((1.0 - lam) * m)
    return float(min(pi0, 1.0))
