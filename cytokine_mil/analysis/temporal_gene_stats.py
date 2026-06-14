"""
Distributional statistics of induced genes across the Sheu time course, for
comparing EARLY/source vs LATE/downstream genes BEYOND THE MEAN — variance,
dispersion, bimodality, recruitment, inequality — and how those evolve over
real biological time.

Direction-AGNOSTIC by construction. The early->late ORDER is already established
(validated onset-time precedence). This module characterizes the distribution-
SHAPE signature of that order; it does NOT re-derive direction or causation
(the symmetric-correlation / cross_asym trap). Every statistic measures
*existence of heterogeneity*, never *who is upstream*.

THE central trap (designed around, mandatory): in count/log single-cell data,
variance / IQR / range / (1 - dropout) all rise mechanically with the mean.
Late ISGs reach higher means by 3-8h, so any RAW spread stat is larger for them
for purely arithmetic reasons — the exact effect-size/SNR confound that sank the
project's learning-order experiment. Every spread statistic is therefore paired
with a mean-DECOUPLED form:
  (1) residual-from-panel-trend  (fit f(mean) across ALL panel genes per timepoint)
  (2) matched-mean stratification (compare within mean-deciles)
and a claim only counts if BOTH agree. n_effective = #pseudo-donors, NOT #cells.

Count-space stats (mean_counts, var_counts, fano, cv2, gini, frac_expressing) use
RAW counts (Poisson reference F=1 is only defined there). Shape stats (skew,
kurtosis, Sarle BC, dip, IQR/QCD) use log1p(normalize_total). scipy.stats for
moments; optional `diptest` for Hartigan's dip with a Sarle-BC + Poisson-null
fallback. No model training, no h5ad/manifest — pure arrays.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional, C-backed Hartigan dip
    import diptest as _diptest
    HAVE_DIPTEST = True
except Exception:
    HAVE_DIPTEST = False

try:
    from scipy import stats as _scs
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

EPS = 1e-8
BC_THRESHOLD = 0.555            # Sarle bimodality coefficient flag
EXPR_BAND = (0.10, 0.90)        # frac_expressing band where bimodality is interpretable


# ===========================================================================
# Per-(stimulus, timepoint, donor) per-gene statistic battery
# ===========================================================================
def _skew_kurt(logx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fisher-Pearson skew (g1, bias-corrected) and Pearson kurtosis (NON-excess,
    bias-corrected) per column. Columns with ~zero variance -> 0 skew, 3 kurt."""
    n = logx.shape[0]
    if HAVE_SCIPY and n >= 4:
        g1 = _scs.skew(logx, axis=0, bias=False, nan_policy="omit")
        k_ex = _scs.kurtosis(logx, axis=0, fisher=True, bias=False, nan_policy="omit")
        kurt = np.asarray(k_ex, dtype=float) + 3.0   # back to Pearson (~3 normal)
        g1 = np.asarray(g1, dtype=float)
    else:  # manual fallback
        mu = logx.mean(0); sd = logx.std(0) + EPS
        z = (logx - mu) / sd
        g1 = (z ** 3).mean(0)
        kurt = (z ** 4).mean(0)
    g1 = np.nan_to_num(g1, nan=0.0)
    kurt = np.nan_to_num(kurt, nan=3.0)
    return g1, kurt


def sarle_bc(logx: np.ndarray) -> np.ndarray:
    """Sarle's bimodality coefficient per column: (g1^2 + 1)/kurt_corrected.
    > 0.555 suggests bimodality (uniform = 0.555, normal -> 0.33)."""
    n = logx.shape[0]
    if n < 4:
        return np.full(logx.shape[1], np.nan)
    g1, kurt = _skew_kurt(logx)
    denom = kurt + (3.0 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    bc = (g1 ** 2 + 1.0) / np.where(denom > 0, denom, np.nan)
    return bc


def gini_columns(counts: np.ndarray) -> np.ndarray:
    """Gini coefficient per column over nonneg counts (scale-invariant inequality).
    0 = all cells equal; ->1 = concentrated in few cells. NaN if column sums to 0."""
    n = counts.shape[0]
    Xs = np.sort(np.clip(counts, 0, None), axis=0)
    idx = np.arange(1, n + 1)[:, None]
    num = np.sum((2 * idx - n - 1) * Xs, axis=0)
    den = n * np.sum(Xs, axis=0)
    return num / np.where(den > 0, den, np.nan)


def _dip_columns(logx: np.ndarray) -> np.ndarray:
    """Hartigan dip per column (if diptest available), else NaN."""
    if not HAVE_DIPTEST:
        return np.full(logx.shape[1], np.nan)
    out = np.full(logx.shape[1], np.nan)
    for g in range(logx.shape[1]):
        col = logx[:, g]
        col = col[np.isfinite(col)]
        if col.size >= 10 and col.std() > EPS:
            try:
                out[g] = float(_diptest.dipstat(col))
            except Exception:
                pass
    return out


def gene_cell_stats(counts: np.ndarray, logx: np.ndarray,
                    expr_only_shape: bool = True) -> Dict[str, np.ndarray]:
    """Compute the full per-gene statistic battery for ONE (stimulus, timepoint,
    donor) cell block.

    Args:
        counts : (n_cells, n_genes) RAW counts.
        logx   : (n_cells, n_genes) log1p(normalize_total) expression.
        expr_only_shape : also compute Sarle BC on expressing-cells-only (dropout control).
    Returns dict of (n_genes,) arrays. n_cells recorded under "n_cells".
    """
    counts = np.asarray(counts, dtype=np.float64)
    logx = np.asarray(logx, dtype=np.float64)
    n, G = counts.shape

    mean_counts = counts.mean(0)
    var_counts = counts.var(0, ddof=1) if n > 1 else np.zeros(G)
    mean_log = logx.mean(0)
    var_log = logx.var(0, ddof=1) if n > 1 else np.zeros(G)

    fano = var_counts / (mean_counts + EPS)
    cv2 = var_counts / (mean_counts ** 2 + EPS)
    frac_expr = (counts > 0).mean(0)
    frac_expr2 = (counts >= 2).mean(0)
    gini = gini_columns(counts)

    q = np.percentile(logx, [10, 25, 50, 75, 90], axis=0)
    iqr = q[3] - q[1]
    qcd = (q[3] - q[1]) / (q[3] + q[1] + EPS)
    range9010 = q[4] - q[0]

    g1, kurt = _skew_kurt(logx)
    bc = sarle_bc(logx)
    dip = _dip_columns(logx)

    out = {
        "n_cells": np.full(G, n),
        "mean_counts": mean_counts, "var_counts": var_counts,
        "mean_log": mean_log, "var_log": var_log,
        "fano_raw": fano, "cv2": cv2,
        "frac_expr": frac_expr, "frac_expr2": frac_expr2,
        "gini": gini, "iqr": iqr, "qcd": qcd, "range9010": range9010,
        "skew": g1, "kurt": kurt, "bc_sarle": bc, "dip": dip,
    }

    if expr_only_shape:
        bc_e = np.full(G, np.nan)
        for gi in range(G):
            col = logx[:, gi][counts[:, gi] > 0]
            if col.size >= 8 and col.std() > EPS:
                bc_e[gi] = sarle_bc(col[:, None])[0]
        out["bc_sarle_expr"] = bc_e
    return out


# ===========================================================================
# Mean-decoupling: residual from the per-timepoint panel trend
# ===========================================================================
def trend_residual(x: np.ndarray, y: np.ndarray, deg: int = 2,
                   support: Optional[np.ndarray] = None,
                   min_pts: int = 25, standardize: bool = True) -> np.ndarray:
    """Standardized residual of y ~ poly_deg(x), fit across genes (the panel trend).

    Used for fano_residual (x=log mean_counts, y=log var_counts) and
    frac_expressing_residual (x=log mean_counts, y=frac_expr). The fit `support`
    (which genes enter the fit) MUST be the full panel of expressed genes, never
    the cascade genes — this is what makes the residual a fair "excess beyond what
    the mean predicts" at matched mean.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    res = np.full(x.shape, np.nan)
    fit_m = np.isfinite(x) & np.isfinite(y)
    if support is not None:
        fit_m = fit_m & support
    if fit_m.sum() < min_pts:
        return res
    coef = np.polyfit(x[fit_m], y[fit_m], deg)
    pred = np.polyval(coef, x)
    r = y - pred
    if standardize:
        sd = np.nanstd(r[fit_m])
        r = r / sd if sd > 0 else r
    out_m = np.isfinite(x) & np.isfinite(y)
    res[out_m] = r[out_m]
    return res


def fano_residual(stats: Dict[str, np.ndarray], expr_floor: float = 0.02) -> np.ndarray:
    """Standardized residual of log(var_counts) ~ poly2(log(mean_counts)) across the
    panel (PRIMARY mean-decoupled overdispersion). Positive => more overdispersed
    than its mean predicts."""
    mc = stats["mean_counts"]; vc = stats["var_counts"]
    x = np.log(mc + EPS); y = np.log(vc + EPS)
    support = (stats["frac_expr"] > expr_floor) & (mc > 0)
    return trend_residual(x, y, deg=2, support=support)


def detection_residual(stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Residual of frac_expr ~ poly2(log mean_counts) across the panel (mean-decoupled
    detection). Negative => detected in FEWER cells than its mean predicts (bursty)."""
    mc = stats["mean_counts"]
    x = np.log(mc + EPS); y = stats["frac_expr"]
    support = (mc > 0)
    return trend_residual(x, y, deg=2, support=support, standardize=False)


# ===========================================================================
# Cell-count harmonization (sampling-variance control)
# ===========================================================================
def subsample_block(counts: np.ndarray, logx: np.ndarray, n_target: int,
                    rng) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample rows (cells) to n_target so higher moments are comparable across
    timepoints (moment sampling variance ~ 1/n). No-op if n_cells <= n_target."""
    n = counts.shape[0]
    if n <= n_target:
        return counts, logx
    idx = rng.choice(n, n_target, replace=False)
    return counts[idx], logx[idx]


# ===========================================================================
# Donor-level group comparison (early vs late) — n_eff = donors, not cells
# ===========================================================================
def group_auc(late_vals: np.ndarray, early_vals: np.ndarray) -> float:
    """AUC = P(late > early) over all gene pairs (ties = 0.5). >0.5 => late larger."""
    a = np.asarray(late_vals, float); a = a[np.isfinite(a)]
    b = np.asarray(early_vals, float); b = b[np.isfinite(b)]
    if a.size < 1 or b.size < 1:
        return float("nan")
    comp = a[:, None] - b[None, :]
    return float(((comp > 0).sum() + 0.5 * (comp == 0).sum()) / (a.size * b.size))


def donor_level_auc(per_donor_vals: List[np.ndarray], is_late: np.ndarray,
                    n_perm: int = 2000, rng=None) -> Dict:
    """Donor-respecting early-vs-late test for one statistic at one timepoint.

    Args:
        per_donor_vals : list (len = n_donors) of (n_union_genes,) arrays — the
                         statistic for the union of early+late genes, per donor.
        is_late : (n_union_genes,) bool aligning to per_donor_vals columns.
    Returns donor-median AUC, per-donor AUCs, sign-agreement (frac donors AUC>0.5),
    and a label-permutation p on the donor-median AUC (labels shuffled within donor).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    is_late = np.asarray(is_late, bool)
    aucs = []
    for v in per_donor_vals:
        v = np.asarray(v, float)
        aucs.append(group_auc(v[is_late], v[~is_late]))
    aucs = np.array(aucs, float)
    finite = aucs[np.isfinite(aucs)]
    if finite.size == 0:
        return {"auc": float("nan"), "per_donor": aucs.tolist(), "n_donors": 0,
                "frac_late_gt": float("nan"), "p": float("nan")}
    obs = float(np.median(finite))
    # permutation: shuffle late/early labels (same per donor) -> null donor-median AUC
    null = np.empty(n_perm)
    n_lab = is_late.size
    for k in range(n_perm):
        perm = rng.permutation(n_lab)
        lab = is_late[perm]
        pa = [group_auc(np.asarray(v, float)[lab], np.asarray(v, float)[~lab])
              for v in per_donor_vals]
        pa = np.array(pa, float); pa = pa[np.isfinite(pa)]
        null[k] = np.median(pa) if pa.size else 0.5
    # two-sided around 0.5
    p = float((np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (n_perm + 1))
    return {"auc": obs, "per_donor": [float(a) for a in aucs],
            "n_donors": int(finite.size),
            "frac_late_gt": float(np.mean(finite > 0.5)), "p": p}


def matched_mean_delta(values: np.ndarray, mean_key: np.ndarray, is_late: np.ndarray,
                       n_bins: int = 10) -> Dict:
    """Within mean-decile, (median late - median early) of `values`; averaged across
    occupied bins. The design-based mean-control complementary to the residual fit.
    Sign must agree with the residual-route AUC for a claim to count."""
    values = np.asarray(values, float); mk = np.asarray(mean_key, float)
    is_late = np.asarray(is_late, bool)
    ok = np.isfinite(values) & np.isfinite(mk)
    if ok.sum() < 6:
        return {"delta": float("nan"), "per_bin": [], "n_bins_used": 0}
    edges = np.quantile(mk[ok], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    deltas, per_bin = [], []
    for b in range(n_bins):
        m = ok & (mk >= edges[b]) & (mk < edges[b + 1])
        lv = values[m & is_late]; ev = values[m & ~is_late]
        if lv.size >= 1 and ev.size >= 1:
            d = float(np.median(lv) - np.median(ev))
            deltas.append(d)
            per_bin.append({"bin": b, "n_late": int(lv.size), "n_early": int(ev.size), "delta": d})
    if not deltas:
        return {"delta": float("nan"), "per_bin": [], "n_bins_used": 0}
    return {"delta": float(np.mean(deltas)), "per_bin": per_bin, "n_bins_used": len(deltas)}


# ===========================================================================
# Per-gene trajectory features (over the time course of a decoupled stat)
# ===========================================================================
def _first_cross(traj: np.ndarray, times: np.ndarray, thr: float) -> float:
    idx = np.where(np.nan_to_num(traj, nan=-np.inf) >= thr)[0]
    return float(times[idx[0]]) if idx.size else float("nan")


def peak_time(traj: np.ndarray, times: np.ndarray) -> float:
    t = np.nan_to_num(traj, nan=-np.inf)
    if not np.any(np.isfinite(traj)) or np.nanmax(traj) <= -np.inf:
        return float("nan")
    return float(times[int(np.argmax(t))])


def transience_index(traj: np.ndarray) -> float:
    """(max - last) / (max - first); ~1 transient bump that resolves, ~0 monotone."""
    tr = traj[np.isfinite(traj)]
    if tr.size < 3:
        return float("nan")
    mx = np.nanmax(traj); first = tr[0]; last = tr[-1]
    denom = mx - first
    return float((mx - last) / denom) if abs(denom) > EPS else float("nan")


def recruitment_features(frac_traj: np.ndarray, times: np.ndarray) -> Dict:
    """t50 (time to 50% of final detection) + max discrete slope of frac_expressing."""
    fr = np.asarray(frac_traj, float)
    fin = fr[np.isfinite(fr)]
    if fin.size < 2 or np.nanmax(fr) <= 0:
        return {"t50": float("nan"), "max_slope": float("nan")}
    final = fin[-1]
    t50 = _first_cross(fr, times, 0.5 * final)
    dt = np.diff(times)
    slopes = np.diff(fr) / np.where(dt > 0, dt, np.nan)
    return {"t50": t50, "max_slope": float(np.nanmax(slopes)) if np.any(np.isfinite(slopes)) else float("nan")}


def trapz_auc(traj: np.ndarray, times: np.ndarray, t_max: float = 3.0) -> float:
    """Trapezoid AUC of a trajectory over t <= t_max (early-window excess)."""
    t = np.asarray(times, float); y = np.asarray(traj, float)
    m = np.isfinite(y) & (t <= t_max + 1e-9)
    if m.sum() < 2:
        return float("nan")
    return float(np.trapz(np.nan_to_num(y[m]), t[m]))


def module_coherence(logx: np.ndarray, gene_idx: Sequence[int]) -> float:
    """Mean off-diagonal Spearman r across cells WITHIN a gene set (co-expression
    coherence). CONSISTENCY check only — never direction evidence."""
    idx = [g for g in gene_idx]
    if len(idx) < 2:
        return float("nan")
    sub = logx[:, idx]
    # rank then Pearson = Spearman
    ranks = np.argsort(np.argsort(sub, axis=0), axis=0).astype(float)
    sd = ranks.std(0)
    keep = sd > EPS
    if keep.sum() < 2:
        return float("nan")
    R = np.corrcoef(ranks[:, keep], rowvar=False)
    iu = np.triu_indices_from(R, k=1)
    vals = R[iu]
    return float(np.nanmean(vals)) if vals.size else float("nan")


# ===========================================================================
# Poisson dropout null for bimodality (is a BC/dip flag biological or just zeros?)
# ===========================================================================
def poisson_bc_null_p(counts_col: np.ndarray, observed_bc: float,
                      n_null: int = 200, rng=None) -> float:
    """Empirical p that a gene's Sarle BC exceeds a Poisson resample at its own mean
    (tests whether the bimodality is more than shot-noise zero-inflation)."""
    if rng is None:
        rng = np.random.default_rng(0)
    if not np.isfinite(observed_bc):
        return float("nan")
    n = counts_col.size
    lam = counts_col.mean()
    if lam <= 0:
        return float("nan")
    null = np.empty(n_null)
    for k in range(n_null):
        c = rng.poisson(lam, n).astype(float)
        null[k] = sarle_bc(np.log1p(c)[:, None])[0]
    return float((np.sum(null >= observed_bc) + 1) / (n_null + 1))


# ===========================================================================
# Synthetic apparatus self-test
# ===========================================================================
def simulate_early_late(n_genes_bg: int = 80, n_cells: int = 600,
                        time_hrs: Sequence[float] = (0, 0.25, 0.5, 1, 3, 5, 8),
                        rng=None) -> Dict:
    """Plant EARLY genes (uniform shift, fast synchronous detection, low/no excess
    dispersion, Fano~1) and LATE genes (fraction-ON ramps 0->1 through 1-3h:
    transient bimodality + overdispersion + gradual recruitment), plus flat
    background. Crucially, LATE genes also reach a HIGH MEAN by 8h (same as a real
    cascade), so the apparatus must show late>early in DECOUPLED stats while raw
    mean barely separates by the late timepoints.

    Returns {"blocks": {t: (counts(n,G), logx(n,G))}, "early_idx", "late_idx",
             "gene_names", "time_hrs"}.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    times = np.asarray(time_hrs, float)
    early = [f"early{i}" for i in range(4)]
    late = [f"late{i}" for i in range(8)]
    bg = [f"bg{i}" for i in range(n_genes_bg)]
    names = early + late + bg
    G = len(names)
    n_e, n_l = len(early), len(late)
    base_lam = 0.6  # background Poisson rate

    def logistic(t, t0, k=2.0):
        return 1.0 / (1.0 + np.exp(-k * (t - t0)))

    blocks = {}
    for t in times:
        lam = np.full((n_cells, G), base_lam)
        # EARLY: uniform amplitude shift in ALL cells, switches on ~0.5h, low extra noise
        amp_e = 6.0 * logistic(t, 0.5)
        lam[:, :n_e] = base_lam + amp_e            # every cell same -> Fano~1, unimodal
        # LATE: a FRACTION of cells turns fully on; the fraction ramps 0->1 through 1-3h.
        frac_on = logistic(t, 2.0, k=1.6)          # responder fraction over time
        for j in range(n_l):
            gi = n_e + j
            on = rng.random(n_cells) < frac_on
            lam[on, gi] = base_lam + 7.0           # ON cells high
            lam[~on, gi] = base_lam                # OFF cells baseline -> bimodal mixture
        counts = rng.poisson(np.clip(lam, 0, None)).astype(np.float64)
        # normalize_total(1e4) + log1p to mirror the real pipeline
        tot = counts.sum(1, keepdims=True); tot[tot == 0] = 1.0
        norm = counts / tot * 1e4
        logx = np.log1p(norm)
        blocks[float(t)] = (counts, logx)
    return {"blocks": blocks, "early_idx": list(range(n_e)),
            "late_idx": list(range(n_e, n_e + n_l)),
            "gene_names": names, "time_hrs": [float(x) for x in times]}
