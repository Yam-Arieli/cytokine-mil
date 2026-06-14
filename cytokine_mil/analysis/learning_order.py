"""
Gene learning-order analysis (the user's hypothesis: cascade-SOURCE genes are learned
before DOWNSTREAM genes). numpy-only analysis layer.

Inputs come from a per-epoch per-gene attribution trajectory (genes x epochs, magnitude)
+ per-gene effect size (log2FC vs PBS) + ground-truth source/downstream gene masks +
(optionally) a per-gene real-time emergence from the raw Sheu time course.

Two controls make a "source-first" result meaningful, because training-order is largely
driven by learnability/SNR (high-effect genes learned first), not causal source:
  H1  effect-size-MATCHED permutation test (within log2FC quantile bins).
  H2  PARTIAL Spearman of training-emergence vs real-time-emergence, controlling effect size.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Emergence epoch from a per-gene magnitude trajectory
# ---------------------------------------------------------------------------
def emergence_epoch(traj_mag, epochs, frac: float = 0.5, floor: Optional[float] = None):
    """traj_mag: (n_genes, n_epochs) NONNEG attribution magnitude. Returns (n_genes,)
    emergence epoch = first epoch reaching frac*final; NaN if final < floor.
    floor defaults to 0.2 * median(final over genes with positive final)."""
    traj = np.abs(np.asarray(traj_mag, dtype=np.float64))
    epochs = np.asarray(epochs)
    final = traj[:, -1]
    if floor is None:
        pos = final[final > 0]
        floor = 0.2 * float(np.median(pos)) if pos.size else 0.0
    out = np.full(traj.shape[0], np.nan)
    for g in range(traj.shape[0]):
        if final[g] < floor or final[g] <= 0:
            continue
        idx = np.where(traj[g] >= frac * final[g])[0]
        out[g] = epochs[idx[0]] if idx.size else epochs[-1]
    return out


# ---------------------------------------------------------------------------
# Rank helpers (numpy-only)
# ---------------------------------------------------------------------------
def _rank(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(np.argsort(a))
    return order.astype(np.float64)


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = _rank(x[m]), _rank(y[m])
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _resid(a, b):
    """residual of a on [1, b] (least squares)."""
    B = np.c_[np.ones_like(b), b]
    coef, *_ = np.linalg.lstsq(B, a, rcond=None)
    return a - B @ coef


def partial_spearman(x, y, z):
    """Partial Spearman of x,y controlling z (corr of rank-residuals)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 4:
        return float("nan")
    rx, ry, rz = _rank(x[m]), _rank(y[m]), _rank(z[m])
    ex, ey = _resid(rx, rz), _resid(ry, rz)
    if ex.std() < 1e-9 or ey.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(ex, ey)[0, 1])


# ---------------------------------------------------------------------------
# H1 — source before downstream, raw + effect-size-matched permutation
# ---------------------------------------------------------------------------
def h1_source_first(emergence, effsize, source_mask, downstream_mask,
                    n_perm: int = 2000, n_bins: int = 5, rng=None) -> Dict[str, float]:
    """stat = median(emergence[downstream]) - median(emergence[source]); >0 => source earlier.
    p_raw: permute source/downstream labels freely. p_matched: permute WITHIN effect-size
    quantile bins (controls for the SNR confound)."""
    if rng is None:
        rng = np.random.default_rng(0)
    emergence = np.asarray(emergence, float)
    src = np.where(source_mask & np.isfinite(emergence))[0]
    dwn = np.where(downstream_mask & np.isfinite(emergence))[0]
    if len(src) < 2 or len(dwn) < 2:
        return {"observed": float("nan"), "p_raw": float("nan"), "p_matched": float("nan"),
                "n_source": int(len(src)), "n_downstream": int(len(dwn))}
    pool = np.concatenate([src, dwn])
    is_src = np.concatenate([np.ones(len(src), bool), np.zeros(len(dwn), bool)])
    em_pool = emergence[pool]

    def stat(mask_src):
        return float(np.median(em_pool[~mask_src]) - np.median(em_pool[mask_src]))
    observed = stat(is_src)

    # raw permutation
    raw = np.array([stat(rng.permutation(is_src)) for _ in range(n_perm)])
    p_raw = float((np.sum(raw >= observed) + 1) / (n_perm + 1))

    # effect-size-matched permutation: bins by effsize quantiles, shuffle labels within bin
    ef = np.asarray(effsize, float)[pool]
    edges = np.quantile(ef[np.isfinite(ef)], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    binid = np.digitize(ef, edges[1:-1])
    matched = np.empty(n_perm)
    for k in range(n_perm):
        perm = is_src.copy()
        for b in np.unique(binid):
            idx = np.where(binid == b)[0]
            perm[idx] = rng.permutation(perm[idx])
        matched[k] = stat(perm)
    p_matched = float((np.sum(matched >= observed) + 1) / (n_perm + 1))
    return {"observed": observed, "p_raw": p_raw, "p_matched": p_matched,
            "n_source": int(len(src)), "n_downstream": int(len(dwn)),
            "median_source": float(np.median(emergence[src])),
            "median_downstream": float(np.median(emergence[dwn]))}


# ---------------------------------------------------------------------------
# H2 — training-emergence vs real-time-emergence, controlling effect size
# ---------------------------------------------------------------------------
def h2_realtime(train_emergence, realtime_emergence, effsize,
                n_perm: int = 2000, rng=None) -> Dict[str, float]:
    """Spearman + PARTIAL Spearman (controlling effect size) of training vs real-time
    emergence. Partial>0 (perm p<=0.05) => learning order tracks real biological time
    beyond SNR. Permutation null shuffles training-emergence."""
    if rng is None:
        rng = np.random.default_rng(0)
    x, y, z = (np.asarray(v, float) for v in (train_emergence, realtime_emergence, effsize))
    spear = _spearman(x, y)
    partial = partial_spearman(x, y, z)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    xs, ys, zs = x[m], y[m], z[m]
    null = np.array([partial_spearman(rng.permutation(xs), ys, zs) for _ in range(n_perm)])
    p_partial = float((np.sum(null >= partial) + 1) / (n_perm + 1)) if np.isfinite(partial) else float("nan")
    return {"spearman": spear, "partial_spearman": partial, "p_partial": p_partial,
            "n_genes": int(m.sum())}


# ---------------------------------------------------------------------------
# Synthetic generator — two regimes for the apparatus self-test
# ---------------------------------------------------------------------------
def simulate_learning_trajectories(regime: str, n_genes: int = 80, n_source: int = 10,
                                   n_down: int = 14, n_epochs: int = 60, rng=None):
    """Per-gene magnitude trajectories + effect size + masks + real-time emergence.
      'cascade_order': emergence set by ROLE (source early, downstream late) at MATCHED
                       effect size; real-time emergence tracks training emergence.
                       -> H1_matched PASS, H2 PASS.
      'snr_confound' : emergence set by EFFECT SIZE only, and source genes given HIGHER
                       effect size (so source looks early via SNR); real-time emergence
                       depends on effect size only.
                       -> H1_raw PASS but H1_matched FAIL, H2 (partial) FAIL.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    epochs = np.arange(n_epochs)
    src = np.arange(n_source)
    dwn = np.arange(n_source, n_source + n_down)
    source_mask = np.zeros(n_genes, bool); source_mask[src] = True
    downstream_mask = np.zeros(n_genes, bool); downstream_mask[dwn] = True

    effsize = rng.uniform(0.5, 3.0, size=n_genes)
    e_true = np.empty(n_genes)
    realtime = np.full(n_genes, np.nan)

    if regime == "cascade_order":
        # role drives emergence; matched effect size for source vs downstream
        effsize[src] = rng.uniform(1.0, 2.5, size=n_source)
        effsize[dwn] = rng.uniform(1.0, 2.5, size=n_down)       # same dist -> matched
        e_true[:] = rng.uniform(0.1, 0.9, n_genes) * n_epochs
        e_true[src] = rng.uniform(0.10, 0.30, n_source) * n_epochs
        e_true[dwn] = rng.uniform(0.55, 0.80, n_down) * n_epochs
        realtime[src] = e_true[src] + rng.normal(0, 2, n_source)  # real time tracks training
        realtime[dwn] = e_true[dwn] + rng.normal(0, 2, n_down)
    elif regime == "snr_confound":
        # emergence driven by effect size only; source given higher effect size
        effsize[src] = rng.uniform(2.0, 3.0, size=n_source)      # source = high effect
        effsize[dwn] = rng.uniform(0.8, 1.6, size=n_down)        # downstream = low effect
        norm = (effsize - effsize.min()) / (effsize.max() - effsize.min() + 1e-9)
        e_true[:] = (1.0 - norm) * (0.8 * n_epochs) + rng.normal(0, 1.5, n_genes)
        # real time depends on effect size only (NOT on training-emergence beyond effsize)
        realtime[src] = (1.0 - norm[src]) * n_epochs + rng.normal(0, 5, n_source)
        realtime[dwn] = (1.0 - norm[dwn]) * n_epochs + rng.normal(0, 5, n_down)
    else:
        raise ValueError(regime)
    e_true = np.clip(e_true, 0, n_epochs - 1)

    # build magnitude trajectories: effsize * sigmoid((t - e_true)/width) + noise
    width = max(2.0, n_epochs / 20)
    t = epochs[None, :]
    traj = effsize[:, None] / (1.0 + np.exp(-(t - e_true[:, None]) / width))
    traj = traj + rng.normal(0, 0.02, traj.shape)
    traj = np.clip(traj, 0, None)
    return {"traj": traj, "epochs": epochs, "effsize": effsize,
            "source_mask": source_mask, "downstream_mask": downstream_mask,
            "realtime_emergence": realtime}
