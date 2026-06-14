"""
Time-resolved gene cascades (Sheu time course): order genes by activation time and
infer directed gene->gene edges from temporal precedence. Direction comes from REAL
biological time (no SNR confound, unlike the snapshot learning-order approach).

Input: per-stimulus PBS-corrected per-gene-per-time matrix `above_baseline` (n_genes, n_t)
       + `time_hrs` (sorted hours). (Produced by compute_sheu_realtime_emergence._compute_time_series.)

Caveat: temporal precedence is necessary-but-NOT-sufficient for causation (a common upstream
driver can create apparent edges). This is "cascade order from real kinetics", validated on
known cascades — not proven causation. numpy-only analysis layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Per-gene timing
# ---------------------------------------------------------------------------
def activation_time(above_baseline, time_hrs, frac: float = 0.5):
    """First time (hr) each gene crosses frac*max(above_baseline). NaN if max<=0."""
    times = np.asarray(time_hrs, dtype=np.float64)
    out = np.full(above_baseline.shape[0], np.nan)
    for g in range(above_baseline.shape[0]):
        traj = above_baseline[g]
        mx = np.nanmax(traj) if np.any(np.isfinite(traj)) else np.nan
        if not np.isfinite(mx) or mx <= 0:
            continue
        idx = np.where(np.nan_to_num(traj, nan=-np.inf) >= frac * mx)[0]
        if idx.size:
            out[g] = times[idx[0]]
    return out


def peak_time(above_baseline, time_hrs):
    times = np.asarray(time_hrs, dtype=np.float64)
    out = np.full(above_baseline.shape[0], np.nan)
    for g in range(above_baseline.shape[0]):
        traj = above_baseline[g]
        if not np.any(np.isfinite(traj)) or np.nanmax(traj) <= 0:
            continue
        out[g] = times[int(np.nanargmax(np.nan_to_num(traj, nan=-np.inf)))]
    return out


def induced_mask(above_baseline, floor: float):
    """Genes with a clear induction (max above-baseline >= floor)."""
    mx = np.nanmax(np.where(np.isfinite(above_baseline), above_baseline, -np.inf), axis=1)
    return mx >= floor


# ---------------------------------------------------------------------------
# Lead-lag (cross-correlation on interpolated trajectories)
# ---------------------------------------------------------------------------
def _interp(above_baseline, time_hrs, n: int = 60):
    times = np.asarray(time_hrs, dtype=np.float64)
    grid = np.linspace(times.min(), times.max(), n)
    G = above_baseline.shape[0]
    out = np.zeros((G, n))
    for g in range(G):
        traj = above_baseline[g]
        m = np.isfinite(traj)
        out[g] = np.interp(grid, times[m], traj[m]) if m.sum() >= 2 else 0.0
    return out, grid


def _norm01(x):
    x = x - x.min()
    s = x.max()
    return x / s if s > 0 else x


def lead_lag(a, b, grid):
    """Lag (hr) at which a leads b (>0 => a rises earlier), by max cross-correlation."""
    an, bn = _norm01(a), _norm01(b)
    n = len(grid); dt = grid[1] - grid[0]; maxshift = n // 2
    best_lag, best_c = 0.0, -np.inf
    for s in range(-maxshift, maxshift + 1):
        if s >= 0:
            x, y = an[s:], bn[:n - s]
        else:
            x, y = an[:n + s], bn[-s:]
        if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best_c:
            best_c, best_lag = c, s * dt
    return best_lag


# ---------------------------------------------------------------------------
# Directed edges by temporal precedence
# ---------------------------------------------------------------------------
def directed_edges(above_baseline, time_hrs, gene_idx, gene_names, act,
                   margin: float = 0.5, require_lead: bool = True):
    """A->B if act[A] < act[B]-margin (precedence) AND (optionally) A leads B by lead-lag.
    Restricted to `gene_idx` (induced genes). Returns list of edge dicts."""
    interp, grid = _interp(above_baseline, time_hrs)
    idx = [g for g in gene_idx if np.isfinite(act[g])]
    edges = []
    for i in idx:
        for j in idx:
            if i == j or not (act[i] < act[j] - margin):
                continue
            ll = lead_lag(interp[i], interp[j], grid)
            if (not require_lead) or ll > 0:
                edges.append({"src": gene_names[i], "dst": gene_names[j],
                              "dt": float(act[j] - act[i]), "lead_lag": float(ll)})
    return edges


# ---------------------------------------------------------------------------
# Validation against known source/downstream labels
# ---------------------------------------------------------------------------
def _auc_earlier(sv, dv):
    """AUC = P(source activates earlier than downstream) over all pairs (ties=0.5).
    >0.5 => source earlier. Powerful + robust to the coarse, tied time grid (unlike
    a median difference)."""
    comp = sv[:, None] - dv[None, :]
    return float(((comp < 0).sum() + 0.5 * (comp == 0).sum()) / (len(sv) * len(dv)))


def validate_source_downstream(act, source_idx, downstream_idx, n_perm: int = 2000, rng=None):
    """V1: source genes activate EARLIER than downstream. Statistic = AUC (Mann-Whitney:
    fraction of source-before-downstream pairs). Permutation p (shuffle the labels)."""
    if rng is None:
        rng = np.random.default_rng(0)
    s = act[list(source_idx)]; s = s[np.isfinite(s)]
    d = act[list(downstream_idx)]; d = d[np.isfinite(d)]
    if len(s) < 2 or len(d) < 2:
        return {"auc": float("nan"), "p": float("nan"), "median_source": float("nan"),
                "median_downstream": float("nan"), "delta": float("nan"),
                "n_source": int(len(s)), "n_downstream": int(len(d))}
    obs = _auc_earlier(s, d)
    pool = np.concatenate([s, d]); ns = len(s)
    perm = np.array([_auc_earlier(p[:ns], p[ns:]) for p in (rng.permutation(pool) for _ in range(n_perm))])
    p = float((np.sum(perm >= obs) + 1) / (n_perm + 1))
    return {"auc": obs, "p": p, "median_source": float(np.median(s)),
            "median_downstream": float(np.median(d)),
            "delta": float(np.median(d) - np.median(s)),
            "n_source": int(len(s)), "n_downstream": int(len(d))}


def edge_direction_fraction(edges, source_set, downstream_set):
    """V2: among edges between source & downstream genes, fraction running source->downstream."""
    src_set, dwn_set = set(source_set), set(downstream_set)
    rel = [e for e in edges
           if (e["src"] in src_set and e["dst"] in dwn_set)
           or (e["src"] in dwn_set and e["dst"] in src_set)]
    if not rel:
        return {"n": 0, "frac_src_to_down": float("nan")}
    fwd = sum(1 for e in rel if e["src"] in src_set and e["dst"] in dwn_set)
    return {"n": len(rel), "frac_src_to_down": float(fwd / len(rel))}


# ---------------------------------------------------------------------------
# Synthetic two-wave cascade (apparatus self-test)
# ---------------------------------------------------------------------------
def simulate_time_cascade(n_genes: int = 60, n_source: int = 8, n_down: int = 12,
                          time_hrs: Sequence[float] = (0, 0.25, 0.5, 1, 3, 5, 8),
                          noise: float = 0.03, rng=None):
    """Plant SOURCE genes rising ~0.5-1h and DOWNSTREAM genes rising ~3-5h (a real
    early->late cascade), plus flat background. Returns (above_baseline, time_hrs,
    source_idx, downstream_idx)."""
    if rng is None:
        rng = np.random.default_rng(0)
    times = np.asarray(time_hrs, dtype=np.float64)
    src = np.arange(n_source)
    dwn = np.arange(n_source, n_source + n_down)
    G = n_genes
    ab = np.zeros((G, len(times)))

    def sigmoid_rise(t0, amp):
        return amp / (1.0 + np.exp(-(times - t0) / 0.4))

    for g in src:
        ab[g] = sigmoid_rise(rng.uniform(0.4, 0.9), rng.uniform(1.0, 2.5))
    for g in dwn:
        ab[g] = sigmoid_rise(rng.uniform(3.0, 5.0), rng.uniform(1.0, 2.5))
    # background: flat ~0 (a few weak random bumps)
    for g in range(n_source + n_down, G):
        ab[g] = rng.normal(0, 0.05, len(times))
    ab = ab + rng.normal(0, noise, ab.shape)
    ab[:, 0] = 0.0   # t=0 is PBS baseline (above-baseline == 0)
    return ab, list(times), src, dwn
