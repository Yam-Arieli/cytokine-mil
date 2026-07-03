"""
Source-potency from training dynamics — a per-cytokine cascade-source score.

`cascadir`/`cross_asym` give directed *edges* (who is upstream in a pair). This
module adds a per-*node* scalar the edge graph lacks: how strongly a cytokine
acts as a cascade *source* (its depth×width "butterfly" impact), read from the
SHAPE of its per-cytokine learning curve in a multiclass model.

Rationale (two assumptions):
  (ML)  a model learns strong primary patterns first, weak secondary patterns late.
  (bio) a cytokine with a bigger downstream cascade injects more weak secondary
        signatures (from the cytokines it induces) into its own tubes.
  => the LATER / SLOWER a cytokine's accuracy plateaus, the richer its cascade.

Score components (per cytokine, on the donor-level p_correct trajectory):
  - P_max                 = ceiling = max over epochs (the early figure's y-axis).
  - normalized_traj_auc   = trapz(traj / P_max) / (n-1)  (the early figure's x-axis;
                            HIGH = plateaus early = shallow).
  - plateau_epoch         = first epoch reaching >= frac * P_max (LATE = deep).
  - late_phase_gain       = rise in p_correct over the final third (§8.3 secondary rise).
  - source_potency        = z(1 - normalized_traj_auc) + z(late_phase_gain), computed
                            over the cytokines that clear the ceiling floor.

CRITICAL confound: a cytokine that is simply HARD / never learned (low P_max) also
plateaus late — that is "unlearnable", not "deep cascade". Read the score only among
cytokines with P_max above a floor (`ceiling_floor`), and always report shape vs P_max.

Pure numpy — consumes the `records` list from train_mil / dynamics.pkl and the two
axes CSVs (as DataFrames). No torch, no file IO, no model.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cytokine_mil.analysis.attention_dynamics import spearman  # numpy-only, no scipy
from cytokine_mil.analysis.dynamics import aggregate_to_donor_level

# Pre-registered pools (provenance: scripts/run_bootstrap.py SIMPLE_POOL/COMPLEX_POOL,
# the same red=deep / blue=shallow sets as the early learnability figure).
SHALLOW_POOL = ["IL-4", "IL-10", "IL-2", "M-CSF", "TNF-alpha",
                "IL-1-beta", "IFN-beta", "IL-7", "G-CSF"]
DEEP_POOL = ["IL-12", "IL-32-beta", "OSM", "IL-22", "VEGF", "HGF", "TGF-beta1", "IL-6"]

# Literature master-regulators (descriptive sanity, not graded).
MASTER_REGULATORS = ["TNF-alpha", "IL-1-beta", "IFN-beta", "IFN-gamma", "IL-12", "GM-CSF"]


# ---------------------------------------------------------------------------
# Per-cytokine curve-shape metrics (on a single donor-mean trajectory)
# ---------------------------------------------------------------------------

def ceiling(traj: Sequence[float]) -> float:
    """P_max: the plateau level = max of the trajectory over epochs."""
    a = np.asarray(traj, dtype=np.float64)
    return float(a.max()) if a.size else 0.0


def normalized_trajectory_auc(traj: Sequence[float]) -> float:
    """AUC of traj / P_max, trapezoid, /(n-1). HIGH => plateaus early (shallow).

    Matches the early figure's x-axis. In [0, 1]; 0 if degenerate.
    """
    a = np.asarray(traj, dtype=np.float64)
    if a.size < 2:
        return 0.0
    pm = a.max()
    if pm <= 0:
        return 0.0
    return float(np.trapz(a / pm) / (a.size - 1))


def plateau_epoch(
    traj: Sequence[float], epochs: Sequence[int], frac: float = 0.9
) -> Optional[int]:
    """First epoch reaching >= frac * P_max. LATE => deep cascade.

    Returns the epoch value, or None if the trajectory never reaches it
    (only when P_max <= 0; otherwise the max itself always qualifies).
    """
    a = np.asarray(traj, dtype=np.float64)
    ep = list(epochs)
    if a.size == 0 or a.size != len(ep):
        return None
    pm = a.max()
    if pm <= 0:
        return None
    crossed = np.where(a >= frac * pm)[0]
    return int(ep[crossed[0]]) if crossed.size else None


def late_phase_gain(traj: Sequence[float], last_frac: float = 1.0 / 3.0) -> float:
    """Rise in p_correct over the final `last_frac` of training (§8.3 secondary rise).

    max(tail) - tail[0], where tail starts at the last-third boundary. >0 => the
    model kept learning late (candidate secondary/cascade patterns).
    """
    a = np.asarray(traj, dtype=np.float64)
    if a.size < 2:
        return 0.0
    start = int(np.clip(round((1.0 - last_frac) * (a.size - 1)), 0, a.size - 1))
    if start >= a.size - 1:
        return 0.0
    tail = a[start:]
    return float(tail.max() - tail[0])


# ---------------------------------------------------------------------------
# Per-cytokine table across donors (+ seeds handled by the caller)
# ---------------------------------------------------------------------------

def per_cytokine_metrics(
    records: List[Dict],
    epochs: Sequence[int],
    exclude: Optional[List[str]] = None,
    plateau_frac: float = 0.9,
    last_frac: float = 1.0 / 3.0,
) -> Dict[str, Dict[str, float]]:
    """
    Curve-shape metrics per cytokine from one run's `records` (one seed).

    Aggregation matches project convention: median across tubes per donor
    (aggregate_to_donor_level), then MEAN across donors -> one trajectory/cytokine.

    Returns {cytokine -> {'P_max','normalized_auc','plateau_epoch','late_gain'}}.
    plateau_epoch may be None (kept as np.nan for downstream averaging).
    """
    exclude_set = set(exclude or [])
    donor_trajs = aggregate_to_donor_level(records, "p_correct_trajectory")
    out: Dict[str, Dict[str, float]] = {}
    for cyt, by_donor in donor_trajs.items():
        if cyt in exclude_set:
            continue
        arrs = [np.asarray(v, dtype=np.float64) for v in by_donor.values()]
        n = min(a.size for a in arrs)
        traj = np.mean(np.stack([a[:n] for a in arrs]), axis=0)
        ep = list(epochs)[:n]
        pe = plateau_epoch(traj, ep, plateau_frac)
        out[cyt] = {
            "P_max": ceiling(traj),
            "normalized_auc": normalized_trajectory_auc(traj),
            "plateau_epoch": float(pe) if pe is not None else float("nan"),
            "late_gain": late_phase_gain(traj, last_frac),
        }
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def source_potency_table(
    metrics_by_cyt: Dict[str, Dict[str, float]],
    ceiling_floor: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    """
    Add the composite source_potency and an `included` flag (P_max >= floor).

    source_potency = z(1 - normalized_auc) + z(late_gain), z-scored over the
    INCLUDED cytokines only (so the unlearnable-late confound cannot inflate it).
    Excluded cytokines keep their raw metrics but source_potency = nan.

    Returns {cytokine -> {..metrics.., 'included':0/1, 'source_potency':float}}.
    """
    cyts = list(metrics_by_cyt)
    included = [c for c in cyts if metrics_by_cyt[c]["P_max"] >= ceiling_floor]
    lateness = _zscore(np.array([1.0 - metrics_by_cyt[c]["normalized_auc"] for c in included]))
    gain = _zscore(np.array([metrics_by_cyt[c]["late_gain"] for c in included]))
    potency = {c: float(lateness[i] + gain[i]) for i, c in enumerate(included)}
    out: Dict[str, Dict[str, float]] = {}
    for c in cyts:
        row = dict(metrics_by_cyt[c])
        row["included"] = 1.0 if c in potency else 0.0
        row["source_potency"] = potency.get(c, float("nan"))
        out[c] = row
    return out


# ---------------------------------------------------------------------------
# Ground-truth graph degrees (from the axes CSVs)
# ---------------------------------------------------------------------------

def graph_coupling_degree(axes_rows: Sequence[Dict]) -> Dict[str, int]:
    """Undirected coupling degree (width): how many axes each cytokine is in.

    `axes_rows`: iterable of dicts with 'axis_a','axis_b' (from cytokine_axes.csv).
    """
    deg: Dict[str, int] = {}
    for r in axes_rows:
        for k in (r["axis_a"], r["axis_b"]):
            deg[k] = deg.get(k, 0) + 1
    return deg


def graph_out_degree(audited_rows: Sequence[Dict]) -> Dict[str, int]:
    """Directed SOURCE out-degree from the audited benchmark.

    Counts, over rows with counts_in_benchmark True, the edges where a cytokine is
    the source: source = axis_a if expected_sign=+1 (a_to_b), axis_b if -1 (b_to_a).
    `audited_rows`: dicts with 'axis_a','axis_b','expected_sign','counts_in_benchmark'.
    """
    out: Dict[str, int] = {}
    for r in audited_rows:
        if str(r.get("counts_in_benchmark", "")).lower() != "true":
            continue
        try:
            sign = int(float(r.get("expected_sign", "")))
        except (ValueError, TypeError):
            continue
        src = r["axis_a"] if sign > 0 else (r["axis_b"] if sign < 0 else None)
        if src is not None:
            out[src] = out.get(src, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_against_degree(
    potency: Dict[str, float], degree: Dict[str, int]
) -> Dict:
    """Spearman(source_potency, graph degree) over cytokines present in both."""
    cyts = [c for c in potency if c in degree and np.isfinite(potency[c])]
    rho, n = spearman([potency[c] for c in cyts], [degree[c] for c in cyts])
    return {"rho": rho, "n": n, "cytokines": cyts,
            "metric_description": "Spearman(source_potency, graph degree)"}


def _perm_test_greater(a: np.ndarray, b: np.ndarray, n_perm: int = 10000,
                       seed: int = 0) -> Dict:
    """One-sided label-permutation test that mean(a) > mean(b) (numpy-only)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    obs = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b]); na = a.size
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if (pooled[:na].mean() - pooled[na:].mean()) >= obs:
            ge += 1
    return {"obs_diff": obs, "p": (ge + 1) / (n_perm + 1),
            "n_a": int(na), "n_b": int(b.size)}


def validate_deep_vs_shallow(
    potency: Dict[str, float],
    deep: Sequence[str] = tuple(DEEP_POOL),
    shallow: Sequence[str] = tuple(SHALLOW_POOL),
    n_perm: int = 10000,
    seed: int = 0,
) -> Dict:
    """One-sided test that source_potency(deep) > source_potency(shallow)."""
    da = np.array([potency[c] for c in deep if c in potency and np.isfinite(potency[c])])
    sa = np.array([potency[c] for c in shallow if c in potency and np.isfinite(potency[c])])
    if da.size < 2 or sa.size < 2:
        return {"obs_diff": float("nan"), "p": float("nan"),
                "n_a": int(da.size), "n_b": int(sa.size),
                "metric_description": "deep vs shallow (insufficient members)"}
    res = _perm_test_greater(da, sa, n_perm, seed)
    res["metric_description"] = ("one-sided permutation: mean source_potency(DEEP) > "
                                 "mean source_potency(SHALLOW)")
    return res
