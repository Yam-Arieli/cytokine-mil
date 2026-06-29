"""
Attention training-dynamics: cell-type-resolved cascade readout (CLAUDE.md §33).

The attention layer of the multiclass AB-MIL assigns a weight a_i to every cell
(token) when classifying which stimulus a pseudo-tube received. Because each
token's cell type is known post-hoc, the *trajectory of attention mass over cell
types during training* is a learnability-ordering readout: a frozen-encoder MIL
learns easy / direct / receptor-driven signal first and weak / secondary /
cascade signal later, so recruitment order is a within-training pseudo-time the
24h snapshot lacks.

This module is the analysis layer that turns the per-cell-type attention
trajectory (produced by ``scripts/extract_attention_trajectory.py``) into three
readouts that add to the Oesinghaus "Human Cytokine Dictionary" Fig 4
**prior-free**:

  1. primary vs secondary responder map        -> classify_primary_secondary
  2. relay-recruitment-lag direction statistic  -> relay_recruitment_lag
  3. intra-cell-type attention concentration     -> concentration_summary

Plus the P1 sanity check against known direct responders
(``attention_primary_vs_groundtruth``).

Pure numpy — consumes the dict structures from ``attention_trajectory.pkl`` and
the ``records`` list from ``train_mil``. No torch, no file IO, no model.

Every public function that produces a ranking / summary returns a
``metric_description`` key stating exactly what was computed (project convention).

Trajectory dict structures (from ``attention_trajectory.pkl``):
    trajectory:           {cytokine -> {cell_type -> np.array(n_epochs)}}            (donor-mean)
    trajectory_per_donor: {cytokine -> {cell_type -> {donor -> np.array(n_epochs)}}}
    concentration:        {cytokine -> {cell_type -> np.array(n_epochs)}}            (donor-mean Gini)
    epochs:               list[int]   (checkpoint epochs, ascending)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Known direct (primary) responders per cytokine, used for the P1 sanity check.
# Mirrors scripts/check_attention_cell_types.py:EXPECTED_DOMINANT. Cell-type
# spellings cover both Oesinghaus author labels and the demo/Leiden variants.
EXPECTED_DOMINANT: Dict[str, List[str]] = {
    "IL-12": ["NK", "NK CD56bright", "NK CD56hi", "NK_CD56hi", "NK CD56low"],
    "IFN-gamma": ["NK", "CD14 Mono", "CD14_Mono", "CD16 Mono", "CD16_Mono"],
    "IL-4": ["B Naive", "B Intermediate/Memory", "B_cell", "CD4_T", "CD4 Naive"],
    "IL-2": ["CD4_T", "CD8_T", "CD4 Naive", "CD8 Naive", "CD4 Memory T cell"],
    "TNF-alpha": ["CD14 Mono", "CD14_Mono", "CD16 Mono", "CD16_Mono"],
    "TNF": ["CD14 Mono", "CD14_Mono", "CD16 Mono", "CD16_Mono"],
}


# ---------------------------------------------------------------------------
# Small numpy helpers (no scipy dependency, matching analysis/direction_null.py)
# ---------------------------------------------------------------------------

def gini(x: Sequence[float]) -> float:
    """
    Gini coefficient of a non-negative vector (concentration of attention).

    0 -> perfectly uniform (all cells of a type attended equally);
    ->1 -> concentrated on a small responding subpopulation.

    Args:
        x: non-negative values (per-cell attention within one cell type).
    Returns:
        Gini coefficient in [0, 1]; 0.0 for an empty/all-zero vector.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return 0.0
    a = np.clip(a, 0.0, None)
    s = a.sum()
    if s <= 0:
        return 0.0
    a_sorted = np.sort(a)
    n = a.size
    idx = np.arange(1, n + 1)
    # Standard Gini from sorted values.
    return float((2.0 * (idx * a_sorted).sum()) / (n * s) - (n + 1.0) / n)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1..n), ties shared — minimal Spearman support, no scipy."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # Resolve ties to their average rank.
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def spearman(a: Sequence[float], b: Sequence[float]) -> Tuple[float, int]:
    """
    Spearman rank correlation (no scipy). Returns (rho, n_pairs).

    rho is NaN when fewer than 3 finite pairs or zero variance.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3:
        return float("nan"), n
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan"), n
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return rho, n


# ---------------------------------------------------------------------------
# 1. Recruitment timing (per cytokine, per cell type)
# ---------------------------------------------------------------------------

def celltype_recruitment(
    traj_cyt: Dict[str, np.ndarray],
    epochs: Sequence[int],
    rise_frac: float = 0.5,
) -> Dict[str, Dict]:
    """
    Recruitment timing of each cell type's attention for one cytokine.

    Attention magnitudes are tiny (~1/N) and differ across cell types, so the
    recruitment epoch is defined *relative to each cell type's own settled
    level*: the first checkpoint where the cell type reaches ``rise_frac`` of
    its final-epoch attention. Earlier recruitment -> primary/direct responder;
    later recruitment -> candidate secondary/cascade responder.

    Args:
        traj_cyt: {cell_type -> np.array(n_epochs)} attention trajectory for one
            cytokine (donor-mean or single-donor).
        epochs: checkpoint epochs (ascending), len == n_epochs.
        rise_frac: fraction of final value defining recruitment (default 0.5).
    Returns:
        {cell_type -> {'tau': int|None (epoch), 'tau_idx': int|None,
                       'final': float, 'peak': float, 'peak_idx': int}}.
        'tau' is None if the trajectory never reaches rise_frac*final
        (e.g. final <= 0).
    """
    epochs = list(epochs)
    out: Dict[str, Dict] = {}
    for ct, traj in traj_cyt.items():
        arr = np.asarray(traj, dtype=np.float64)
        if arr.size == 0:
            out[ct] = {"tau": None, "tau_idx": None, "final": 0.0,
                       "peak": 0.0, "peak_idx": 0}
            continue
        final = float(arr[-1])
        peak_idx = int(arr.argmax())
        if final <= 0:
            tau_idx = None
        else:
            crossed = np.where(arr >= rise_frac * final)[0]
            tau_idx = int(crossed[0]) if crossed.size > 0 else None
        out[ct] = {
            "tau": (epochs[tau_idx] if tau_idx is not None else None),
            "tau_idx": tau_idx,
            "final": final,
            "peak": float(arr[peak_idx]),
            "peak_idx": peak_idx,
        }
    return out


def recruitment_order(
    traj_cyt: Dict[str, np.ndarray],
    epochs: Sequence[int],
    rise_frac: float = 0.5,
) -> Dict:
    """
    Order cell types by attention-recruitment epoch for one cytokine.

    Metric: cell types sorted by tau (first epoch reaching rise_frac of own
    final attention); never-recruited cell types sorted last; ties broken by
    higher final attention.

    Returns:
        dict with 'order': list of (cell_type, tau, final) and
        'metric_description'.
    """
    rec = celltype_recruitment(traj_cyt, epochs, rise_frac)
    big = max(epochs) + 1 if len(epochs) else 1

    def _key(item):
        ct, info = item
        tau = info["tau"]
        return (big if tau is None else tau, -info["final"])

    order = [
        (ct, info["tau"], info["final"])
        for ct, info in sorted(rec.items(), key=_key)
    ]
    return {
        "order": order,
        "metric_description": (
            f"cell types ordered by attention-recruitment epoch tau (first checkpoint "
            f"reaching rise_frac={rise_frac} of own final attention); never-recruited last, "
            f"ties by higher final attention"
        ),
    }


def attention_primary(traj_cyt: Dict[str, np.ndarray]) -> Optional[str]:
    """Cell type with the highest final-epoch attention for one cytokine."""
    best, best_val = None, -np.inf
    for ct, traj in traj_cyt.items():
        arr = np.asarray(traj, dtype=np.float64)
        if arr.size == 0:
            continue
        v = float(arr[-1])
        if v > best_val:
            best, best_val = ct, v
    return best


# ---------------------------------------------------------------------------
# 2. Primary / secondary responder classification
# ---------------------------------------------------------------------------

def _second_rise(p_correct_traj: Sequence[float], after_frac: float,
                 min_delta: float = 0.02) -> bool:
    """
    True if p_correct rises by >= min_delta over the tail beginning at
    ``after_frac`` of training (a 'second rise' coinciding with late attention
    recruitment -> the model needed an additional, secondary signal).

    ``after_frac`` (not a raw index) is used because the attention trajectory and
    the p_correct trajectory may live on different epoch grids (checkpoints every
    N vs per-epoch logging).
    """
    p = np.asarray(p_correct_traj, dtype=np.float64)
    if p.size < 2:
        return False
    start = int(np.clip(round(after_frac * (p.size - 1)), 0, p.size - 1))
    if start >= p.size - 1:
        return False
    tail = p[start:]
    return float(tail.max() - tail[0]) >= min_delta


def classify_primary_secondary(
    trajectory: Dict[str, Dict[str, np.ndarray]],
    epochs: Sequence[int],
    p_correct_by_cyt: Optional[Dict[str, np.ndarray]] = None,
    rise_frac: float = 0.5,
    first_third: float = 1.0 / 3.0,
    last_third: float = 2.0 / 3.0,
    final_quantile: float = 0.5,
) -> Dict:
    """
    Label each (cytokine, cell_type) as 'primary', 'secondary' or 'minor'.

    primary  = recruited in the first third of training AND final attention in
               the top (1-final_quantile) of this cytokine's cell types.
    secondary= recruited in the last third AND (if p_correct supplied) a second
               rise in p_correct occurs at/after that recruitment index.
    minor    = otherwise.

    The data-driven analog of the paper's receptor-defined primary/secondary
    targets (Fig 4h): timing replaces the receptor-expression prior.

    Args:
        trajectory: {cytokine -> {cell_type -> np.array(n_epochs)}} (donor-mean).
        epochs: checkpoint epochs (ascending).
        p_correct_by_cyt: optional {cytokine -> np.array(n_epochs)} donor-mean
            p_correct trajectory (for the secondary second-rise gate).
        rise_frac, first_third, last_third, final_quantile: operationalizations.
    Returns:
        dict with 'labels': {cytokine -> {cell_type -> str}} and
        'metric_description'.
    """
    n_epochs = len(epochs)
    early_idx = max(0, int(np.floor(first_third * (n_epochs - 1))))
    late_idx = int(np.ceil(last_third * (n_epochs - 1)))

    labels: Dict[str, Dict[str, str]] = {}
    for cyt, traj_cyt in trajectory.items():
        rec = celltype_recruitment(traj_cyt, epochs, rise_frac)
        finals = {ct: info["final"] for ct, info in rec.items()}
        if finals:
            thresh = float(np.quantile(list(finals.values()), final_quantile))
        else:
            thresh = 0.0
        p_traj = (p_correct_by_cyt or {}).get(cyt)

        labels[cyt] = {}
        for ct, info in rec.items():
            tau_idx = info["tau_idx"]
            if tau_idx is None:
                labels[cyt][ct] = "minor"
                continue
            tau_frac = tau_idx / max(n_epochs - 1, 1)
            if tau_idx <= early_idx and info["final"] >= thresh:
                labels[cyt][ct] = "primary"
            elif tau_idx >= late_idx and (
                p_traj is None or _second_rise(p_traj, tau_frac)
            ):
                labels[cyt][ct] = "secondary"
            else:
                labels[cyt][ct] = "minor"
    return {
        "labels": labels,
        "metric_description": (
            f"primary = attention recruited in first third (idx<={early_idx}) with final "
            f"attention >= per-cytokine {final_quantile}-quantile; secondary = recruited in "
            f"last third (idx>={late_idx}) with a coincident p_correct second-rise (>=0.02); "
            f"else minor. Recruitment via rise_frac={rise_frac}."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Relay-recruitment-lag direction statistic
# ---------------------------------------------------------------------------

def _tau_for(traj: np.ndarray, epochs: Sequence[int], rise_frac: float) -> Optional[int]:
    arr = np.asarray(traj, dtype=np.float64)
    if arr.size == 0:
        return None
    final = float(arr[-1])
    if final <= 0:
        return None
    crossed = np.where(arr >= rise_frac * final)[0]
    return int(epochs[crossed[0]]) if crossed.size > 0 else None


def relay_recruitment_lag(
    trajectory: Dict[str, Dict[str, np.ndarray]],
    trajectory_per_donor: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    epochs: Sequence[int],
    A: str,
    B: str,
    T_B: Optional[str] = None,
    rise_frac: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict:
    """
    Temporal, IG-independent direction statistic for a coupled pair (A, B).

    Idea (Fig 4f/i cascade A -> B): the cell type that is B's direct responder
    (T_B) carries B's *secondary* program in A's tubes, so it should be
    recruited LATER in A's tubes than in B's own tubes. Per donor d (with both A
    and B and T_B present):

        lag_d = tau(A, T_B; d) - tau(B, T_B; d)

    lag > 0  => B-responder recruited later in A => A is upstream (A -> B).

    T_B defaults to B's data-driven attention-primary cell type (highest final
    attention in B's donor-mean trajectory) — no receptor / biology prior.

    Aggregation: mean lag across donors with a donor-bootstrap CI; sign
    consistency = fraction of donors with lag_d > 0. The direction CALL is the
    sign of the mean lag; it is reliable only when the bootstrap CI excludes 0.

    Args:
        trajectory: donor-mean {cytokine -> {cell_type -> array}} (for T_B pick).
        trajectory_per_donor: {cytokine -> {cell_type -> {donor -> array}}}.
        epochs: checkpoint epochs.
        A, B: cytokine names.
        T_B: relay cell type; default = attention_primary(B).
        rise_frac, n_boot, seed, ci: operationalizations.
    Returns:
        dict with: 'A','B','T_B','call' ('A->B'|'B->A'|'ambiguous'),
        'mean_lag','sign_consistency','ci_low','ci_high','n_donors',
        'per_donor_lag', 'metric_description'. Returns call='ambiguous' with
        NaN stats if <2 usable donors.
    """
    if T_B is None:
        T_B = attention_primary(trajectory.get(B, {}))

    md = (
        "relay-recruitment lag = tau(A, T_B) - tau(B, T_B) per donor, where tau is the "
        f"recruitment epoch (rise_frac={rise_frac}) of relay cell type T_B (= B's "
        "attention-primary cell type) and >0 => A upstream (A->B). Mean over donors with "
        f"donor-bootstrap {int(ci*100)}% CI; call reliable only if CI excludes 0."
    )
    empty = {
        "A": A, "B": B, "T_B": T_B, "call": "ambiguous",
        "mean_lag": float("nan"), "sign_consistency": float("nan"),
        "ci_low": float("nan"), "ci_high": float("nan"),
        "n_donors": 0, "per_donor_lag": {}, "metric_description": md,
    }
    if T_B is None:
        return empty

    a_pd = trajectory_per_donor.get(A, {}).get(T_B, {})
    b_pd = trajectory_per_donor.get(B, {}).get(T_B, {})
    donors = sorted(set(a_pd) & set(b_pd))

    per_donor: Dict[str, float] = {}
    for d in donors:
        tau_a = _tau_for(a_pd[d], epochs, rise_frac)
        tau_b = _tau_for(b_pd[d], epochs, rise_frac)
        if tau_a is not None and tau_b is not None:
            per_donor[d] = float(tau_a - tau_b)

    lags = np.array(list(per_donor.values()), dtype=np.float64)
    if lags.size < 2:
        empty["per_donor_lag"] = per_donor
        empty["n_donors"] = int(lags.size)
        return empty

    mean_lag = float(lags.mean())
    sign_consistency = float((lags > 0).mean())

    rng = np.random.default_rng(seed)
    boots = np.array([
        rng.choice(lags, size=lags.size, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))

    if lo > 0:
        call = "A->B"
    elif hi < 0:
        call = "B->A"
    else:
        call = "ambiguous"

    return {
        "A": A, "B": B, "T_B": T_B, "call": call,
        "mean_lag": mean_lag, "sign_consistency": sign_consistency,
        "ci_low": lo, "ci_high": hi, "n_donors": int(lags.size),
        "per_donor_lag": per_donor, "metric_description": md,
    }


# ---------------------------------------------------------------------------
# 4. Intra-cell-type attention concentration
# ---------------------------------------------------------------------------

def concentration_summary(
    concentration: Dict[str, Dict[str, np.ndarray]],
    epochs: Sequence[int],
) -> Dict:
    """
    Summarise within-cell-type attention concentration (Gini) over training.

    For each (cytokine, cell_type): final-epoch Gini and its linear trend
    (slope across checkpoints). High/rising Gini => a responding subpopulation;
    low/flat => whole-population engagement.

    Args:
        concentration: {cytokine -> {cell_type -> np.array(n_epochs)}} donor-mean
            within-type Gini trajectory (from the extractor).
        epochs: checkpoint epochs.
    Returns:
        dict with 'summary': {cytokine -> {cell_type -> {'final','slope'}}} and
        'metric_description'.
    """
    e = np.asarray(epochs, dtype=np.float64)
    summary: Dict[str, Dict[str, Dict]] = {}
    for cyt, by_ct in concentration.items():
        summary[cyt] = {}
        for ct, traj in by_ct.items():
            arr = np.asarray(traj, dtype=np.float64)
            if arr.size == 0:
                summary[cyt][ct] = {"final": 0.0, "slope": 0.0}
                continue
            if arr.size >= 2 and e.size == arr.size and e.std() > 0:
                slope = float(np.polyfit(e, arr, 1)[0])
            else:
                slope = 0.0
            summary[cyt][ct] = {"final": float(arr[-1]), "slope": slope}
    return {
        "summary": summary,
        "metric_description": (
            "within-cell-type attention concentration: final-epoch Gini of per-cell "
            "attention within each cell type, and its slope across checkpoints "
            "(rising => responding subpopulation; flat/low => whole-population)"
        ),
    }


# ---------------------------------------------------------------------------
# P1 sanity: attention-primary vs known direct responders
# ---------------------------------------------------------------------------

def attention_primary_vs_groundtruth(
    trajectory: Dict[str, Dict[str, np.ndarray]],
    epochs: Sequence[int],
    expected_dominant: Optional[Dict[str, List[str]]] = None,
    top_k: int = 3,
    rise_frac: float = 0.5,
    first_third: float = 1.0 / 3.0,
) -> Dict:
    """
    P1: does the attention-primary cell type match the known direct responder,
    and is it recruited early?

    For each cytokine in ``expected_dominant``: rank cell types by final
    attention, check whether any expected responder is in the top-``top_k``, and
    whether the top cell type is recruited within the first third of training.

    Args:
        trajectory: donor-mean {cytokine -> {cell_type -> array}}.
        epochs: checkpoint epochs.
        expected_dominant: {cytokine -> [expected cell types]} (default module
            EXPECTED_DOMINANT).
        top_k: match if an expected responder is in the top_k by final attention.
        rise_frac, first_third: early-recruitment operationalization.
    Returns:
        dict with 'per_cytokine', 'frac_match', 'frac_match_and_early',
        'n_evaluated', 'metric_description'.
    """
    expected_dominant = expected_dominant or EXPECTED_DOMINANT
    n_epochs = len(epochs)
    early_idx = max(0, int(np.floor(first_third * (n_epochs - 1))))

    per_cyt: Dict[str, Dict] = {}
    matches, early_matches, n_eval = 0, 0, 0
    for cyt, expected in expected_dominant.items():
        traj_cyt = trajectory.get(cyt)
        if not traj_cyt:
            continue
        n_eval += 1
        ranked = sorted(
            traj_cyt.items(),
            key=lambda kv: float(np.asarray(kv[1])[-1]) if np.asarray(kv[1]).size else -np.inf,
            reverse=True,
        )
        top = [ct for ct, _ in ranked[:top_k]]
        exp_set = set(expected)
        match = any(ct in exp_set for ct in top)
        # Early recruitment of the #1 cell type.
        early = False
        if ranked:
            rec = celltype_recruitment({ranked[0][0]: ranked[0][1]}, epochs, rise_frac)
            tidx = rec[ranked[0][0]]["tau_idx"]
            early = tidx is not None and tidx <= early_idx
        matches += int(match)
        early_matches += int(match and early)
        per_cyt[cyt] = {
            "top": top, "expected": expected, "match": match,
            "primary": ranked[0][0] if ranked else None, "primary_early": early,
        }
    return {
        "per_cytokine": per_cyt,
        "frac_match": (matches / n_eval) if n_eval else float("nan"),
        "frac_match_and_early": (early_matches / n_eval) if n_eval else float("nan"),
        "n_evaluated": n_eval,
        "metric_description": (
            f"fraction of evaluated cytokines whose attention-primary cell type (top-{top_k} "
            f"by final attention) includes a known direct responder, and additionally is "
            f"recruited in the first third (rise_frac={rise_frac})"
        ),
    }


# ---------------------------------------------------------------------------
# P3 primacy / subtlety correlation
# ---------------------------------------------------------------------------

def primacy_subtlety_correlation(
    primary_tau: Dict[str, float],
    directness: Dict[str, float],
) -> Dict:
    """
    P3: do textbook-direct cytokines recruit their primary cell type earlier?

    Spearman(primary recruitment epoch, response-directness proxy) over
    cytokines present in both dicts. Prediction: negative (more direct => earlier
    recruitment => smaller tau).

    Args:
        primary_tau: {cytokine -> recruitment epoch of its primary cell type}.
        directness: {cytokine -> directness proxy} (e.g. learnability AUC or the
            paper's response magnitude; larger = more direct).
    Returns:
        dict with 'rho','n','metric_description'.
    """
    cyts = sorted(set(primary_tau) & set(directness))
    a = [primary_tau[c] for c in cyts]
    b = [directness[c] for c in cyts]
    rho, n = spearman(a, b)
    return {
        "rho": rho, "n": n, "cytokines": cyts,
        "metric_description": (
            "Spearman(primary-cell-type recruitment epoch tau, response-directness proxy) "
            "across cytokines; predicted negative (direct -> earlier recruitment)"
        ),
    }


# ---------------------------------------------------------------------------
# Adapter: donor-mean p_correct trajectory from train_mil records
# ---------------------------------------------------------------------------

def p_correct_by_cytokine(records: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Donor-mean p_correct trajectory per cytokine from train_mil ``records``.

    Aggregation matches the project convention: median across tubes per donor,
    then mean across donors.

    Args:
        records: list of per-tube dicts with 'cytokine','donor',
            'p_correct_trajectory'.
    Returns:
        {cytokine -> np.array(n_logged_epochs)}.
    """
    raw: Dict[str, Dict[str, List[List[float]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        traj = r.get("p_correct_trajectory")
        if traj is not None:
            raw[r["cytokine"]][r["donor"]].append(list(traj))
    out: Dict[str, np.ndarray] = {}
    for cyt, donors in raw.items():
        donor_med = [np.median(np.array(ts), axis=0) for ts in donors.values()]
        out[cyt] = np.mean(np.stack(donor_med), axis=0)
    return out
