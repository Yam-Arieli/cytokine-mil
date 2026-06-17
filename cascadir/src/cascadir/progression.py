"""Progression-direction helpers — recover a *total order* of conditions from a
single cross-sectional snapshot, with a **nested-donor** donor-bootstrap.

CONTEXT (CLAUDE.md §30). The cytokine `cross_asym` direction method assumes each
donor saw every condition, so per-donor-paired stats (`signature_coupling(
donor_level=True)`, Path A `discover_axes`) are well-defined. On a disease atlas
where each patient has exactly ONE state (severity grade), donors are **nested in
conditions** — there is no within-donor pairing and no per-donor control centroid.
Two consequences this module addresses:

  1. Pooled `cross_asym(a,b) = M[a,b] - M[b,a]`, with
     `M[a,b] = median_T( mean_{a-cells@T}(score on S_b) - mean_{control@T}(score on S_b) )`,
     is STILL well-defined (a population statistic; control = the healthy donors).
     The point estimate comes from cascadir's `direction_table()`.

  2. Donor-level rigour comes from a **donor-bootstrap**: resample donors WITH
     replacement *within each condition and within the control group* (donor = the
     unit of independence, per CLAUDE.md §16/§27.6), recompute the pooled
     `cross_asym`, and report a CI. This is the nested-donor-correct replacement for
     within-donor pairing.

To make the bootstrap cheap (B≈1000 over ~10^5 cells), it operates on a small
**score cache** — one row per (donor, cell_type) with the donor's condition, the
cell count, and that group's mean score on each condition's signature S_j. The cache
is produced once by the run driver after `CascadeDirection.fit()`; everything here is
pure (donor, cell_type)-level aggregation.

Also provides `recover_order` (Borda count over pairwise `cross_asym` signs → a total
order) and `kendall_tau` (vs the true clinical order) — the headline
"did a single snapshot recover the progression order?" metric.

Allowed imports: numpy + pandas (matches the cascadir analysis layer).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Column-name contract for the score cache (one row per donor × cell_type).
DONOR_COL = "donor"
CONDITION_COL = "condition"
CELLTYPE_COL = "cell_type"
NCELLS_COL = "n_cells"
SIG_PREFIX = "s__"  # f"{SIG_PREFIX}{condition}" = mean score of this group on S_condition


# ---------------------------------------------------------------------------
# Internal: per-condition packed arrays for fast bootstrap aggregation
# ---------------------------------------------------------------------------

class _ConditionPack:
    """Packed (row -> donor_idx, ct_idx, n, score-vector) arrays for one condition.

    `aggregate(donor_counts)` returns, per cell type, the n-weighted mean score
    vector over the donors selected (with multiplicity) by `donor_counts`.
    """

    def __init__(self, sub: pd.DataFrame, sig_cols: List[str], cell_types: List[str]):
        self.donors: List[str] = sorted(sub[DONOR_COL].unique().tolist())
        self._donor_idx = {d: i for i, d in enumerate(self.donors)}
        self._ct_idx = {t: i for i, t in enumerate(cell_types)}
        self.n_ct = len(cell_types)
        self.n_sig = len(sig_cols)
        self.row_donor = sub[DONOR_COL].map(self._donor_idx).to_numpy(dtype=np.int64)
        self.row_ct = sub[CELLTYPE_COL].map(self._ct_idx).to_numpy(dtype=np.int64)
        self.row_n = sub[NCELLS_COL].to_numpy(dtype=np.float64)
        self.row_s = sub[sig_cols].to_numpy(dtype=np.float64)  # (rows, n_sig)

    def aggregate(self, donor_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """n-weighted mean score per (cell_type, signature) for a donor multiset.

        Args:
            donor_counts: (n_donors,) int — how many times each donor is sampled.
        Returns:
            mean: (n_ct, n_sig) weighted mean (NaN where a cell type has no weight).
            present: (n_ct,) bool — cell types with any sampled cells.
        """
        w = donor_counts[self.row_donor] * self.row_n  # (rows,)
        num = np.zeros((self.n_ct, self.n_sig), dtype=np.float64)
        den = np.zeros(self.n_ct, dtype=np.float64)
        np.add.at(num, self.row_ct, w[:, None] * self.row_s)
        np.add.at(den, self.row_ct, w)
        present = den > 0
        mean = np.full((self.n_ct, self.n_sig), np.nan, dtype=np.float64)
        mean[present] = num[present] / den[present, None]
        return mean, present


def _pack(score_cache: pd.DataFrame, conditions: Sequence[str], control_label: str):
    """Validate the cache and build packed structures for all conditions + control."""
    sig_cols = [f"{SIG_PREFIX}{c}" for c in conditions]
    missing = [c for c in sig_cols if c not in score_cache.columns]
    if missing:
        raise ValueError(f"score_cache missing signature columns {missing}; "
                         f"expected one '{SIG_PREFIX}<condition>' per condition.")
    cell_types = sorted(score_cache[CELLTYPE_COL].unique().tolist())
    sig_index = {c: j for j, c in enumerate(conditions)}
    packs: Dict[str, _ConditionPack] = {}
    for c in list(conditions) + [control_label]:
        sub = score_cache[score_cache[CONDITION_COL] == c]
        if len(sub) == 0:
            raise ValueError(f"score_cache has no rows for condition '{c}'.")
        packs[c] = _ConditionPack(sub, sig_cols, cell_types)
    return packs, sig_index, cell_types


def _cross_asym_from_means(
    mean_a: np.ndarray, pres_a: np.ndarray,
    mean_b: np.ndarray, pres_b: np.ndarray,
    mean_ctrl: np.ndarray, pres_ctrl: np.ndarray,
    ja: int, jb: int,
) -> float:
    """cross_asym(a,b) = median_T(mean_a@Sb - ctrl@Sb) - median_T(mean_b@Sa - ctrl@Sa).

    Median taken over cell types where the relevant groups are all present.
    Returns NaN if no cell type qualifies for either term.
    """
    ok_ab = pres_a & pres_ctrl
    ok_ba = pres_b & pres_ctrl
    m_ab = np.nanmedian((mean_a[:, jb] - mean_ctrl[:, jb])[ok_ab]) if ok_ab.any() else np.nan
    m_ba = np.nanmedian((mean_b[:, ja] - mean_ctrl[:, ja])[ok_ba]) if ok_ba.any() else np.nan
    return float(m_ab - m_ba)


# ---------------------------------------------------------------------------
# Public: pooled cross_asym + donor-bootstrap
# ---------------------------------------------------------------------------

def pooled_cross_asym(
    score_cache: pd.DataFrame,
    conditions: Sequence[str],
    control_label: str,
    donor_counts: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[Tuple[str, str], float]:
    """Pooled `cross_asym` for every unordered pair (a<b alphabetically).

    Args:
        score_cache: rows = (donor, condition, cell_type) with n_cells + one
            `s__<condition>` column per condition (incl. control rows scored on every S).
        conditions: the non-control conditions (signature owners).
        control_label: the control condition value.
        donor_counts: optional {condition_or_control: (n_donors,) int} multiset for a
            bootstrap replicate; default = each donor once (the point estimate).
    Returns:
        {(a, b): cross_asym} with a < b lexicographically; positive ⇒ a upstream.
    """
    packs, sig_index, _ = _pack(score_cache, conditions, control_label)
    if donor_counts is None:
        donor_counts = {c: np.ones(len(p.donors), dtype=np.int64) for c, p in packs.items()}
    means = {c: packs[c].aggregate(donor_counts[c]) for c in packs}
    out: Dict[Tuple[str, str], float] = {}
    for a, b in combinations(sorted(conditions), 2):
        ma, pa = means[a]
        mb, pb = means[b]
        mc, pc = means[control_label]
        out[(a, b)] = _cross_asym_from_means(ma, pa, mb, pb, mc, pc,
                                             sig_index[a], sig_index[b])
    return out


def bootstrap_cross_asym(
    score_cache: pd.DataFrame,
    conditions: Sequence[str],
    control_label: str,
    oracle: Optional[Sequence[Tuple[str, str]]] = None,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, object]:
    """Nested-donor donor-bootstrap of pooled `cross_asym`.

    Resamples donors WITH replacement within each condition and within the control
    group (donor = unit of independence), recomputes pooled `cross_asym`, and reports
    per-pair CIs plus — if `oracle` is given — bootstrap distributions of the overall
    sign-accuracy and of Kendall τ between the recovered and true orders.

    Args:
        oracle: list of (upstream, downstream) ground-truth ordered pairs.
        n_boot, seed, ci: bootstrap settings.
    Returns dict with:
        per_pair: DataFrame [condition_a, condition_b, cross_asym, ci_lo, ci_hi,
                  frac_sign_pos]  (ci on the bootstrap distribution)
        accuracy: {point, ci_lo, ci_hi} of sign-accuracy vs oracle (None if no oracle)
        kendall_tau: {point, ci_lo, ci_hi} vs the oracle order (None if no oracle)
        boot_cross: {pair: np.ndarray(n_boot)} raw draws
    """
    packs, sig_index, _ = _pack(score_cache, conditions, control_label)
    rng = np.random.default_rng(seed)
    pairs = list(combinations(sorted(conditions), 2))
    boot_cross = {p: np.empty(n_boot, dtype=np.float64) for p in pairs}
    boot_acc = np.full(n_boot, np.nan)
    boot_tau = np.full(n_boot, np.nan)

    oracle_order = _order_from_oracle(oracle, conditions) if oracle else None

    for k in range(n_boot):
        donor_counts = {}
        for c, p in packs.items():
            nd = len(p.donors)
            idx = rng.integers(0, nd, size=nd)  # resample donors with replacement
            counts = np.zeros(nd, dtype=np.int64)
            np.add.at(counts, idx, 1)
            donor_counts[c] = counts
        ca = pooled_cross_asym(score_cache, conditions, control_label, donor_counts)
        for p in pairs:
            boot_cross[p][k] = ca[p]
        if oracle:
            boot_acc[k] = accuracy_vs_oracle(ca, oracle)
            rec = recover_order(ca, conditions)
            boot_tau[k] = kendall_tau(rec, oracle_order)

    point = pooled_cross_asym(score_cache, conditions, control_label)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    rows = []
    for (a, b) in pairs:
        draws = boot_cross[(a, b)]
        finite = draws[np.isfinite(draws)]
        rows.append({
            "condition_a": a, "condition_b": b,
            "cross_asym": point[(a, b)],
            "ci_lo": float(np.quantile(finite, lo_q)) if finite.size else np.nan,
            "ci_hi": float(np.quantile(finite, hi_q)) if finite.size else np.nan,
            "frac_sign_pos": float(np.mean(finite > 0)) if finite.size else np.nan,
        })
    per_pair = pd.DataFrame(rows)

    def _summ(arr):
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return None
        return {"point": float(np.mean(a)),
                "ci_lo": float(np.quantile(a, lo_q)),
                "ci_hi": float(np.quantile(a, hi_q))}

    acc_pt = accuracy_vs_oracle(point, oracle) if oracle else None
    tau_pt = (kendall_tau(recover_order(point, conditions), oracle_order)
              if oracle else None)
    accuracy = None
    kendall = None
    if oracle:
        accuracy = _summ(boot_acc) or {}
        accuracy["point"] = acc_pt
        kendall = _summ(boot_tau) or {}
        kendall["point"] = tau_pt

    return {"per_pair": per_pair, "accuracy": accuracy,
            "kendall_tau": kendall, "boot_cross": boot_cross}


# ---------------------------------------------------------------------------
# Public: order recovery + Kendall tau
# ---------------------------------------------------------------------------

def recover_order(
    cross_asym_by_pair: Dict[Tuple[str, str], float],
    conditions: Sequence[str],
) -> List[str]:
    """Total order (most-upstream first) via Borda count over pairwise cross_asym signs.

    For pair (a,b) with a<b: cross_asym>0 ⇒ a beats b (a upstream); <0 ⇒ b beats a.
    Borda score = number of conditions each one is judged upstream of. Ties broken by
    summed signed cross_asym magnitude, then alphabetically (deterministic).
    """
    conds = sorted(conditions)
    wins = {c: 0.0 for c in conds}
    mag = {c: 0.0 for c in conds}
    for a, b in combinations(conds, 2):
        v = cross_asym_by_pair.get((a, b))
        if v is None or not np.isfinite(v):
            continue
        if v > 0:
            wins[a] += 1
        elif v < 0:
            wins[b] += 1
        mag[a] += v
        mag[b] -= v
    return sorted(conds, key=lambda c: (-wins[c], -mag[c], c))


def _order_from_oracle(
    oracle: Sequence[Tuple[str, str]], conditions: Sequence[str]
) -> List[str]:
    """Total order (most-upstream first) implied by oracle (upstream, downstream) pairs."""
    conds = sorted(conditions)
    wins = {c: 0 for c in conds}
    for up, down in oracle:
        if up in wins:
            wins[up] += 1
    return sorted(conds, key=lambda c: (-wins[c], c))


def accuracy_vs_oracle(
    cross_asym_by_pair: Dict[Tuple[str, str], float],
    oracle: Sequence[Tuple[str, str]],
) -> float:
    """Fraction of oracle pairs whose cross_asym sign names the correct upstream."""
    correct = 0
    total = 0
    for up, down in oracle:
        a, b = sorted([up, down])
        v = cross_asym_by_pair.get((a, b))
        if v is None or not np.isfinite(v) or v == 0:
            total += 1
            continue
        called_a_upstream = v > 0
        expected_a_upstream = (up == a)
        correct += int(called_a_upstream == expected_a_upstream)
        total += 1
    return correct / total if total else float("nan")


def kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    """Kendall's τ between two total orders over the same items (−1..+1)."""
    items = list(order_a)
    rank_a = {c: i for i, c in enumerate(order_a)}
    rank_b = {c: i for i, c in enumerate(order_b)}
    common = [c for c in items if c in rank_b]
    n = len(common)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = common[i], common[j]
            s = np.sign(rank_a[ci] - rank_a[cj]) * np.sign(rank_b[ci] - rank_b[cj])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    denom = n * (n - 1) / 2
    return (conc - disc) / denom if denom else float("nan")


__all__ = [
    "pooled_cross_asym",
    "bootstrap_cross_asym",
    "recover_order",
    "accuracy_vs_oracle",
    "kendall_tau",
    "DONOR_COL", "CONDITION_COL", "CELLTYPE_COL", "NCELLS_COL", "SIG_PREFIX",
]
