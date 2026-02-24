"""
Validation helpers: seed stability, cascade recovery, multiple-testing correction.

All statistical claims about relative learnability should use these functions
after aggregating to donor level (effective N = 12).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from cytokine_mil.analysis.dynamics import (
    aggregate_to_donor_level,
    rank_cytokines_by_learnability,
)


# ---------------------------------------------------------------------------
# Seed stability
# ---------------------------------------------------------------------------

def check_seed_stability(
    dynamics_list: List[Dict],
    exclude: Optional[List[str]] = None,
) -> Dict:
    """
    Assess whether cytokine learnability ordering is stable across seeds.

    Computes the Spearman rank correlation between every pair of seed runs.
    High correlation (> 0.7) across all pairs suggests a robust dynamics signal.

    Args:
        dynamics_list: List of dynamics dicts, one per seed (from train_mil).
        exclude: Cytokines to exclude (e.g., ['PBS']).
    Returns:
        dict with:
            'rankings': list of ranked cytokine lists (one per seed),
            'spearman_matrix': np.array (n_seeds x n_seeds) of rho values,
            'mean_rho': float mean of off-diagonal rho values,
            'stable': bool (True if mean_rho > 0.7).
    """
    rankings = []
    for dynamics in dynamics_list:
        donor_traj = aggregate_to_donor_level(dynamics["records"])
        ranked = rank_cytokines_by_learnability(donor_traj, exclude=exclude)
        rankings.append([cyt for cyt, _ in ranked])

    n = len(rankings)
    spearman_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho = _spearman_between_rankings(rankings[i], rankings[j])
            spearman_matrix[i, j] = rho
            spearman_matrix[j, i] = rho

    off_diag = spearman_matrix[np.triu_indices(n, k=1)]
    mean_rho = float(np.mean(off_diag)) if len(off_diag) > 0 else 1.0

    return {
        "rankings": rankings,
        "spearman_matrix": spearman_matrix,
        "mean_rho": mean_rho,
        "stable": mean_rho > 0.7,
    }


def _spearman_between_rankings(
    ranking_a: List[str], ranking_b: List[str]
) -> float:
    """Spearman rho between two orderings of the same cytokines."""
    all_cyts = list(dict.fromkeys(ranking_a + ranking_b))
    rank_a = {c: i for i, c in enumerate(ranking_a)}
    rank_b = {c: i for i, c in enumerate(ranking_b)}
    common = [c for c in all_cyts if c in rank_a and c in rank_b]
    if len(common) < 2:
        return 0.0
    a = np.array([rank_a[c] for c in common], dtype=float)
    b = np.array([rank_b[c] for c in common], dtype=float)
    rho, _ = stats.spearmanr(a, b)
    return float(rho)


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------

def apply_fdr_correction(
    pvalues: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Required for any statistical claim comparing learnability across the
    91 cytokine classes due to the large number of pairwise comparisons.

    Args:
        pvalues: 1-D array of raw p-values.
        alpha: FDR threshold (default 0.05).
    Returns:
        (rejected, corrected_pvalues):
            rejected: boolean array, True where null is rejected at FDR alpha.
            corrected_pvalues: BH-adjusted p-values.
    """
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)

    corrected = pvalues * n / ranks
    # Enforce monotonicity: corrected[i] <= corrected[i+1]
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)

    rejected = corrected <= alpha
    return rejected, corrected


# ---------------------------------------------------------------------------
# Known-group validation
# ---------------------------------------------------------------------------

def check_functional_groupings(
    donor_trajectories: Dict[str, Dict[str, np.ndarray]],
    known_groups: Dict[str, List[str]],
) -> Dict[str, Dict]:
    """
    Verify that cytokines with known similar biology cluster in learnability space.

    For each group, computes the within-group vs. between-group AUC similarity.
    Example known groups: {'IL-2_family': ['IL-2', 'IL-15'], 'type_I_IFN': ['IFN-alpha', 'IFN-beta']}.

    Args:
        donor_trajectories: Output of aggregate_to_donor_level().
        known_groups: dict mapping group_name -> list of cytokine names.
    Returns:
        dict with per-group stats:
            {'within_auc_corr': float, 'between_auc_corr': float, 'passes': bool}
    """
    # Compute mean AUC per cytokine (averaged across donors)
    auc_per_cyt = _compute_mean_aucs(donor_trajectories)
    results = {}

    for group_name, members in known_groups.items():
        present = [c for c in members if c in auc_per_cyt]
        if len(present) < 2:
            results[group_name] = {"error": f"fewer than 2 members found: {present}"}
            continue

        within_aucs = [auc_per_cyt[c] for c in present]
        all_others = [v for k, v in auc_per_cyt.items() if k not in set(present)]

        within_std = float(np.std(within_aucs))
        between_std = float(np.std(all_others)) if all_others else 0.0

        results[group_name] = {
            "members_found": present,
            "within_auc_std": within_std,
            "between_auc_std": between_std,
            # Passes if within-group spread is smaller than global spread
            "passes": within_std < between_std,
        }

    return results


def _compute_mean_aucs(
    donor_trajectories: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """Return mean (across donors) AUC of the p_correct trajectory per cytokine."""
    return {
        cyt: float(np.mean([np.trapz(traj) for traj in donors.values()]))
        for cyt, donors in donor_trajectories.items()
    }


# ---------------------------------------------------------------------------
# Snapshot confound analysis
# ---------------------------------------------------------------------------

def check_snapshot_confound(
    donor_trajectories: Dict[str, Dict[str, np.ndarray]],
    expected_maturity: Dict[str, str],
) -> Dict:
    """
    Test whether 'hard to learn' cytokines correlate with 'slow 24h kinetics'.

    This is a post-hoc analysis only. Pre-register directional predictions
    before running this function.

    Args:
        donor_trajectories: Output of aggregate_to_donor_level().
        expected_maturity: dict mapping cytokine_name ->
            'fast' | 'slow' | 'unknown' (from literature).
    Returns:
        dict with point-biserial correlation and interpretation.
    """
    auc_per_cyt = _compute_mean_aucs(donor_trajectories)
    fast_aucs, slow_aucs = [], []
    for cyt, maturity in expected_maturity.items():
        if cyt not in auc_per_cyt:
            continue
        if maturity == "fast":
            fast_aucs.append(auc_per_cyt[cyt])
        elif maturity == "slow":
            slow_aucs.append(auc_per_cyt[cyt])

    if not fast_aucs or not slow_aucs:
        return {"error": "Insufficient data for confound test."}

    # Mann-Whitney U: do fast-kinetics cytokines have higher AUC?
    stat, pval = stats.mannwhitneyu(fast_aucs, slow_aucs, alternative="greater")
    return {
        "fast_mean_auc": float(np.mean(fast_aucs)),
        "slow_mean_auc": float(np.mean(slow_aucs)),
        "mannwhitney_stat": float(stat),
        "pvalue": float(pval),
        "confound_likely": pval < 0.05,
        "interpretation": (
            "Confound likely: hard-to-learn cytokines correlate with slow kinetics."
            if pval < 0.05
            else "No strong confound: learnability not explained by snapshot timing."
        ),
    }
