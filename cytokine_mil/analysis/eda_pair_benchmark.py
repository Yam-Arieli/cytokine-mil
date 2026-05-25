"""
Pair-level EDA benchmark for cascade discovery.

Computes a battery of statistics for every ordered cytokine pair (A, B)
stratified by cell type, on labeled cascade vs no-cascade pairs from
Sheu §21. No model required — operates directly on normalized gene
expression data.

Goal: rather than designing a method based on assumptions about where
cascade signal lives, compute many candidate statistics on a labeled
benchmark and let the data reveal which statistics actually discriminate
known cascades from known non-cascades.

Cytokine names follow the Sheu pseudotube manifest convention:
  "PIC"   ↔ polyIC
  "P3CSK" ↔ Pam3CSK4
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from scipy import stats as scstats


# §21 labeled pairs (canonical Sheu BMDM @ 3hr).
# Names match the Sheu pseudotube manifest.
POSITIVE_PAIRS: List[Tuple[str, str]] = [
    ("LPS",   "TNF"),    # MUST: TLR4 → NF-κB → autocrine TNF
    ("PIC",   "IFNb"),   # MUST: TLR3/TRIF → IRF3 → type-I IFN
    ("LPS",   "IFNb"),   # SHOULD: LPS engages TRIF arm too
    ("P3CSK", "CpG"),    # SHOULD: both MyD88-only
    ("LPSlo", "P3CSK"),  # SHOULD: both MyD88-biased
]

NEGATIVE_PAIRS: List[Tuple[str, str]] = [
    ("P3CSK", "IFNb"),   # MUST-NOT: TLR2 has no TRIF arm
    ("CpG",   "IFNb"),   # MUST-NOT: TLR9 IFN restricted to pDC
    ("TNF",   "IFNb"),   # MUST-NOT: no cross-induction in macrophages
]

NAME_ALIAS: Dict[str, str] = {"PIC": "polyIC", "P3CSK": "Pam3CSK4"}


def labeled_pair_status(A: str, B: str) -> Optional[str]:
    """Return 'positive', 'negative', or None for the unordered pair {A, B}."""
    pair = frozenset({A, B})
    for p in POSITIVE_PAIRS:
        if frozenset(p) == pair:
            return "positive"
    for p in NEGATIVE_PAIRS:
        if frozenset(p) == pair:
            return "negative"
    return None


def unordered_key(A: str, B: str) -> str:
    """Stable string key for an unordered pair (lexicographic)."""
    a, b = sorted([A, B])
    return f"{a}—{b}"


# --------------------------------------------------------------------- IO

def load_phase1_cells(
    manifest_path: str,
    gene_names: Optional[List[str]] = None,
    time_filter: Optional[str] = None,
    donors: Optional[List[str]] = None,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], List[str]]:
    """
    Load Sheu pseudotubes and pool cells by (cytokine, cell_type).

    Cell-type labels come from `adata.obs['cell_type']`. The Sheu manifest
    does not carry a 'time_point' field (time is in h5ad.obs); when
    `time_filter` is set, cells are filtered on `adata.obs['time_point']`
    *for stimulated tubes only* — PBS pseudotubes are always kept in full
    (the Sheu adapter pools 0hr Unstim into the PBS class).

    Args:
        manifest_path: Path to manifest.json. Entries must have 'path' and
            'cytokine'; 'donor' is honoured if `donors` is set.
        gene_names: Optional gene subset (in order). If None, all genes used.
        time_filter: If set, drop stim cells whose obs.time_point != this value.
            Set to None for "use whatever the manifest already contains" —
            phase-1 manifest is already filtered to 0hr + 3hr.
        donors: Optional list of donor (= Sheu pseudo-donor) names to keep.

    Returns:
        cells_by_pair: {(cytokine, cell_type) -> (N_cells, G) float32 array}
        gene_names:    list of gene names in column order
    """
    with open(manifest_path) as f:
        entries = json.load(f)

    pooled: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
    resolved_genes: Optional[List[str]] = None
    donor_set = set(donors) if donors else None

    for entry in entries:
        if donor_set is not None and entry.get("donor") not in donor_set:
            continue
        cytokine = entry["cytokine"]
        adata = sc.read_h5ad(entry["path"])
        if gene_names is not None:
            adata = adata[:, gene_names]

        # Cell-level time filter (only for stim tubes; PBS = 0hr by adapter convention)
        if (time_filter is not None
                and cytokine != "PBS"
                and "time_point" in adata.obs.columns):
            tp_mask = adata.obs["time_point"].astype(str).values == time_filter
            if not tp_mask.any():
                continue
            adata = adata[tp_mask].copy()

        X = adata.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if resolved_genes is None:
            resolved_genes = list(adata.var_names)

        cell_types = (
            adata.obs["cell_type"].astype(str).values
            if "cell_type" in adata.obs.columns
            else np.full(len(X), "unknown")
        )
        for ct in np.unique(cell_types):
            mask = cell_types == ct
            if mask.any():
                pooled[(cytokine, ct)].append(X[mask])

    cells_by_pair = {k: np.concatenate(v, axis=0) for k, v in pooled.items()}
    return cells_by_pair, resolved_genes or []


# --------------------------------------------------------------------- helpers

def _safe_norm(v: np.ndarray, eps: float = 1e-9) -> float:
    return float(max(np.linalg.norm(v), eps))


def _log2fc(stim: np.ndarray, baseline: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.log2((stim.mean(axis=0) + eps) / (baseline.mean(axis=0) + eps))


def _sarle_bimodality(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient. Range ~[0, 1]; >5/9 ≈ bimodal."""
    n = len(x)
    if n < 4:
        return float("nan")
    sk = scstats.skew(x)
    ku = scstats.kurtosis(x)  # excess kurtosis
    denom = ku + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    if denom == 0:
        return float("nan")
    return float((sk ** 2 + 1.0) / denom)


def _kde_kl(x: np.ndarray, y: np.ndarray, n_grid: int = 200, eps: float = 1e-10) -> float:
    """KL(P_x || P_y) via 1-D KDE on a shared grid."""
    if len(x) < 5 or len(y) < 5:
        return float("nan")
    try:
        kde_x = scstats.gaussian_kde(x)
        kde_y = scstats.gaussian_kde(y)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if hi <= lo:
        return float("nan")
    grid = np.linspace(lo, hi, n_grid)
    px = kde_x(grid) + eps
    py = kde_y(grid) + eps
    px /= px.sum()
    py /= py.sum()
    return float(np.sum(px * np.log(px / py)))


def _signature_score(
    cells: np.ndarray,
    sig_idx: np.ndarray,
    ctrl_idx: np.ndarray,
) -> np.ndarray:
    """Mean(signature genes) - Mean(control genes), per cell."""
    sig = cells[:, sig_idx].mean(axis=1)
    ctrl = cells[:, ctrl_idx].mean(axis=1)
    return sig - ctrl


# --------------------------------------------------------------------- battery

def compute_pair_statistics(
    cells_A: np.ndarray,
    cells_B: np.ndarray,
    cells_PBS: np.ndarray,
    n_sig: int = 20,
    n_top_de: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Compute the full statistic battery for one (A, B, cell_type) triple.

    All inputs are (N_cells, G) arrays in the same gene space.

    Statistics returned (and what they conceptually measure):

    Symmetric / similarity:
        centroid_distance       — L2 distance between cytokine centroids
        log2fc_spearman         — Spearman corr of (A vs PBS) and (B vs PBS) per-gene log2FC
        de_jaccard              — Jaccard of top-N |log2FC| genes
        var_ratio_AB            — Var(A on û_{A→B}) / Var(B on û_{A→B})

    Asymmetric (cascade-relevant by construction):
        frac_A_closer_to_B      — fraction of A-cells whose nearest centroid is µ_B not µ_A
        frac_B_closer_to_A      — symmetric counterpart
        reciprocal_asymmetry    — diff of the two
        mean_sigB_in_A          — mean B-signature score on A-tube cells
        mean_sigA_in_B          — symmetric counterpart
        sigB_in_A_norm          — mean_sigB_in_A / mean_sigB_in_B (cascade share vs direct)
        sigA_in_B_norm          — symmetric counterpart
        signature_asymmetry     — sigB_in_A_norm − sigA_in_B_norm
        frac_A_with_high_sigB   — fraction of A-cells with B-signature > 0.5σ above B-tube baseline
        frac_B_with_high_sigA   — symmetric counterpart
        tail_asymmetry          — diff of the two
        kl_A_to_B_along_AB      — KL(P_A‖P_B) projected onto û_{A→B}
        kl_B_to_A_along_AB      — symmetric counterpart
        kl_asymmetry            — diff of the two

    Heterogeneity / mixture (within-tube shape, NOT means):
        var_A_along_AB          — variance of A-cells projected on û_{A→B}
        var_B_along_AB          — variance of B-cells projected on û_{A→B}
        bimodality_A_along_AB   — Sarle's bimodality coefficient of A on û_{A→B}
        bimodality_B_along_AB   — symmetric counterpart
    """
    if rng is None:
        rng = np.random.default_rng(0)

    G = cells_A.shape[1]
    out: Dict[str, float] = {}

    mu_A = cells_A.mean(axis=0)
    mu_B = cells_B.mean(axis=0)

    # ----- symmetric similarity
    out["centroid_distance"] = float(np.linalg.norm(mu_A - mu_B))

    log2fc_A = _log2fc(cells_A, cells_PBS)
    log2fc_B = _log2fc(cells_B, cells_PBS)
    sp = scstats.spearmanr(log2fc_A, log2fc_B)
    rho = float(sp.correlation) if sp.correlation == sp.correlation else float("nan")
    out["log2fc_spearman"] = rho

    top_A_de = set(np.argsort(np.abs(log2fc_A))[-n_top_de:])
    top_B_de = set(np.argsort(np.abs(log2fc_B))[-n_top_de:])
    out["de_jaccard"] = len(top_A_de & top_B_de) / max(len(top_A_de | top_B_de), 1)

    # ----- reciprocal-closer fractions (asymmetric, model-free)
    dA_self = np.linalg.norm(cells_A - mu_A, axis=1)
    dA_to_B = np.linalg.norm(cells_A - mu_B, axis=1)
    out["frac_A_closer_to_B"] = float((dA_to_B < dA_self).mean())

    dB_self = np.linalg.norm(cells_B - mu_B, axis=1)
    dB_to_A = np.linalg.norm(cells_B - mu_A, axis=1)
    out["frac_B_closer_to_A"] = float((dB_to_A < dB_self).mean())
    out["reciprocal_asymmetry"] = out["frac_A_closer_to_B"] - out["frac_B_closer_to_A"]

    # ----- signature-score asymmetry
    sig_A_up = np.argsort(log2fc_A)[-n_sig:]      # top up-regulated in A vs PBS
    sig_B_up = np.argsort(log2fc_B)[-n_sig:]
    used = set(sig_A_up.tolist()) | set(sig_B_up.tolist())
    pool = np.array([i for i in range(G) if i not in used])
    if len(pool) < 4:
        ctrl_idx = np.arange(min(n_sig, G))
    else:
        ctrl_idx = rng.choice(pool, size=min(n_sig, len(pool)), replace=False)

    score_B_in_A = _signature_score(cells_A, sig_B_up, ctrl_idx)
    score_A_in_B = _signature_score(cells_B, sig_A_up, ctrl_idx)
    score_B_in_B = _signature_score(cells_B, sig_B_up, ctrl_idx)  # direct B response
    score_A_in_A = _signature_score(cells_A, sig_A_up, ctrl_idx)  # direct A response

    out["mean_sigB_in_A"] = float(score_B_in_A.mean())
    out["mean_sigA_in_B"] = float(score_A_in_B.mean())
    out["sigB_in_A_norm"] = float(score_B_in_A.mean() / max(score_B_in_B.mean(), 1e-6))
    out["sigA_in_B_norm"] = float(score_A_in_B.mean() / max(score_A_in_A.mean(), 1e-6))
    out["signature_asymmetry"] = out["sigB_in_A_norm"] - out["sigA_in_B_norm"]

    thr_B = score_B_in_B.mean() + 0.5 * score_B_in_B.std()
    thr_A = score_A_in_A.mean() + 0.5 * score_A_in_A.std()
    out["frac_A_with_high_sigB"] = float((score_B_in_A > thr_B).mean())
    out["frac_B_with_high_sigA"] = float((score_A_in_B > thr_A).mean())
    out["tail_asymmetry"] = out["frac_A_with_high_sigB"] - out["frac_B_with_high_sigA"]

    # ----- 1-D projection on û_{A→B}
    direction = mu_B - mu_A
    direction = direction / _safe_norm(direction)
    proj_A = cells_A @ direction
    proj_B = cells_B @ direction

    out["var_A_along_AB"] = float(proj_A.var())
    out["var_B_along_AB"] = float(proj_B.var())
    out["var_ratio_AB"] = float(proj_A.var() / max(proj_B.var(), 1e-9))
    out["bimodality_A_along_AB"] = _sarle_bimodality(proj_A)
    out["bimodality_B_along_AB"] = _sarle_bimodality(proj_B)
    out["kl_A_to_B_along_AB"] = _kde_kl(proj_A, proj_B)
    out["kl_B_to_A_along_AB"] = _kde_kl(proj_B, proj_A)
    out["kl_asymmetry"] = out["kl_A_to_B_along_AB"] - out["kl_B_to_A_along_AB"]

    return out


# --------------------------------------------------------------------- driver

def compute_all_pairs(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    stimuli: Sequence[str],
    pbs_label: str = "PBS",
    min_cells: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compute the full statistic battery for every ordered (A, B) pair across all
    cell types with at least `min_cells` cells in each of {A, B, PBS}.

    Returns long-format DataFrame with columns:
        A, B, cell_type, n_A, n_B, n_PBS, statistic, value,
        pair_label ('positive' / 'negative' / None), unordered_pair
    """
    rng = np.random.default_rng(seed)
    cell_types = sorted({ct for (_, ct) in cells_by_pair.keys()})

    rows: List[Dict] = []
    for A in stimuli:
        if A == pbs_label:
            continue
        for B in stimuli:
            if B == pbs_label or A == B:
                continue
            for T in cell_types:
                if (A, T) not in cells_by_pair or (B, T) not in cells_by_pair:
                    continue
                if (pbs_label, T) not in cells_by_pair:
                    continue
                cA = cells_by_pair[(A, T)]
                cB = cells_by_pair[(B, T)]
                cP = cells_by_pair[(pbs_label, T)]
                if len(cA) < min_cells or len(cB) < min_cells or len(cP) < min_cells:
                    continue

                stats_dict = compute_pair_statistics(cA, cB, cP, rng=rng)
                label = labeled_pair_status(A, B)
                up = unordered_key(A, B)
                for k, v in stats_dict.items():
                    rows.append({
                        "A": A, "B": B, "cell_type": T,
                        "n_A": int(len(cA)), "n_B": int(len(cB)),
                        "n_PBS": int(len(cP)),
                        "statistic": k, "value": float(v) if v == v else np.nan,
                        "pair_label": label,
                        "unordered_pair": up,
                    })

    return pd.DataFrame(rows)


def aggregate_to_unordered(
    stats_df: pd.DataFrame,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Collapse per-(A,B,cell_type) rows down to one row per (unordered_pair, statistic).

    Step 1: mean across cell types within each ordered direction.
    Step 2: across the two ordered directions, take mean / min / max / std.

    The 'max' aggregate is what we use for AUC by default: an asymmetric
    statistic that is large in one direction but small in the other should
    be characterised by its larger ordered value.
    """
    by_dir = (
        stats_df.groupby(
            ["A", "B", "statistic", "unordered_pair", "pair_label"],
            dropna=False,
        )["value"].agg(agg).reset_index()
    )
    out = (
        by_dir.groupby(
            ["unordered_pair", "statistic", "pair_label"],
            dropna=False,
        )["value"].agg(["mean", "min", "max", "std"]).reset_index()
    )
    return out


# --------------------------------------------------------------------- AUC + null

def _auc_pos_high(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) Mann–Whitney-style AUC with ties = 0.5."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pairs = pos[:, None] - neg[None, :]
    return float((np.sign(pairs) + 1).sum() / (2.0 * pairs.size))


def compute_auc_per_statistic(
    summary_df: pd.DataFrame,
    statistic_value_col: str = "max",
) -> pd.DataFrame:
    """
    Per statistic, compute AUC of separating labeled positives from negatives.

    Both directions are tried (value larger for positives vs smaller for
    positives); `direction` records the winning interpretation.
    """
    rows = []
    labeled = summary_df.dropna(subset=["pair_label"])
    for stat, sub in labeled.groupby("statistic"):
        pos = sub[sub["pair_label"] == "positive"][statistic_value_col].dropna().values
        neg = sub[sub["pair_label"] == "negative"][statistic_value_col].dropna().values
        if len(pos) == 0 or len(neg) == 0:
            continue
        auc_high = _auc_pos_high(pos, neg)
        if auc_high >= 0.5:
            auc = auc_high
            direction = "high_means_cascade"
        else:
            auc = 1.0 - auc_high
            direction = "low_means_cascade"
        rows.append({
            "statistic": stat,
            "auc": float(auc),
            "direction": direction,
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "pos_mean": float(np.mean(pos)),
            "neg_mean": float(np.mean(neg)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("auc", ascending=False).reset_index(drop=True)


def permutation_null_auc(
    summary_df: pd.DataFrame,
    statistic_value_col: str = "max",
    n_permutations: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Permutation null: shuffle pair_label across labeled pairs, recompute AUC
    per statistic. Returns long DataFrame [statistic, perm, auc].

    AUC under shuffled labels is always taken as the *higher* of the two
    directional AUCs (matching the orientation choice in
    `compute_auc_per_statistic`).
    """
    rng = np.random.default_rng(seed)
    labeled = summary_df.dropna(subset=["pair_label"]).copy()
    labels = labeled["pair_label"].values.copy()

    rows = []
    for perm in range(n_permutations):
        shuffled = rng.permutation(labels)
        tmp = labeled.assign(pair_label=shuffled)
        for stat, sub in tmp.groupby("statistic"):
            pos = sub[sub["pair_label"] == "positive"][statistic_value_col].dropna().values
            neg = sub[sub["pair_label"] == "negative"][statistic_value_col].dropna().values
            if len(pos) == 0 or len(neg) == 0:
                continue
            auc_high = _auc_pos_high(pos, neg)
            auc = max(auc_high, 1.0 - auc_high)
            rows.append({"statistic": stat, "perm": perm, "auc": auc})
    return pd.DataFrame(rows)
