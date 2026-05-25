"""
Adversarial audit of the pathway-signature cascade methodology (§23).

Four independent checks, each designed to falsify a specific potential
weakness in the original result:

  1) Cytokine-label permutation null
       Tests: is the observed binary AUC structurally inflated, or is it
       distinguishable from random label assignment?

  2) Random-pathway null
       Tests: does the curated gene list carry specific information, or do
       random gene sets of the same size give equally high AUCs (which
       would imply the test is dominated by overall expression level, not
       pathway specificity)?

  3) Donor-level inference
       Tests: do the magnitude predictions survive when statistics respect
       pseudo-replication (per-donor means + Wilcoxon signed-rank), or were
       the per-cell p-values fiction?

  4) Directional asymmetry test
       Tests: does the data actually show cascade DIRECTION (A→B engages
       both pathways; B engages only its own), or only pathway engagement?
       This is the test the original analysis silently skipped.
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


# ---------------------------------------------------------------------------
# Loaders that preserve donor identity (needed for audit 3 — donor-level test)
# ---------------------------------------------------------------------------

def load_cells_by_donor(
    manifest_path: str,
    gene_names: Optional[List[str]] = None,
    time_filter: Optional[str] = None,
) -> Tuple[Dict[Tuple[str, str, str], np.ndarray], List[str]]:
    """
    Load pseudotubes and return cells indexed by (cytokine, cell_type, donor).

    Unlike `load_phase1_cells` (which pools across donors), this version keeps
    donor identity so we can compute per-donor statistics for honest
    inference under pseudo-replication.
    """
    with open(manifest_path) as f:
        entries = json.load(f)

    pooled: Dict[Tuple[str, str, str], List[np.ndarray]] = defaultdict(list)
    resolved_genes: Optional[List[str]] = None

    for entry in entries:
        cytokine = entry["cytokine"]
        donor = entry["donor"]
        adata = sc.read_h5ad(entry["path"])
        if gene_names is not None:
            adata = adata[:, gene_names]
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
                pooled[(cytokine, ct, donor)].append(X[mask])

    by_triple = {k: np.concatenate(v, axis=0) for k, v in pooled.items()}
    return by_triple, resolved_genes or []


# ---------------------------------------------------------------------------
# Audit 1: cytokine-label permutation null on binary AUC
# ---------------------------------------------------------------------------

def _auc_pos_above_neg(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) with ties = 0.5."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pairs = pos[:, None] - neg[None, :]
    return float((np.sign(pairs) + 1).sum() / (2.0 * pairs.size))


def permutation_null_binary_auc(
    penetration_df: pd.DataFrame,
    pos_stims: Sequence[str],
    neg_stims: Sequence[str],
    primary_stim: str,
    pathway: str = "IFNAR_induced",
    n_permutations: int = 2000,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    For each cell type, compute the observed binary AUC (pos_stims vs neg_stims
    on penetration values), then shuffle the labels among the union pos∪neg
    stimuli n_permutations times and recompute the AUC. Report empirical
    p-value = fraction of null AUCs ≥ observed.

    NOTE: the primary stimulus (which has penetration=1 by definition) is
    excluded from both the observed test and the null shuffling. Only
    cascade-test stimuli (PIC, LPS, LPSlo on the positive side; P3CSK, CpG,
    TNF on the negative side, for the IFNAR test) get shuffled.
    """
    rng = np.random.default_rng(seed)
    sub = penetration_df[
        (penetration_df["pathway"] == pathway)
        & (penetration_df["primary_stim"] == primary_stim)
    ].copy()

    test_pos = [s for s in pos_stims if s != primary_stim]
    test_neg = list(neg_stims)
    all_test = test_pos + test_neg
    n_pos = len(test_pos)
    n_neg = len(test_neg)
    if n_pos == 0 or n_neg == 0:
        return {"observed": pd.DataFrame(), "null_summary": pd.DataFrame()}

    obs_rows = []
    null_summary_rows = []
    cell_types = sorted(sub["cell_type"].unique())
    for T in cell_types:
        sub_T = sub[sub["cell_type"] == T].copy()
        pen = dict(zip(sub_T["A"], sub_T["penetration"]))
        # Values for the test set
        vals = np.array([pen[s] for s in all_test if s in pen and not np.isnan(pen[s])])
        valid_stims = [s for s in all_test if s in pen and not np.isnan(pen[s])]
        if len(vals) < 2:
            continue
        # Observed AUC: first n_pos positions correspond to test_pos (if present)
        present_pos = [i for i, s in enumerate(valid_stims) if s in test_pos]
        present_neg = [i for i, s in enumerate(valid_stims) if s in test_neg]
        if len(present_pos) == 0 or len(present_neg) == 0:
            continue
        obs_auc = _auc_pos_above_neg(vals[present_pos], vals[present_neg])

        # Null distribution
        null_aucs = []
        n_present_pos = len(present_pos)
        n_present_neg = len(present_neg)
        for _ in range(n_permutations):
            perm = rng.permutation(len(vals))
            null_pos = vals[perm[:n_present_pos]]
            null_neg = vals[perm[n_present_pos:n_present_pos + n_present_neg]]
            null_aucs.append(_auc_pos_above_neg(null_pos, null_neg))
        null_aucs = np.array(null_aucs)

        # Empirical one-sided p
        p_emp = float((null_aucs >= obs_auc).mean())

        obs_rows.append({
            "cell_type": T,
            "n_pos": int(n_present_pos),
            "n_neg": int(n_present_neg),
            "observed_auc": float(obs_auc),
            "null_mean": float(np.mean(null_aucs)),
            "null_q95": float(np.quantile(null_aucs, 0.95)),
            "null_q99": float(np.quantile(null_aucs, 0.99)),
            "p_emp": p_emp,
        })

        for a in null_aucs:
            null_summary_rows.append({"cell_type": T, "auc": float(a)})

    return {
        "observed": pd.DataFrame(obs_rows),
        "null_summary": pd.DataFrame(null_summary_rows),
    }


# ---------------------------------------------------------------------------
# Audit 2: random-pathway null (curated genes vs random gene sets)
# ---------------------------------------------------------------------------

def random_pathway_null(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    gene_names: Sequence[str],
    pathway_idx_curated: np.ndarray,
    pos_stims: Sequence[str],
    neg_stims: Sequence[str],
    primary_stim: str,
    excluded_gene_indices: Optional[Sequence[int]] = None,
    n_random_pathways: int = 200,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Generate n random gene sets (same SIZE as the curated pathway), drawn from
    genes NOT in any curated pathway. For each random gene set:
        - compute mean per-cell pathway score for every cytokine A
        - compute "penetration" using the same primary_stim as denominator
        - compute binary AUC (pos vs neg) for each cell type
    Returns the AUC distribution under random gene-set selection.

    If random gene sets give AUCs similar to the curated one, the test is
    NOT specific to the curated genes; we are reading overall expression
    activation, not pathway-specific induction.

    `excluded_gene_indices` should contain ALL indices used by any curated
    pathway, to ensure the random sets do not overlap with the curated ones.
    """
    rng = np.random.default_rng(seed)
    n_genes = len(gene_names)
    n_set = len(pathway_idx_curated)
    excluded = set(excluded_gene_indices or [])
    excluded.update(pathway_idx_curated.tolist())
    pool = np.array([i for i in range(n_genes) if i not in excluded])
    if len(pool) < n_set:
        raise ValueError(
            f"Not enough non-curated genes to sample random sets "
            f"of size {n_set}: only {len(pool)} available."
        )

    def _penetration_for_set(idx: np.ndarray, sub_cells_by_pair):
        """Compute penetration for every (cytokine, cell_type) using this gene set."""
        rows = []
        cell_types = sorted({ct for (_, ct) in sub_cells_by_pair.keys()})
        stims = sorted({s for (s, _) in sub_cells_by_pair.keys()})
        for T in cell_types:
            if (primary_stim, T) not in sub_cells_by_pair:
                continue
            if ("PBS", T) not in sub_cells_by_pair:
                continue
            cB = sub_cells_by_pair[(primary_stim, T)]
            cP = sub_cells_by_pair[("PBS", T)]
            sB = float(cB[:, idx].mean())
            sP = float(cP[:, idx].mean())
            denom = sB - sP
            if abs(denom) < 1e-9:
                continue
            for A in stims:
                if (A, T) not in sub_cells_by_pair or A in (primary_stim, "PBS"):
                    if A != primary_stim:
                        continue
                cA = sub_cells_by_pair[(A, T)]
                sA = float(cA[:, idx].mean())
                pen = (sA - sP) / denom
                rows.append({"A": A, "cell_type": T, "penetration": pen})
        return pd.DataFrame(rows)

    # AUC per (random-pathway, cell-type)
    null_aucs_rows = []
    for k in range(n_random_pathways):
        random_idx = rng.choice(pool, size=n_set, replace=False)
        df = _penetration_for_set(random_idx, cells_by_pair)
        for T, sub in df.groupby("cell_type"):
            pen = dict(zip(sub["A"], sub["penetration"]))
            pos_v = np.array([pen[s] for s in pos_stims if s in pen and s != primary_stim])
            neg_v = np.array([pen[s] for s in neg_stims if s in pen])
            pos_v = pos_v[~np.isnan(pos_v)]
            neg_v = neg_v[~np.isnan(neg_v)]
            if len(pos_v) == 0 or len(neg_v) == 0:
                continue
            auc = _auc_pos_above_neg(pos_v, neg_v)
            null_aucs_rows.append({
                "random_pathway_id": k,
                "cell_type": T,
                "auc": float(auc),
            })
    null_aucs = pd.DataFrame(null_aucs_rows)

    # Per cell type: quantiles
    null_summary = (
        null_aucs.groupby("cell_type")["auc"]
        .agg(mean="mean", q05=lambda x: x.quantile(0.05),
             q50=lambda x: x.quantile(0.50),
             q95=lambda x: x.quantile(0.95),
             q99=lambda x: x.quantile(0.99), count="count")
        .reset_index()
    )

    return {"random_aucs": null_aucs, "summary": null_summary}


# ---------------------------------------------------------------------------
# Audit 3: per-donor inference (Wilcoxon signed-rank)
# ---------------------------------------------------------------------------

def per_donor_magnitude_test(
    cells_by_triple: Dict[Tuple[str, str, str], np.ndarray],
    pathway_idx: np.ndarray,
    A_upstream: str,
    B_downstream: str,
    pbs_label: str = "PBS",
    min_cells: int = 10,
) -> pd.DataFrame:
    """
    For each cell type T: compute mean s_pathway per donor for A, B, PBS.
    Then test A_means > B_means > PBS_means using paired Wilcoxon signed-rank
    on donors present in both A and B (and PBS).

    Returns per-cell-type DataFrame: cell_type, n_donors_paired, mean_A_overall,
    mean_B_overall, mean_PBS_overall, wilcoxon_p_A_gt_B, wilcoxon_p_B_gt_PBS,
    sign_test_p_A_gt_B (binomial), pass_ordering, pass_significant_alpha.
    """
    cell_types = sorted({ct for (_, ct, _) in cells_by_triple.keys()})
    rows = []
    for T in cell_types:
        # Per-donor means
        donor_A = {}
        donor_B = {}
        donor_PBS = {}
        for (cyt, ct, d), cells in cells_by_triple.items():
            if ct != T or len(cells) < min_cells:
                continue
            mean_val = float(cells[:, pathway_idx].mean())
            if cyt == A_upstream:
                donor_A[d] = mean_val
            elif cyt == B_downstream:
                donor_B[d] = mean_val
            elif cyt == pbs_label:
                donor_PBS[d] = mean_val

        # Donors present in both A and B
        paired_AB = sorted(set(donor_A) & set(donor_B))
        paired_BP = sorted(set(donor_B) & set(donor_PBS))

        if len(paired_AB) == 0:
            continue

        A_vals = np.array([donor_A[d] for d in paired_AB])
        B_vals = np.array([donor_B[d] for d in paired_AB])
        diff_AB = A_vals - B_vals
        n_pos = int(np.sum(diff_AB > 0))
        # one-sided binomial sign test
        try:
            sign_p = float(scstats.binomtest(n_pos, len(diff_AB), 0.5,
                                              alternative="greater").pvalue)
        except Exception:
            sign_p = float("nan")
        # one-sided Wilcoxon signed-rank
        if len(diff_AB) >= 1 and not np.all(diff_AB == 0):
            try:
                _, w_p_AB = scstats.wilcoxon(
                    A_vals, B_vals, alternative="greater",
                    zero_method="zsplit",
                )
                w_p_AB = float(w_p_AB)
            except Exception:
                w_p_AB = float("nan")
        else:
            w_p_AB = float("nan")

        # B > PBS
        if paired_BP:
            B2 = np.array([donor_B[d] for d in paired_BP])
            P2 = np.array([donor_PBS[d] for d in paired_BP])
            diff_BP = B2 - P2
            if not np.all(diff_BP == 0):
                try:
                    _, w_p_BP = scstats.wilcoxon(
                        B2, P2, alternative="greater",
                        zero_method="zsplit",
                    )
                    w_p_BP = float(w_p_BP)
                except Exception:
                    w_p_BP = float("nan")
            else:
                w_p_BP = float("nan")
        else:
            w_p_BP = float("nan")

        rows.append({
            "cell_type": T,
            "A_upstream": A_upstream,
            "B_downstream": B_downstream,
            "n_donors_paired_AB": int(len(paired_AB)),
            "n_donors_paired_BP": int(len(paired_BP)),
            "mean_A": float(A_vals.mean()),
            "mean_B": float(B_vals.mean()),
            "mean_PBS": (
                float(np.mean([donor_PBS[d] for d in paired_BP]))
                if paired_BP else float("nan")
            ),
            "sign_p_A_gt_B": sign_p,
            "wilcoxon_p_A_gt_B": w_p_AB,
            "wilcoxon_p_B_gt_PBS": w_p_BP,
            "n_pos_A_gt_B": n_pos,
            "n_total": int(len(diff_AB)),
            "pass_ordering": bool(A_vals.mean() > B_vals.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Audit 4: directional asymmetry — the test the original analysis skipped
# ---------------------------------------------------------------------------

def directional_asymmetry_test(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    pathway_idx_dict: Dict[str, np.ndarray],
    A: str,
    B: str,
    P_A: str,
    P_B: str,
    pbs_label: str = "PBS",
    min_cells: int = 10,
) -> pd.DataFrame:
    """
    A→B cascade (textbook: PIC→IFN-β) predicts:
        s(A, P_A) > s(B, P_A)   ← only A engages P_A directly
        s(A, P_B) ≈ s(B, P_B)   ← both engage P_B (A via cascade, B directly)

    Anti-cascade direction (B→A, false) would predict:
        s(B, P_B) > s(A, P_B)   ← only B engages P_B
        s(B, P_A) ≈ s(A, P_A)   ← both engage P_A

    NO cascade either way:
        s(A, P_A) > s(B, P_A) AND s(B, P_B) > s(A, P_B); independent.

    This function reports all four quantities and the asymmetries, so the
    user can see whether the data actually shows directional cascade
    structure or only pathway engagement.

    Returns DataFrame per cell type with columns:
        cell_type, sA_in_PA, sA_in_PB, sB_in_PA, sB_in_PB, sPBS_in_PA,
        sPBS_in_PB, asym_PA (sA-sB on P_A), asym_PB (sA-sB on P_B),
        directional_score = asym_PA - asym_PB
        (positive ⇒ consistent with A→B cascade)
    """
    idx_A = pathway_idx_dict[P_A]
    idx_B = pathway_idx_dict[P_B]
    cell_types = sorted({ct for (_, ct) in cells_by_pair.keys()})
    rows = []
    for T in cell_types:
        needed = [(A, T), (B, T), (pbs_label, T)]
        if any(k not in cells_by_pair for k in needed):
            continue
        cA = cells_by_pair[(A, T)]
        cB = cells_by_pair[(B, T)]
        cP = cells_by_pair[(pbs_label, T)]
        if len(cA) < min_cells or len(cB) < min_cells or len(cP) < min_cells:
            continue
        sA_in_PA = float(cA[:, idx_A].mean())
        sA_in_PB = float(cA[:, idx_B].mean())
        sB_in_PA = float(cB[:, idx_A].mean())
        sB_in_PB = float(cB[:, idx_B].mean())
        sP_in_PA = float(cP[:, idx_A].mean())
        sP_in_PB = float(cP[:, idx_B].mean())
        # Normalise to PBS baseline
        sA_PA_norm = sA_in_PA - sP_in_PA
        sA_PB_norm = sA_in_PB - sP_in_PB
        sB_PA_norm = sB_in_PA - sP_in_PA
        sB_PB_norm = sB_in_PB - sP_in_PB
        asym_PA = sA_PA_norm - sB_PA_norm    # A engages P_A more than B does?
        asym_PB = sA_PB_norm - sB_PB_norm    # A engages P_B more than B does?
        # Cascade A→B predicts asym_PA > 0 (A engages P_A directly; B doesn't)
        # AND asym_PB ≈ 0 (both engage P_B). So directional_score = asym_PA − asym_PB
        # should be POSITIVE for true cascade A→B.
        directional_score = asym_PA - asym_PB
        rows.append({
            "cell_type": T,
            "A": A, "B": B, "P_A": P_A, "P_B": P_B,
            "sA_in_PA": sA_in_PA, "sB_in_PA": sB_in_PA,
            "sA_in_PB": sA_in_PB, "sB_in_PB": sB_in_PB,
            "sPBS_in_PA": sP_in_PA, "sPBS_in_PB": sP_in_PB,
            "sA_PA_norm": sA_PA_norm, "sA_PB_norm": sA_PB_norm,
            "sB_PA_norm": sB_PA_norm, "sB_PB_norm": sB_PB_norm,
            "asym_PA": asym_PA,
            "asym_PB": asym_PB,
            "directional_score": directional_score,
            "interpretation": (
                "A->B (cascade A drives B's pathway, not vice versa)" if directional_score > 0
                else "B->A (reverse direction would be implied)" if directional_score < 0
                else "ambiguous/symmetric"
            ),
        })
    return pd.DataFrame(rows)
