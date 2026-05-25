"""
Pathway-signature analysis for cascade discovery.

Replaces the empirical top-DE signature (which collapses onto a diagonal in
500-gene targeted panels) with **literature-curated, signaling-adaptor-
specific gene sets**. Each pathway has:

  - `up`: marker genes that go up when the pathway is engaged
  - `primary_for`: stimuli that engage this pathway DIRECTLY (no cascade)
  - `cascade_from`: stimuli that engage this pathway VIA autocrine cascade

The central readout is the **cascade penetration** score:

    penetration(A → P, B) =
        (mean_s_P(A-tube) − mean_s_P(PBS)) / (mean_s_P(B-tube) − mean_s_P(PBS))

where B is the "primary" stimulus for pathway P. A score near 1 means
A-tube cells recapitulate B's pathway as fully as direct B stimulation;
near 0 means A does not engage P at all. Asymmetric by construction.

Gene symbols are mouse (Sheu BMDM data).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scstats


# ---------------------------------------------------------------------------
# Curated pathway library (mouse gene symbols)
# ---------------------------------------------------------------------------

PATHWAY_SIGNATURES: Dict[str, Dict] = {
    "IRF3_direct": {
        "up": ["Ifnb1", "Ccl5", "Cxcl10", "Ifit2", "Ifit3"],
        "primary_for": ["PIC", "LPS"],          # TLR3 → TRIF and TLR4 → TRIF arm
        "cascade_from": [],
        "rationale": "TRIF → TBK1 → IRF3 → ISRE-binding. Induced DIRECTLY upon "
                     "TLR3 or (weaker) TLR4 engagement; does not require autocrine IFN.",
    },
    "IFNAR_induced": {
        # Note: Sheu's targeted 500-gene panel excludes some canonical ISGs
        # (Isg15, Stat1, Usp18, Oas1a/2/3); we use the IFN-induced genes that
        # ARE in the panel, including OAS-family paralog Oasl1 and Ifit
        # paralogs Ifit1bl1 / Ifit3b that capture similar biology.
        "up": ["Mx1", "Mx2", "Ifit1", "Ifit1bl1", "Ifit3", "Ifit3b",
               "Rsad2", "Irf7", "Oasl1"],
        "primary_for": ["IFNb"],                # IFNb ligand → IFNAR directly
        "cascade_from": ["PIC", "LPS"],         # via autocrine IFN-β
        "rationale": "IFNAR → JAK1/TYK2 → STAT1/STAT2 + IRF9 → ISGF3 → ISRE. "
                     "Clean cascade product: appears in PIC/LPS tubes only via "
                     "autocrine IFN-β; appears in IFNb tubes directly.",
    },
    "NFkB_canonical": {
        "up": ["Tnf", "Il1b", "Il6", "Nfkbia", "Nfkbid", "Tnfaip3",
               "Cxcl1", "Cxcl2", "Ccl3", "Ccl4", "Birc3"],
        "primary_for": ["LPS", "LPSlo", "P3CSK", "CpG", "TNF"],
        "cascade_from": [],
        "rationale": "p65/p50 NF-κB → κB sites. Engaged by basically all "
                     "TLR-MyD88 and TNFR-NF-κB pathways → poor stimulus "
                     "specificity. Useful for MAGNITUDE comparisons "
                     "(LPS+autocrine-TNF > TNF-direct > PBS) not cascade existence.",
    },
    "TNFR_autocrine": {
        # Birc3 is not in the Sheu panel; replaced with Nfkbie + Nfkbiz, both
        # NF-κB inhibitors classically induced by TNFR1 / IL-1R signaling.
        "up": ["Tnfaip3", "Nfkbid", "Nfkbie", "Nfkbiz"],
        "primary_for": ["TNF"],
        "cascade_from": ["LPS", "LPSlo", "P3CSK", "CpG"],
        "rationale": "Subset of NF-κB regulators classically induced via TNFR1→TRADD "
                     "→TRAF2→IKK. Overlaps with general NF-κB so works best as a "
                     "consistency check rather than a primary discriminator.",
    },
}


# Per-stimulus primary pathway map (derived from PATHWAY_SIGNATURES.primary_for).
STIMULUS_PRIMARY_PATHWAYS: Dict[str, List[str]] = {
    "LPS":   ["NFkB_canonical", "IRF3_direct"],
    "LPSlo": ["NFkB_canonical"],
    "P3CSK": ["NFkB_canonical"],
    "PIC":   ["IRF3_direct"],
    "TNF":   ["NFkB_canonical", "TNFR_autocrine"],
    "CpG":   ["NFkB_canonical"],
    "IFNb":  ["IFNAR_induced"],
    "PBS":   [],
}


# ---------------------------------------------------------------------------
# Human pathway library for Oesinghaus (PBMC 24h, 91 cytokines, 4000 HVGs).
# All gene symbols are HUMAN uppercase. The IFNAR_induced signature is much
# larger here because the 4000-HVG dataset contains the canonical ISGs that
# were missing from Sheu's targeted 500-gene panel.
# ---------------------------------------------------------------------------

PATHWAY_SIGNATURES_HUMAN: Dict[str, Dict] = {
    "IFNAR_induced": {
        "up": [
            # Canonical ISGs — should virtually all be in the 4000 HVGs
            "ISG15", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "OASL",
            "IFIT1", "IFIT2", "IFIT3", "IFI44", "IFI44L", "IFI27",
            "RSAD2", "STAT1", "IRF7", "USP18", "ISG20", "BST2",
            "IFITM1", "IFITM3", "DDX60", "HERC5", "EPSTI1",
        ],
        # Cytokines that DIRECTLY engage IFNAR (predicted high penetration).
        # Auto-matched by substring against actual cytokine names.
        "primary_patterns": ["IFN-alpha", "IFN-beta", "IFN-α", "IFN-β",
                              "IFNA", "IFNB", "IFNa1", "IFNb1"],
        # Other IFN family members that engage STAT1/2 and share many ISGs
        # downstream (predicted partial penetration — also positives).
        "extended_positive_patterns": ["IFN-gamma", "IFN-γ", "IFNG",
                                        "IFN-lambda", "IFN-λ", "IL-29",
                                        "IL-28", "IFNL"],
        "rationale": "Direct ISGs induced by type-I/II/III IFN via JAK-STAT.",
    },
    "NFkB_canonical": {
        "up": [
            "TNF", "IL1B", "IL6", "NFKBIA", "NFKBID", "NFKBIE", "NFKBIZ",
            "TNFAIP3", "BIRC3", "CXCL1", "CXCL2", "CXCL3", "CCL3", "CCL4",
            "PTGS2", "CXCL8", "IL8", "ICAM1",
        ],
        # Cytokines that strongly activate NF-κB through receptor-proximal signalling.
        "primary_patterns": ["TNF", "TNF-alpha", "TNF-α", "TNFA",
                              "IL-1", "IL1A", "IL1B", "IL-1α", "IL-1β"],
        "rationale": "p65/p50 NF-κB targets. Engaged by TNFR1, IL1R, and several "
                     "TLR pathways — broad positive set.",
    },
}


# Validation pseudo-donor convention (Oesinghaus: D2 and D3 are held out per §16).
OESINGHAUS_VAL_DONORS_DEFAULT = ["Donor2", "Donor3"]


def subsample_oesinghaus_manifest(
    manifest: List[Dict],
    val_donors: Optional[Sequence[str]] = None,
    tubes_per_pair: int = 1,
) -> List[Dict]:
    """
    Subsample Oesinghaus manifest (~9100 entries) to train-donors × one tube per
    (donor, cytokine). Default keeps 1 tube/pair, so the subset has ~910
    entries (≈91 cytokines × 10 train donors). Excludes val donors entirely.

    Args:
        manifest: full Oesinghaus manifest list
        val_donors: donor names to drop entirely (defaults to D2 and D3 per §16)
        tubes_per_pair: number of tubes per (donor, cytokine) to keep

    Returns:
        subsampled manifest list (sorted by donor then cytokine then tube_idx)
    """
    if val_donors is None:
        val_donors = OESINGHAUS_VAL_DONORS_DEFAULT
    val_set = set(val_donors)

    by_pair: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for e in manifest:
        if e["donor"] in val_set:
            continue
        by_pair[(e["donor"], e["cytokine"])].append(e)

    subset: List[Dict] = []
    for (d, c), entries in by_pair.items():
        entries_sorted = sorted(entries, key=lambda x: x.get("tube_idx", 0))
        subset.extend(entries_sorted[:tubes_per_pair])
    subset.sort(key=lambda e: (e["donor"], e["cytokine"], e.get("tube_idx", 0)))
    return subset


def match_cytokines_by_patterns(
    cytokines: Sequence[str],
    patterns: Sequence[str],
) -> List[str]:
    """Substring match (case-insensitive) of cytokine names against pattern list.
    Returns the actual cytokine names that match any pattern."""
    pats = [p.lower() for p in patterns]
    return sorted([
        c for c in cytokines
        if any(p in c.lower() for p in pats)
    ])


# Pre-registered cascade-existence binary benchmark for IFNAR_induced
# (Sheu §21 pairs collapsed to "does this stimulus drive IFN-cascade?").
IFNAR_POSITIVE_STIMULI = ["PIC", "LPS", "LPSlo", "IFNb"]
IFNAR_NEGATIVE_STIMULI = ["P3CSK", "CpG", "TNF"]


# ---------------------------------------------------------------------------
# Gene resolution
# ---------------------------------------------------------------------------

def resolve_pathway_genes(
    pathway: str,
    gene_names: Sequence[str],
    min_hits: int = 3,
    signatures: Optional[Dict[str, Dict]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Map a curated pathway to integer column indices in the loaded data.

    Args:
        signatures: dict to look up the pathway from. Defaults to
            PATHWAY_SIGNATURES (mouse, Sheu). Pass PATHWAY_SIGNATURES_HUMAN
            for Oesinghaus.

    Returns:
        idx: int array of column indices present in gene_names
        found: list of gene symbols that were resolved
        missing: list of gene symbols not in gene_names
    Raises:
        ValueError if fewer than `min_hits` curated genes are present.
    """
    if signatures is None:
        signatures = PATHWAY_SIGNATURES
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    found, missing = [], []
    idx_list = []
    for g in signatures[pathway]["up"]:
        if g in gene_to_idx:
            found.append(g)
            idx_list.append(gene_to_idx[g])
        else:
            missing.append(g)
    if len(found) < min_hits:
        raise ValueError(
            f"Pathway {pathway}: only {len(found)} of "
            f"{len(signatures[pathway]['up'])} curated genes "
            f"present in panel (need >= {min_hits}). Missing: {missing}"
        )
    return np.array(idx_list, dtype=np.int64), found, missing


def resolve_all_pathways(
    gene_names: Sequence[str],
    min_hits: int = 3,
    signatures: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """
    Resolve all pathways against a gene panel. Pathways with too few hits
    are flagged ok=False but their actual found/missing lists are preserved
    for diagnostic reporting.

    Args:
        signatures: dict of {pathway -> {"up": [genes], ...}}. Defaults to
            PATHWAY_SIGNATURES (mouse). Pass PATHWAY_SIGNATURES_HUMAN for
            Oesinghaus.

    Returns dict {pathway -> {idx, found, missing, ok, reason?}}.
    """
    if signatures is None:
        signatures = PATHWAY_SIGNATURES
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    resolved = {}
    for p in signatures:
        found, missing, idx_list = [], [], []
        for g in signatures[p]["up"]:
            if g in gene_to_idx:
                found.append(g)
                idx_list.append(gene_to_idx[g])
            else:
                missing.append(g)
        ok = len(found) >= min_hits
        resolved[p] = {
            "idx": np.array(idx_list, dtype=np.int64) if ok else None,
            "found": found,
            "missing": missing,
            "ok": ok,
        }
        if not ok:
            resolved[p]["reason"] = (
                f"Only {len(found)} of {len(signatures[p]['up'])} "
                f"genes resolved (need >= {min_hits})"
            )
    return resolved


def pick_control_genes(
    n_ctrl: int,
    pathway_idx: Dict[str, np.ndarray],
    n_genes: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Pick `n_ctrl` random control gene indices that are NOT in any pathway
    signature. Used to subtract overall expression background per cell.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    used = set()
    for idx in pathway_idx.values():
        if idx is not None:
            used.update(idx.tolist())
    pool = np.array([i for i in range(n_genes) if i not in used])
    n = min(n_ctrl, len(pool))
    return rng.choice(pool, size=n, replace=False)


# ---------------------------------------------------------------------------
# Per-cell pathway score
# ---------------------------------------------------------------------------

def compute_pathway_score(
    cells: np.ndarray,
    pathway_idx: np.ndarray,
    control_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-cell pathway score = mean(pathway genes).

    NOTE: control_idx is accepted for back-compatibility but IGNORED. On a
    500-gene targeted immune-response panel every gene moves under some
    stimulation, so "random control genes" carry their own pathway signal
    and bias the subtraction. The PBS baseline is taken at the tube level
    in `compute_penetration`, which already removes resting-state activity.
    """
    return cells[:, pathway_idx].mean(axis=1)


# ---------------------------------------------------------------------------
# Cascade penetration
# ---------------------------------------------------------------------------

def compute_penetration(
    s_A: np.ndarray,
    s_B: np.ndarray,
    s_PBS: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Cascade penetration of stimulus A toward pathway P, where B is the
    primary stimulus for P. Vectors are per-cell pathway scores.

        penetration = (mean(s_A) − mean(s_PBS)) / (mean(s_B) − mean(s_PBS))

    Returns float (can be > 1 if A induces more than direct B, or negative
    if A is less than baseline).
    """
    num = float(np.mean(s_A) - np.mean(s_PBS))
    den = float(np.mean(s_B) - np.mean(s_PBS))
    if abs(den) < eps:
        return float("nan")
    return num / den


def compute_pair_penetration(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    pathway: str,
    pathway_idx: np.ndarray,
    control_idx: np.ndarray,
    primary_stim: str,
    pbs_label: str = "PBS",
    min_cells: int = 10,
) -> pd.DataFrame:
    """
    For one pathway P (with primary_stim B), compute penetration of every
    stimulus A toward P, stratified by cell type.

    Returns long DataFrame with columns:
        pathway, primary_stim, A, cell_type, n_A, n_B, n_PBS,
        mean_score_A, mean_score_B, mean_score_PBS, penetration
    """
    cell_types = sorted({ct for (_, ct) in cells_by_pair.keys()})
    stimuli = sorted({cyt for (cyt, _) in cells_by_pair.keys()})

    rows = []
    for T in cell_types:
        if (primary_stim, T) not in cells_by_pair:
            continue
        if (pbs_label, T) not in cells_by_pair:
            continue
        cB = cells_by_pair[(primary_stim, T)]
        cP = cells_by_pair[(pbs_label, T)]
        if len(cB) < min_cells or len(cP) < min_cells:
            continue
        sB = compute_pathway_score(cB, pathway_idx, control_idx)
        sP = compute_pathway_score(cP, pathway_idx, control_idx)

        for A in stimuli:
            if (A, T) not in cells_by_pair:
                continue
            cA = cells_by_pair[(A, T)]
            if len(cA) < min_cells:
                continue
            sA = compute_pathway_score(cA, pathway_idx, control_idx)
            pen = compute_penetration(sA, sB, sP)
            rows.append({
                "pathway": pathway,
                "primary_stim": primary_stim,
                "A": A,
                "cell_type": T,
                "n_A": int(len(cA)),
                "n_B": int(len(cB)),
                "n_PBS": int(len(cP)),
                "mean_score_A": float(np.mean(sA)),
                "mean_score_B": float(np.mean(sB)),
                "mean_score_PBS": float(np.mean(sP)),
                "penetration": float(pen),
            })
    return pd.DataFrame(rows)


def compute_all_penetrations(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    gene_names: Sequence[str],
    n_ctrl: int = 20,
    min_cells: int = 10,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Compute cascade penetration of every stimulus toward every pathway's
    primary stimulus, across cell types. Returns (df, resolved_pathways).

    Only pathways whose primary stimulus is uniquely defined are processed
    (one B per pathway — the first in `primary_for`).
    """
    rng = np.random.default_rng(seed)
    resolved = resolve_all_pathways(gene_names)

    pathway_idx_map = {p: r["idx"] for p, r in resolved.items() if r["ok"]}
    n_genes = next(iter(cells_by_pair.values())).shape[1]
    control_idx = pick_control_genes(n_ctrl, pathway_idx_map, n_genes, rng=rng)

    all_rows = []
    for pathway, r in resolved.items():
        if not r["ok"]:
            continue
        primary_list = PATHWAY_SIGNATURES[pathway]["primary_for"]
        if not primary_list:
            continue
        # Use first primary_for entry as the reference B; if multiple,
        # also test each as B by adding extra rows.
        for primary_stim in primary_list:
            df = compute_pair_penetration(
                cells_by_pair, pathway, r["idx"], control_idx,
                primary_stim, min_cells=min_cells,
            )
            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame(), resolved
    return pd.concat(all_rows, ignore_index=True), resolved


# ---------------------------------------------------------------------------
# Pre-registered IFNAR binary test
# ---------------------------------------------------------------------------

def ifnar_binary_test(
    penetration_df: pd.DataFrame,
    pathway: str = "IFNAR_induced",
    primary_stim: str = "IFNb",
    pos_stimuli: Sequence[str] = tuple(IFNAR_POSITIVE_STIMULI),
    neg_stimuli: Sequence[str] = tuple(IFNAR_NEGATIVE_STIMULI),
) -> pd.DataFrame:
    """
    The pre-registered cascade-existence binary test (§23).

    For each cell type T, rank each non-primary stimulus by penetration of
    IFNAR_induced toward IFNb. Compute AUC of (pos_stimuli ranked above
    neg_stimuli). pos includes the primary (penetration = 1 by construction)
    and the cascade stimuli (PIC, LPS, LPSlo).

    Returns DataFrame indexed by cell type with columns:
        cell_type, n_pos, n_neg, auc, mean_pen_pos, mean_pen_neg,
        sep_clean (bool: all pos > all neg)
    """
    df = penetration_df[
        (penetration_df["pathway"] == pathway)
        & (penetration_df["primary_stim"] == primary_stim)
    ].copy()
    rows = []
    for T, sub in df.groupby("cell_type"):
        pen_by_stim = dict(zip(sub["A"], sub["penetration"]))
        pos = np.array(
            [pen_by_stim[s] for s in pos_stimuli if s in pen_by_stim and not np.isnan(pen_by_stim[s])]
        )
        neg = np.array(
            [pen_by_stim[s] for s in neg_stimuli if s in pen_by_stim and not np.isnan(pen_by_stim[s])]
        )
        if len(pos) == 0 or len(neg) == 0:
            continue
        # AUC = P(pos > neg) with ties = 0.5
        pairs = pos[:, None] - neg[None, :]
        auc = (np.sign(pairs) + 1).sum() / (2.0 * pairs.size)
        rows.append({
            "cell_type": T,
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
            "auc": float(auc),
            "mean_pen_pos": float(pos.mean()),
            "mean_pen_neg": float(neg.mean()),
            "min_pen_pos": float(pos.min()),
            "max_pen_neg": float(neg.max()),
            "sep_clean": bool(pos.min() > neg.max()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Magnitude test for shared-pathway cascade (e.g., LPS→TNF on NF-κB)
# ---------------------------------------------------------------------------

def magnitude_cascade_test(
    penetration_df: pd.DataFrame,
    pathway: str,
    A_upstream: str,
    B_downstream: str,
) -> pd.DataFrame:
    """
    For a cascade A→B that shares the same pathway P, the upstream A should
    show HIGHER pathway activation than direct-B (because A engages B's pathway
    directly AND gets autocrine boost from cascade B).

    Test per cell type: is mean_score_A > mean_score_B > mean_score_PBS?

    NOTE: This is the ordering-only version. For full inference (with
    Mann-Whitney p-values), use `magnitude_cascade_test_with_stats`.
    """
    df = penetration_df[
        (penetration_df["pathway"] == pathway)
        & (penetration_df["primary_stim"] == B_downstream)
        & (penetration_df["A"] == A_upstream)
    ].copy()
    if df.empty:
        return df
    df["pass_ordering"] = (
        (df["mean_score_A"] > df["mean_score_B"])
        & (df["mean_score_B"] > df["mean_score_PBS"])
    )
    return df[["cell_type", "mean_score_A", "mean_score_B", "mean_score_PBS", "pass_ordering"]]


def magnitude_cascade_test_with_stats(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    pathway: str,
    pathway_idx: np.ndarray,
    A_upstream: str,
    B_downstream: str,
    pbs_label: str = "PBS",
    min_cells: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Full per-cell magnitude cascade test with Mann-Whitney inference.

    For each cell type T:
        H1a: mean(s_P, A-tube) > mean(s_P, B-tube)   (cascade adds magnitude)
        H1b: mean(s_P, B-tube) > mean(s_P, PBS)      (sanity: pathway responds)

    Reports one-sided Mann-Whitney U p-values for both contrasts. Pass requires
    both orderings to hold AND both p-values to be below alpha.
    """
    cell_types = sorted({ct for (_, ct) in cells_by_pair.keys()})
    rows = []
    for T in cell_types:
        keys = [(A_upstream, T), (B_downstream, T), (pbs_label, T)]
        if any(k not in cells_by_pair for k in keys):
            continue
        cA = cells_by_pair[(A_upstream, T)]
        cB = cells_by_pair[(B_downstream, T)]
        cP = cells_by_pair[(pbs_label, T)]
        if len(cA) < min_cells or len(cB) < min_cells or len(cP) < min_cells:
            continue
        sA = cA[:, pathway_idx].mean(axis=1)
        sB = cB[:, pathway_idx].mean(axis=1)
        sP = cP[:, pathway_idx].mean(axis=1)

        # One-sided Mann-Whitney U: H1 sA > sB
        u_AB, p_AB = scstats.mannwhitneyu(sA, sB, alternative="greater")
        u_BP, p_BP = scstats.mannwhitneyu(sB, sP, alternative="greater")

        rows.append({
            "pathway": pathway,
            "A_upstream": A_upstream,
            "B_downstream": B_downstream,
            "cell_type": T,
            "mean_score_A": float(sA.mean()),
            "mean_score_B": float(sB.mean()),
            "mean_score_PBS": float(sP.mean()),
            "pass_ordering": bool(sA.mean() > sB.mean() > sP.mean()),
            "p_A_gt_B": float(p_AB),
            "p_B_gt_PBS": float(p_BP),
            "pass_significant_alpha": bool(
                (sA.mean() > sB.mean() > sP.mean())
                and p_AB < alpha
                and p_BP < alpha
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pre-registered test battery
# ---------------------------------------------------------------------------

# Pre-registered cascade tests. Each entry is one (test_kind, ...) tuple:
#   ("binary",    pathway, primary_stim, pos_stims, neg_stims)
#   ("magnitude", pathway, A_upstream,   B_downstream)
PREREG_CASCADE_TESTS = [
    ("binary",    "IFNAR_induced", "IFNb",
                  IFNAR_POSITIVE_STIMULI, IFNAR_NEGATIVE_STIMULI),
    ("magnitude", "NFkB_canonical", "LPS",    "TNF"),
    ("magnitude", "NFkB_canonical", "LPSlo",  "TNF"),
    ("magnitude", "NFkB_canonical", "P3CSK",  "TNF"),
    ("magnitude", "NFkB_canonical", "CpG",    "TNF"),
]


def run_preregistered_battery(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    resolved_pathways: Dict[str, Dict],
    penetration_df: pd.DataFrame,
    alpha: float = 0.05,
    min_cells: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full pre-registered cascade-test battery.

    Returns a dict with keys:
        binary_per_celltype:   AUC per cell type for the IFNAR binary test
        magnitude_per_test:    long DataFrame across all magnitude tests, per cell type
        summary:               one row per (test_name, cell_type) with pass + p
        bonferroni:            Bonferroni-corrected pass-flag at α/N_tests

    `N_tests` for Bonferroni is the number of pre-registered tests
    (currently 5: 1 binary + 4 magnitude). Per-cell-type tests within a single
    pre-registered test do NOT each count separately — the cell-type with
    strongest effect carries the test (we report the best cell-type p-value).
    """
    n_tests = len(PREREG_CASCADE_TESTS)
    alpha_bonf = alpha / n_tests

    rows = []
    magnitude_dfs = []

    for entry in PREREG_CASCADE_TESTS:
        if entry[0] == "binary":
            _, pathway, primary, pos, neg = entry
            test_name = f"binary:{pathway}->{primary}"
            df_b = ifnar_binary_test(
                penetration_df, pathway=pathway, primary_stim=primary,
                pos_stimuli=pos, neg_stimuli=neg,
            )
            best_p = float("nan")
            best_T = None
            best_auc = float("nan")
            sep_clean_any = False
            for _, r in df_b.iterrows():
                # Empirical p for AUC=auc with n_pos vs n_neg:
                # exact one-sided test = sum_{k>=hits} C(n_pos,k)*C(n_neg,n_pos-k) / C(n_pos+n_neg,n_pos)
                # but we approximate: use scstats.mannwhitneyu on penetration values.
                sub = penetration_df[
                    (penetration_df["pathway"] == pathway)
                    & (penetration_df["primary_stim"] == primary)
                    & (penetration_df["cell_type"] == r["cell_type"])
                ]
                pen_by_stim = dict(zip(sub["A"], sub["penetration"]))
                pos_v = np.array([pen_by_stim[s] for s in pos if s in pen_by_stim and not np.isnan(pen_by_stim[s])])
                neg_v = np.array([pen_by_stim[s] for s in neg if s in pen_by_stim and not np.isnan(pen_by_stim[s])])
                if len(pos_v) >= 1 and len(neg_v) >= 1:
                    try:
                        _, p_emp = scstats.mannwhitneyu(pos_v, neg_v, alternative="greater")
                        p_emp = float(p_emp)
                    except ValueError:
                        p_emp = float("nan")
                else:
                    p_emp = float("nan")
                if r["sep_clean"]:
                    sep_clean_any = True
                # Track best (lowest p) cell type
                if np.isnan(best_p) or (not np.isnan(p_emp) and p_emp < best_p):
                    best_p = p_emp
                    best_T = r["cell_type"]
                    best_auc = float(r["auc"])

            rows.append({
                "test_name": test_name,
                "kind": "binary",
                "best_cell_type": best_T,
                "best_auc_or_means": best_auc,
                "best_p": best_p,
                "pass_alpha": (not np.isnan(best_p)) and best_p < alpha,
                "pass_bonferroni": (not np.isnan(best_p)) and best_p < alpha_bonf,
                "sep_clean_any_cell_type": sep_clean_any,
            })
        elif entry[0] == "magnitude":
            _, pathway, A_up, B_dn = entry
            test_name = f"magnitude:{pathway}:{A_up}>{B_dn}>PBS"
            pw = resolved_pathways.get(pathway)
            if not pw or not pw["ok"]:
                continue
            df_m = magnitude_cascade_test_with_stats(
                cells_by_pair, pathway, pw["idx"], A_up, B_dn,
                alpha=alpha, min_cells=min_cells,
            )
            magnitude_dfs.append(df_m)
            if df_m.empty:
                continue
            best_p = df_m["p_A_gt_B"].min()
            best_T = df_m.loc[df_m["p_A_gt_B"].idxmin(), "cell_type"]
            best_pass = bool(df_m["pass_significant_alpha"].any())
            rows.append({
                "test_name": test_name,
                "kind": "magnitude",
                "best_cell_type": best_T,
                "best_auc_or_means": float(df_m.loc[df_m["p_A_gt_B"].idxmin(), "mean_score_A"]),
                "best_p": float(best_p),
                "pass_alpha": (not np.isnan(best_p)) and best_p < alpha,
                "pass_bonferroni": (not np.isnan(best_p)) and best_p < alpha_bonf,
                "sep_clean_any_cell_type": best_pass,
            })

    summary = pd.DataFrame(rows)
    summary["alpha"] = alpha
    summary["alpha_bonferroni"] = alpha_bonf
    summary["n_preregistered_tests"] = n_tests

    magnitude_per_test = (
        pd.concat(magnitude_dfs, ignore_index=True) if magnitude_dfs else pd.DataFrame()
    )

    return {
        "summary": summary,
        "magnitude_per_test": magnitude_per_test,
    }
