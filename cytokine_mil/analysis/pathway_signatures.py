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
        "up": ["Isg15", "Mx1", "Mx2", "Oas1a", "Oas2", "Oas3",
               "Ifit1", "Rsad2", "Stat1", "Irf7", "Usp18"],
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
        "up": ["Tnfaip3", "Nfkbid", "Birc3"],
        "primary_for": ["TNF"],
        "cascade_from": ["LPS", "LPSlo", "P3CSK", "CpG"],
        "rationale": "Subset of NF-κB targets classically induced via TNFR1→TRADD "
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
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Map a curated pathway to integer column indices in the loaded data.

    Returns:
        idx: int array of column indices present in gene_names
        found: list of gene symbols that were resolved
        missing: list of gene symbols not in gene_names
    Raises:
        ValueError if fewer than `min_hits` curated genes are present.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    found, missing = [], []
    idx_list = []
    for g in PATHWAY_SIGNATURES[pathway]["up"]:
        if g in gene_to_idx:
            found.append(g)
            idx_list.append(gene_to_idx[g])
        else:
            missing.append(g)
    if len(found) < min_hits:
        raise ValueError(
            f"Pathway {pathway}: only {len(found)} of "
            f"{len(PATHWAY_SIGNATURES[pathway]['up'])} curated genes "
            f"present in panel (need >= {min_hits}). Missing: {missing}"
        )
    return np.array(idx_list, dtype=np.int64), found, missing


def resolve_all_pathways(
    gene_names: Sequence[str],
    min_hits: int = 3,
) -> Dict[str, Dict]:
    """
    Resolve all pathways against a gene panel. Pathways with too few hits
    are dropped with a warning string in their entry.

    Returns dict {pathway -> {idx, found, missing, ok}}.
    """
    resolved = {}
    for p in PATHWAY_SIGNATURES:
        try:
            idx, found, missing = resolve_pathway_genes(p, gene_names, min_hits)
            resolved[p] = {"idx": idx, "found": found, "missing": missing, "ok": True}
        except ValueError as e:
            resolved[p] = {"idx": None, "found": [], "missing": list(PATHWAY_SIGNATURES[p]["up"]),
                           "ok": False, "reason": str(e)}
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
    show HIGHER pathway activation than direct-B (because A also produces
    cascade autocrine on top of A-primary).

    Test per cell type: is mean_score_A > mean_score_B > mean_score_PBS?
    Returns DataFrame with columns: cell_type, score_A, score_B, score_PBS, pass.
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
