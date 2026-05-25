"""
Plotting helpers for the pathway-signature cascade analysis (§23).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cytokine_mil.analysis.pathway_signatures import (
    PATHWAY_SIGNATURES,
    IFNAR_POSITIVE_STIMULI,
    IFNAR_NEGATIVE_STIMULI,
    compute_pathway_score,
)


# ---------------------------------------------------------------------------
# Penetration heatmap
# ---------------------------------------------------------------------------

def plot_penetration_heatmap(
    penetration_df: pd.DataFrame,
    save_path: str,
    cell_types: Optional[List[str]] = None,
) -> None:
    """
    Heatmap of penetration values: rows = (pathway, primary_stim) pairs,
    cols = stimulus A, faceted by cell type.

    Color: penetration; clipped to [-0.5, 1.5] for visibility.
    """
    if cell_types is None:
        cell_types = sorted(penetration_df["cell_type"].unique())
    n_panels = len(cell_types)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.0), sharey=True)
    if n_panels == 1:
        axes = [axes]

    # Stable col order across panels
    all_A = sorted(penetration_df["A"].unique())
    # Stable row order: pathway then primary_stim
    pairs = sorted({(r["pathway"], r["primary_stim"]) for _, r in penetration_df.iterrows()})

    for ax, T in zip(axes, cell_types):
        sub = penetration_df[penetration_df["cell_type"] == T]
        M = np.full((len(pairs), len(all_A)), np.nan)
        for i, (pw, prim) in enumerate(pairs):
            for j, A in enumerate(all_A):
                v = sub[(sub["pathway"] == pw) & (sub["primary_stim"] == prim) & (sub["A"] == A)]
                if not v.empty:
                    M[i, j] = float(v["penetration"].iloc[0])
        M_clip = np.clip(M, -0.5, 1.5)
        im = ax.imshow(M_clip, cmap="RdBu_r", vmin=-0.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(all_A)))
        ax.set_xticklabels(all_A, rotation=45, ha="right", fontsize=8)
        if ax is axes[0]:
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels([f"{p[0]} → {p[1]}" for p in pairs], fontsize=9)
        ax.set_title(T, fontsize=10)
    fig.colorbar(im, ax=axes, label="penetration", shrink=0.8)
    fig.suptitle("Cascade penetration = (s_A − s_PBS) / (s_primary − s_PBS)", fontsize=11)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-pathway ridge / strip
# ---------------------------------------------------------------------------

def plot_pathway_score_strip(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    pathway: str,
    pathway_idx: np.ndarray,
    control_idx: np.ndarray,
    save_path: str,
    stim_order: Optional[List[str]] = None,
    pbs_label: str = "PBS",
) -> None:
    """
    Strip + violin plot of per-cell pathway scores, one row per cell type,
    one violin per stimulus.

    The plot to inspect directly: does the curated pathway separate stimuli
    that should engage it from stimuli that should not?
    """
    cell_types = sorted({ct for (_, ct) in cells_by_pair.keys()})
    if stim_order is None:
        stim_order = sorted({cyt for (cyt, _) in cells_by_pair.keys()})

    n_ct = len(cell_types)
    fig, axes = plt.subplots(n_ct, 1, figsize=(max(7, 0.7 * len(stim_order) + 2), 2.4 * n_ct),
                             sharex=True)
    if n_ct == 1:
        axes = [axes]

    primary = PATHWAY_SIGNATURES[pathway]["primary_for"]
    cascade = PATHWAY_SIGNATURES[pathway]["cascade_from"]

    def _color(s):
        if s == pbs_label:
            return "#999999"
        if s in primary:
            return "#1F77B4"   # blue: pathway primary
        if s in cascade:
            return "#D62728"   # red: predicted cascade-producer
        return "#CCCCCC"       # gray: should be low

    for ax, T in zip(axes, cell_types):
        data, colors = [], []
        for s in stim_order:
            if (s, T) not in cells_by_pair:
                data.append(np.array([]))
                colors.append("#FFFFFF")
                continue
            cells = cells_by_pair[(s, T)]
            scores = compute_pathway_score(cells, pathway_idx, control_idx)
            data.append(scores)
            colors.append(_color(s))
        positions = np.arange(len(stim_order))
        # Violin
        for pos, d, c in zip(positions, data, colors):
            if len(d) < 3:
                continue
            parts = ax.violinplot([d], positions=[pos], widths=0.7, showmeans=False,
                                  showmedians=True, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(c)
                body.set_alpha(0.65)
                body.set_edgecolor("black")
                body.set_linewidth(0.5)
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(stim_order, rotation=0)
        ax.set_ylabel(f"{T}\ns_{pathway}", fontsize=9)

    fig.suptitle(
        f"{pathway}  —  primary: {', '.join(primary) or '—'}  |  "
        f"cascade-from: {', '.join(cascade) or '—'}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Binary test summary
# ---------------------------------------------------------------------------

def plot_ifnar_binary_summary(
    binary_df: pd.DataFrame,
    penetration_df: pd.DataFrame,
    save_path: str,
    primary_stim: str = "IFNb",
    pathway: str = "IFNAR_induced",
) -> None:
    """
    Per-cell-type strip plot of penetration values for IFNAR_induced toward IFNb.
    Pre-registered positives (PIC, LPS, LPSlo, IFNb) annotated as positive,
    negatives (P3CSK, CpG, TNF) as negative. AUC per cell type printed.
    """
    sub = penetration_df[
        (penetration_df["pathway"] == pathway)
        & (penetration_df["primary_stim"] == primary_stim)
    ]
    cell_types = sorted(sub["cell_type"].unique())
    if not cell_types:
        return

    n = len(cell_types)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    stim_order = list(IFNAR_POSITIVE_STIMULI) + list(IFNAR_NEGATIVE_STIMULI)
    label_of = {s: ("pos" if s in IFNAR_POSITIVE_STIMULI else "neg") for s in stim_order}
    color_of = {s: ("#1F77B4" if s in IFNAR_POSITIVE_STIMULI else "#D62728") for s in stim_order}

    for ax, T in zip(axes, cell_types):
        sub_T = sub[sub["cell_type"] == T]
        pen_by_stim = dict(zip(sub_T["A"], sub_T["penetration"]))
        xs, ys, cs, labels = [], [], [], []
        for i, s in enumerate(stim_order):
            v = pen_by_stim.get(s, np.nan)
            xs.append(i); ys.append(v); cs.append(color_of[s]); labels.append(s)
        ax.scatter(xs, ys, c=cs, s=120, edgecolor="black", linewidth=0.8, zorder=3)
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.axhline(1, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(-0.5, 1.8)
        row = binary_df[binary_df["cell_type"] == T]
        if not row.empty:
            auc = float(row["auc"].iloc[0])
            clean = bool(row["sep_clean"].iloc[0])
            ax.set_title(f"{T}  AUC={auc:.2f}  clean={clean}", fontsize=10)
        else:
            ax.set_title(T, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(f"penetration of {pathway} (→ {primary_stim})")

    fig.suptitle(
        f"Pre-registered binary test: {pathway} positives "
        f"({', '.join(IFNAR_POSITIVE_STIMULI)}) vs negatives "
        f"({', '.join(IFNAR_NEGATIVE_STIMULI)})",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
