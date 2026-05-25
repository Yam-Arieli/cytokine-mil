"""
Plotting helpers for the pair-level EDA benchmark.

All plots are designed to surface cascade asymmetry visually, on top of the
numeric battery computed in `eda_pair_benchmark.py`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cytokine_mil.analysis.eda_pair_benchmark import (
    _log2fc,
    _signature_score,
    NAME_ALIAS,
)


def _display_name(stim: str) -> str:
    return NAME_ALIAS.get(stim, stim)


def _zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return x * 0.0
    return (x - x.mean()) / sd


# ---------------------------------------------------------------- heatmap

def plot_statistic_heatmap(
    summary_df: pd.DataFrame,
    save_path: str,
    statistic_value_col: str = "max",
    statistics: Optional[List[str]] = None,
) -> None:
    """
    Heatmap of statistics (cols) × labeled pairs (rows).

    Rows ordered: positives on top, negatives at bottom, separator in between.
    Color = column-wise z-score of the chosen aggregator.
    """
    df = summary_df.dropna(subset=["pair_label"]).copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No labeled pairs present — skipping heatmap",
                ha="center", va="center")
        ax.set_axis_off()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    if statistics is None:
        statistics = sorted(df["statistic"].unique())
    df = df[df["statistic"].isin(statistics)]

    table = (
        df.pivot_table(
            index=["unordered_pair", "pair_label"],
            columns="statistic",
            values=statistic_value_col,
            aggfunc="first",
        )
        .reset_index()
    )
    table["sort_key"] = (table["pair_label"] != "positive").astype(int)
    table = table.sort_values(["sort_key", "unordered_pair"]).reset_index(drop=True)

    z = table[statistics].apply(_zscore, axis=0).values.astype(float)
    pair_labels = table["unordered_pair"].values
    row_label = table["pair_label"].values

    fig_h = max(2.4, 0.42 * len(pair_labels) + 1.5)
    fig_w = max(7.0, 0.55 * len(statistics) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(statistics)))
    ax.set_xticklabels(statistics, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(
        [f"[{lab[0].upper()}] {pp}" for pp, lab in zip(pair_labels, row_label)],
        fontsize=9,
    )
    split = int(np.sum(row_label == "positive"))
    if 0 < split < len(pair_labels):
        ax.axhline(split - 0.5, color="black", linewidth=1.5)

    plt.colorbar(im, ax=ax, label=f"column z-score of {statistic_value_col}")
    ax.set_title("Labeled pairs (P=positive on top, N=negative below) × statistics")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- AUC bars

def plot_auc_bars(
    auc_df: pd.DataFrame,
    null_df: pd.DataFrame,
    save_path: str,
    p_threshold: float = 0.05,
) -> None:
    """
    Bar chart of per-statistic AUC, with permutation-null upper quantile overlaid.
    """
    if auc_df is None or auc_df.empty or "auc" not in auc_df.columns:
        # Nothing labeled — write an empty placeholder so the path exists.
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No labeled pairs present — skipping AUC bars",
                ha="center", va="center")
        ax.set_axis_off()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    auc_df = auc_df.copy().sort_values("auc", ascending=True).reset_index(drop=True)
    q = 1.0 - p_threshold
    null_q = (
        null_df.groupby("statistic")["auc"].quantile(q).rename(f"null_q{q:.2f}").reset_index()
    )
    auc_df = auc_df.merge(null_q, on="statistic", how="left")

    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.32 * len(auc_df))))
    y = np.arange(len(auc_df))
    colors = ["#4C72B0" if a > nq else "#BBBBBB"
              for a, nq in zip(auc_df["auc"], auc_df[f"null_q{q:.2f}"])]
    ax.barh(y, auc_df["auc"], color=colors)
    ax.scatter(
        auc_df[f"null_q{q:.2f}"], y,
        marker="|", color="red", s=140, zorder=3,
        label=f"null {q:.2f} quantile",
    )
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--", label="chance")
    ax.set_yticks(y)
    ax.set_yticklabels(auc_df["statistic"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC (positives ranked above negatives)")
    ax.set_title(
        f"Per-statistic discrimination on n={int(auc_df['n_positive'].iloc[0])}+"
        f"{int(auc_df['n_negative'].iloc[0])} labeled pairs"
    )
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- signature scatter

def plot_signature_scatter(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    A: str,
    B: str,
    save_path: str,
    pbs_label: str = "PBS",
    n_sig: int = 20,
    max_cells_per_class: int = 1500,
    pair_label: Optional[str] = None,
) -> None:
    """
    Scatter (s_A, s_B) for cells in A-tube, B-tube, and PBS. Faceted by cell type.

    s_X = mean expression of top-n up-DE genes of X-vs-PBS minus mean of control genes.

    Visually: cascade A→B should show A-tube cells extending up the s_B axis
    relative to PBS. B-tube cells extending in s_A axis would suggest reverse.
    """
    cell_types = sorted({
        ct for (cyt, ct) in cells_by_pair.keys()
        if cyt in (A, B, pbs_label)
    })
    cell_types = [
        ct for ct in cell_types
        if (A, ct) in cells_by_pair
        and (B, ct) in cells_by_pair
        and (pbs_label, ct) in cells_by_pair
    ]
    if not cell_types:
        return

    rng = np.random.default_rng(0)
    n_panels = len(cell_types)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.6 * n_panels, 3.6), sharex=False, sharey=False
    )
    if n_panels == 1:
        axes = [axes]

    A_disp, B_disp = _display_name(A), _display_name(B)

    for ax, T in zip(axes, cell_types):
        cA = cells_by_pair[(A, T)]
        cB = cells_by_pair[(B, T)]
        cP = cells_by_pair[(pbs_label, T)]

        log2fc_A = _log2fc(cA, cP)
        log2fc_B = _log2fc(cB, cP)
        sig_A = np.argsort(log2fc_A)[-n_sig:]
        sig_B = np.argsort(log2fc_B)[-n_sig:]
        G = cA.shape[1]
        used = set(sig_A.tolist()) | set(sig_B.tolist())
        pool = np.array([i for i in range(G) if i not in used])
        if len(pool) < 4:
            ctrl = np.arange(min(n_sig, G))
        else:
            ctrl = rng.choice(pool, size=min(n_sig, len(pool)), replace=False)

        sA_A = _signature_score(cA, sig_A, ctrl)
        sB_A = _signature_score(cA, sig_B, ctrl)
        sA_B = _signature_score(cB, sig_A, ctrl)
        sB_B = _signature_score(cB, sig_B, ctrl)
        sA_P = _signature_score(cP, sig_A, ctrl)
        sB_P = _signature_score(cP, sig_B, ctrl)

        def _sub(x, y, n):
            if len(x) <= n:
                return x, y
            idx = rng.choice(len(x), n, replace=False)
            return x[idx], y[idx]

        sA_P_s, sB_P_s = _sub(sA_P, sB_P, max_cells_per_class)
        sA_A_s, sB_A_s = _sub(sA_A, sB_A, max_cells_per_class)
        sA_B_s, sB_B_s = _sub(sA_B, sB_B, max_cells_per_class)

        ax.scatter(sA_P_s, sB_P_s, s=4, alpha=0.30, c="#999999", label="PBS")
        ax.scatter(sA_A_s, sB_A_s, s=4, alpha=0.45, c="#D62728", label=f"{A_disp}-tube")
        ax.scatter(sA_B_s, sB_B_s, s=4, alpha=0.45, c="#2CA02C", label=f"{B_disp}-tube")

        ax.axhline(0, c="black", linewidth=0.5, linestyle="--")
        ax.axvline(0, c="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel(f"s_{A_disp}")
        ax.set_ylabel(f"s_{B_disp}")
        ax.set_title(T, fontsize=10)

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    label_tag = f" [{pair_label.upper()}]" if pair_label else ""
    fig.suptitle(f"{A_disp} ↔ {B_disp}{label_tag} — signature joint distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- projection density

def plot_projection_density(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    A: str,
    B: str,
    save_path: str,
    pbs_label: str = "PBS",
    n_bins: int = 60,
    pair_label: Optional[str] = None,
) -> None:
    """
    Overlaid KDE-like histograms of cells projected onto û_{A→B}, per cell type.

    Predictive use: if A-tube on this axis is bimodal (primary + relay subpops)
    and B-tube is unimodal, A→B cascade is visible.
    """
    cell_types = sorted({
        ct for (cyt, ct) in cells_by_pair.keys()
        if cyt in (A, B, pbs_label)
    })
    cell_types = [
        ct for ct in cell_types
        if (A, ct) in cells_by_pair
        and (B, ct) in cells_by_pair
        and (pbs_label, ct) in cells_by_pair
    ]
    if not cell_types:
        return

    n_panels = len(cell_types)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.4 * n_panels, 3.0), sharex=False, sharey=False
    )
    if n_panels == 1:
        axes = [axes]

    A_disp, B_disp = _display_name(A), _display_name(B)

    for ax, T in zip(axes, cell_types):
        cA = cells_by_pair[(A, T)]
        cB = cells_by_pair[(B, T)]
        cP = cells_by_pair[(pbs_label, T)]

        mu_A = cA.mean(axis=0)
        mu_B = cB.mean(axis=0)
        direction = mu_B - mu_A
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            ax.set_axis_off()
            continue
        direction = direction / norm

        proj_A = cA @ direction
        proj_B = cB @ direction
        proj_P = cP @ direction

        lo = float(min(proj_A.min(), proj_B.min(), proj_P.min()))
        hi = float(max(proj_A.max(), proj_B.max(), proj_P.max()))
        bins = np.linspace(lo, hi, n_bins)

        ax.hist(proj_P, bins=bins, density=True, alpha=0.30, color="#999999", label="PBS")
        ax.hist(proj_A, bins=bins, density=True, alpha=0.50, color="#D62728", label=f"{A_disp}")
        ax.hist(proj_B, bins=bins, density=True, alpha=0.50, color="#2CA02C", label=f"{B_disp}")
        ax.set_xlabel(f"projection on û_({A_disp}→{B_disp})")
        ax.set_ylabel("density")
        ax.set_title(T, fontsize=10)

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    label_tag = f" [{pair_label.upper()}]" if pair_label else ""
    fig.suptitle(f"{A_disp} ↔ {B_disp}{label_tag} — projection density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
