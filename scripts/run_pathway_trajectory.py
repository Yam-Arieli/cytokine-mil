#!/usr/bin/env python3
"""
Time-trajectory pathway analysis on Sheu (§23 extension).

Runs the cascade-penetration computation on multiple Sheu pseudotube
manifests, each tagged with a time-point label, and combines the outputs
into a single long DataFrame indexed by (time_point, pathway, primary_stim,
A, cell_type). Generates trajectory plots.

Pre-registered prediction for the IFNAR cascade:
  penetration(PIC → IFNAR_induced → IFNb)  ramps UP over time
  penetration(LPS → IFNAR_induced → IFNb)  ramps UP over time, possibly later
  penetration(P3CSK/CpG/TNF → IFNAR_induced → IFNb)  stays NEAR ZERO

For the NF-κB autocrine cascade:
  gap between s_NFkB(A_upstream) and s_NFkB(TNF) should GROW over time
  for A in {LPS, LPSlo, P3CSK, CpG}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.eda_pair_benchmark import load_phase1_cells
from cytokine_mil.analysis.pathway_signatures import (
    PATHWAY_SIGNATURES,
    IFNAR_POSITIVE_STIMULI,
    IFNAR_NEGATIVE_STIMULI,
    resolve_all_pathways,
    compute_all_penetrations,
)


# Default trajectory: (time_label, hours_numeric, manifest_path)
# Hours are numeric values used for x-axis ordering; labels match Sheu's "timept" strings.
DEFAULT_TRAJECTORY = [
    ("1hr", 1.0, "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_1hr_pseudotubes/manifest.json"),
    ("3hr", 3.0, "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json"),
]


def parse_trajectory_arg(arg_list: List[str]) -> List[Tuple[str, float, str]]:
    """
    Parse `--trajectory label:hours:manifest_path` repeated args.
    Example: --trajectory "1hr:1.0:/path/to/manifest1.json" "3hr:3.0:/path/to/manifest2.json"
    """
    parsed = []
    for s in arg_list:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"--trajectory expects 'label:hours:path', got {s!r}")
        parsed.append((parts[0], float(parts[1]), parts[2]))
    return parsed


def run_trajectory(
    trajectory: List[Tuple[str, float, str]],
    out_dir: Path,
    min_cells: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """For each (time_label, hours, manifest_path), run compute_all_penetrations
    and tag the rows with time_point + hours. Return concatenated long DataFrame."""
    all_dfs = []
    for time_label, hours, manifest_path in trajectory:
        print(f"\n[traj] === {time_label} ({hours} hr) — {manifest_path} ===", flush=True)
        cells_by_pair, gene_names = load_phase1_cells(manifest_path)
        print(f"[traj] {len(cells_by_pair)} (cyt,ct) buckets, {len(gene_names)} genes", flush=True)
        if not cells_by_pair:
            print(f"[traj] WARNING: no cells at {time_label}, skipping", flush=True)
            continue
        pen, _ = compute_all_penetrations(
            cells_by_pair, gene_names,
            n_ctrl=20, min_cells=min_cells, seed=seed,
        )
        pen["time_point"] = time_label
        pen["hours"] = hours
        all_dfs.append(pen)
        print(f"[traj] {len(pen)} rows at {time_label}", flush=True)

    if not all_dfs:
        return pd.DataFrame()
    out = pd.concat(all_dfs, ignore_index=True)
    out.to_parquet(out_dir / "penetration_trajectory.parquet", index=False)
    print(f"\n[traj] wrote {len(out)} rows -> penetration_trajectory.parquet", flush=True)
    return out


def plot_penetration_trajectory(
    df: pd.DataFrame,
    pathway: str,
    primary_stim: str,
    save_path: str,
    pos_stimuli: List[str] = None,
    neg_stimuli: List[str] = None,
) -> None:
    """
    Trajectory plot: x = time, y = penetration, one curve per stimulus A,
    faceted by cell_type. Cascade positives in red, negatives in blue.
    """
    sub = df[(df["pathway"] == pathway) & (df["primary_stim"] == primary_stim)].copy()
    if sub.empty:
        return
    if pos_stimuli is None:
        pos_stimuli = []
    if neg_stimuli is None:
        neg_stimuli = []

    cell_types = sorted(sub["cell_type"].unique())
    n = len(cell_types)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), sharey=True)
    if n == 1:
        axes = [axes]

    sub = sub.sort_values("hours")

    for ax, T in zip(axes, cell_types):
        sub_T = sub[sub["cell_type"] == T]
        for A in sorted(sub_T["A"].unique()):
            xy = sub_T[sub_T["A"] == A].sort_values("hours")
            if A == primary_stim:
                color, ls, lw = "black", "--", 1.5
            elif A in pos_stimuli:
                color, ls, lw = "#D62728", "-", 1.8  # red — cascade positive
            elif A in neg_stimuli:
                color, ls, lw = "#1F77B4", "-", 1.8  # blue — cascade negative
            elif A == "PBS":
                color, ls, lw = "#999999", ":", 1.2
            else:
                color, ls, lw = "#CCCCCC", "-", 1.0
            ax.plot(xy["hours"], xy["penetration"], color=color, linestyle=ls, linewidth=lw,
                    marker="o", markersize=5, label=A)
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.axhline(1, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
        ax.set_xlabel("hours")
        ax.set_title(T, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(f"penetration of {pathway} → {primary_stim}")

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=1)
    fig.suptitle(
        f"Cascade penetration over time: {pathway} (primary = {primary_stim})",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_nfkb_magnitude_gap(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    """For NFkB_canonical primary=TNF, plot mean_score over time per stimulus.
    Cascade prediction: gap between LPS/LPSlo/P3CSK/CpG and TNF grows over time."""
    sub = df[
        (df["pathway"] == "NFkB_canonical")
        & (df["primary_stim"] == "TNF")
    ].copy()
    if sub.empty:
        return
    cell_types = sorted(sub["cell_type"].unique())
    n = len(cell_types)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), sharey=True)
    if n == 1:
        axes = [axes]
    sub = sub.sort_values("hours")
    upstream_pos = ["LPS", "LPSlo", "P3CSK", "CpG"]
    for ax, T in zip(axes, cell_types):
        sT = sub[sub["cell_type"] == T]
        # For each A, take mean_score_A over time, plot
        # Note: penetration_long stores per-A row with mean_score_A. We use that.
        for A in sorted(sT["A"].unique()):
            xy = sT[sT["A"] == A].sort_values("hours")
            if A == "TNF":
                color, ls, lw = "black", "--", 2.0
            elif A in upstream_pos:
                color, ls, lw = "#D62728", "-", 1.8
            elif A == "PBS":
                color, ls, lw = "#999999", ":", 1.2
            else:
                color, ls, lw = "#CCCCCC", "-", 1.0
            ax.plot(xy["hours"], xy["mean_score_A"], color=color, linestyle=ls, linewidth=lw,
                    marker="o", markersize=5, label=A)
        ax.set_xlabel("hours")
        ax.set_title(T, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("mean s_NFkB(A)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle("NFkB magnitude over time — cascade predicts gap upstream > TNF grows", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory", nargs="+", default=None,
        help="List of 'label:hours:manifest_path' triples. Defaults to (1hr,3hr).",
    )
    parser.add_argument(
        "--out_dir",
        default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_pathway_trajectory",
    )
    parser.add_argument("--min_cells", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.trajectory:
        trajectory = parse_trajectory_arg(args.trajectory)
    else:
        trajectory = DEFAULT_TRAJECTORY
    print(f"[traj] {len(trajectory)} time points: "
          f"{[(t, h) for t, h, _ in trajectory]}", flush=True)

    df = run_trajectory(trajectory, out_dir, min_cells=args.min_cells, seed=args.seed)
    if df.empty:
        print("[traj] no data — aborting")
        return

    print("\n[plot] IFNAR cascade trajectory", flush=True)
    plot_penetration_trajectory(
        df, "IFNAR_induced", "IFNb",
        save_path=str(plots_dir / "trajectory_IFNAR_induced.pdf"),
        pos_stimuli=list(IFNAR_POSITIVE_STIMULI),
        neg_stimuli=list(IFNAR_NEGATIVE_STIMULI),
    )

    print("[plot] IRF3 cascade trajectory", flush=True)
    plot_penetration_trajectory(
        df, "IRF3_direct", "PIC",
        save_path=str(plots_dir / "trajectory_IRF3_direct.pdf"),
        pos_stimuli=["PIC", "LPS"],
        neg_stimuli=["P3CSK", "CpG", "TNF"],
    )

    print("[plot] NFkB magnitude over time", flush=True)
    plot_nfkb_magnitude_gap(
        df, save_path=str(plots_dir / "trajectory_NFkB_magnitude.pdf"),
    )

    # Also try TNFR_autocrine
    print("[plot] TNFR_autocrine trajectory", flush=True)
    plot_penetration_trajectory(
        df, "TNFR_autocrine", "TNF",
        save_path=str(plots_dir / "trajectory_TNFR_autocrine.pdf"),
        pos_stimuli=["LPS", "LPSlo", "P3CSK", "CpG"],
        neg_stimuli=["IFNb", "PIC"],
    )

    print(f"\n[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
