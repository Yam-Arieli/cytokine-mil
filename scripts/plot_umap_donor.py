#!/usr/bin/env python
"""
Render the two-panel donor UMAP figure from `umap_coords.parquet`.

Standalone and RE-RUNNABLE: reads the precomputed coordinates written by
`umap_donor_embedding.py`, so the figure can be retuned (colours, cap, dpi)
without recomputing the embedding.

Message: LEFT panel coloured by cell type shows cells cluster by cell identity;
RIGHT panel coloured by cytokine shows the SAME points as an intermixed colour
jumble -- cell identity dominates single-cell variation, cytokines do not
separate. The right panel therefore carries NO legend (the jumble is the point).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mirror the project reporting style (scripts/make_report_figures.py).
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    }
)

S = 4
ALPHA = 0.35


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coords_path",
        default="results/umap_donor1/umap_coords.parquet",
        help="Parquet written by umap_donor_embedding.py.",
    )
    p.add_argument(
        "--out_dir",
        default="results/umap_donor1",
        help="Directory to write umap.png / umap.pdf.",
    )
    p.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="Cap cells per cell_type (0 = no cap). Same rows used in both panels.",
    )
    p.add_argument("--dpi", type=int, default=220, help="Output DPI (>=200).")
    return p.parse_args()


def cap_per_class(df: pd.DataFrame, max_per_class: int, group_col: str = "cell_type") -> pd.DataFrame:
    """Subsample each `group_col` group to at most `max_per_class` rows (seeded)."""
    if max_per_class and max_per_class > 0:
        rng = np.random.default_rng(0)

        def _take(g: pd.DataFrame) -> pd.DataFrame:
            if len(g) <= max_per_class:
                return g
            idx = rng.choice(len(g), size=max_per_class, replace=False)
            return g.iloc[np.sort(idx)]

        df = df.groupby(group_col, group_keys=False).apply(_take)
    return df.reset_index(drop=True)


def _colour_map(categories: list[str], cmap_name: str):
    """Map each category (sorted) to a distinct colour sampled across `cmap_name`."""
    cmap = plt.get_cmap(cmap_name)
    n = len(categories)
    if n == 0:
        return {}
    # Sample across the full colour range so 91 cytokines span the whole gamut.
    positions = np.linspace(0.0, 1.0, n, endpoint=False)
    return {cat: cmap(pos) for cat, pos in zip(categories, positions)}


def _scatter_by_class(ax, df, colour_col, colours, add_legend: bool):
    for cat in sorted(df[colour_col].unique()):
        sub = df[df[colour_col] == cat]
        ax.scatter(
            sub["umap_x"].to_numpy(),
            sub["umap_y"].to_numpy(),
            s=S,
            alpha=ALPHA,
            c=[colours[cat]],
            linewidths=0,
            label=cat if add_legend else None,
            rasterized=True,
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.coords_path)
    df = cap_per_class(df, args.max_per_class, group_col="cell_type")

    cell_types = sorted(df["cell_type"].unique())
    cytokines = sorted(df["cytokine"].unique())
    print(
        f"Plotting {len(df)} cells | {len(cell_types)} cell types | "
        f"{len(cytokines)} cytokines"
    )

    # tab20 handles up to ~20 cell-type categories distinctly; a large qualitative
    # (rainbow) sampling gives 91 cytokines maximally-spread hues (jumble by design).
    ct_colours = _colour_map(cell_types, "tab20")
    cyt_colours = _colour_map(cytokines, "gist_rainbow")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5.2))

    _scatter_by_class(ax_l, df, "cell_type", ct_colours, add_legend=True)
    ax_l.set_title("coloured by cell type")
    # Small legend placed under the left panel so it never collides with the right.
    ax_l.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=7,
        markerscale=3.0,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    _scatter_by_class(ax_r, df, "cytokine", cyt_colours, add_legend=False)
    ax_r.set_title("coloured by cytokine")

    fig.tight_layout()

    png_path = out_dir / "umap.png"
    pdf_path = out_dir / "umap.pdf"
    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
