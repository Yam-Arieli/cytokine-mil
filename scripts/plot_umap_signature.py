#!/usr/bin/env python
"""
Render the 3-panel signature-UMAP figure from `umap_coords.parquet` + `stats.json`
written by `umap_signature_embedding.py`.

Panels (left to right): all_hvg, signature, random -- same cells, same PCA/UMAP
hyperparameters, only the gene subset differs. Each panel is coloured by
condition (cytokine vs PBS) and titled with its silhouette score (condition
labels, PCA space) -- the quantitative pairing to the visual.

Message: if `signature` shows visibly better condition separation than BOTH
`all_hvg` and `random` (and a higher silhouette score), that is non-circular
evidence the discovered signature carries condition-specific structure beyond
its size and beyond the full gene panel -- without needing a ground-truth
label for "is this the correct signature".

Standalone and RE-RUNNABLE: reads precomputed coordinates so the figure can be
retuned (colours, dpi) without recomputing the embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless cluster
import matplotlib.pyplot as plt
import pandas as pd

# Mirror the project reporting style (scripts/plot_umap_donor.py).
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    }
)

S = 5
ALPHA = 0.4
VARIANT_TITLES = {
    "all_hvg": "all HVGs",
    "signature": r"signature $S_X$",
    "random": "random genes (same size)",
}
CONDITION_COLOURS = {"PBS": "#9e9e9e", "cytokine": "#d1495b"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coords_path", default="results/umap_signature/umap_coords.parquet")
    p.add_argument("--stats_path", default="results/umap_signature/stats.json")
    p.add_argument("--out_dir", default="results/umap_signature")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def _scatter_by_condition(ax, df: pd.DataFrame, add_legend: bool) -> None:
    for cond in ["PBS", "cytokine"]:
        sub = df[df["condition"] == cond]
        ax.scatter(
            sub["umap_x"].to_numpy(),
            sub["umap_y"].to_numpy(),
            s=S,
            alpha=ALPHA,
            c=CONDITION_COLOURS[cond],
            linewidths=0,
            label=cond if add_legend else None,
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
    with open(args.stats_path) as fh:
        stats = json.load(fh)
    cytokine = stats["cytokine"]

    variants = ["all_hvg", "signature", "random"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))

    for i, variant in enumerate(variants):
        ax = axes[i]
        sub = df[df["variant"] == variant]
        _scatter_by_condition(ax, sub, add_legend=(i == 0))
        sil = stats["variants"][variant]["silhouette"]
        n_genes = stats["variants"][variant]["n_genes"]
        ax.set_title(f"{VARIANT_TITLES[variant]} ({n_genes} genes)\nsilhouette = {sil:+.3f}")

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        fontsize=9,
        markerscale=3.0,
        frameon=False,
    )
    fig.suptitle(f"{cytokine} vs PBS -- same cells, gene subset varies by panel", y=1.02)
    fig.tight_layout()

    png_path = out_dir / "umap_signature.png"
    pdf_path = out_dir / "umap_signature.pdf"
    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
