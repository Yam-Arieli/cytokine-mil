#!/usr/bin/env python
"""
Compute a UMAP embedding of Oesinghaus pseudo-tube cells for ONE donor.

Purpose (figure): show that single cells separate by CELL TYPE while CYTOKINES
are intermixed -- evidence that cell identity dominates single-cell variation.

Data: Oesinghaus pseudo-tubes, one donor (default Donor1), the first N pseudo-tubes
per cytokine (default 3). The non-raw `pseudotube_*.h5ad` files are ALREADY
normalized + log1p + subset to the 4000 HVGs, so they are fed straight into
PCA/UMAP (no re-normalization).

Steps:
  read manifest -> filter donor + first N tubes/cytokine -> load & concat
  -> sc.pp.pca(n_comps) -> sc.pp.neighbors -> sc.tl.umap
  -> write <out_dir>/umap_coords.parquet with columns
     [umap_x, umap_y, cell_type, cytokine] (one row per cell).

This is the compute half; `plot_umap_donor.py` renders the figure from the
parquet so it can be retuned without recomputing the embedding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest_path",
        default="/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json",
        help="Path to the Oesinghaus pseudo-tube manifest.json.",
    )
    p.add_argument("--donor", default="Donor1", help="Donor to embed.")
    p.add_argument(
        "--tubes_per_cytokine",
        type=int,
        default=3,
        help="Keep pseudo-tubes with tube_idx < this value per cytokine.",
    )
    p.add_argument(
        "--out_dir",
        default="results/umap_donor1",
        help="Output directory for umap_coords.parquet.",
    )
    p.add_argument("--n_pcs", type=int, default=50, help="Number of PCA components.")
    return p.parse_args()


def load_manifest(manifest_path: str) -> list[dict]:
    with open(manifest_path) as fh:
        return json.load(fh)


def select_entries(manifest: list[dict], donor: str, tubes_per_cytokine: int) -> list[dict]:
    """Filter to one donor and the first `tubes_per_cytokine` tubes per cytokine."""
    entries = [
        e
        for e in manifest
        if e.get("donor") == donor and int(e.get("tube_idx", 0)) < tubes_per_cytokine
    ]
    if not entries:
        raise ValueError(
            f"No manifest entries for donor={donor!r} with tube_idx < {tubes_per_cytokine}."
        )
    return entries


def load_and_concat(entries: list[dict]) -> ad.AnnData:
    """Load each entry's .h5ad, stamp the cytokine onto every cell, concat."""
    adatas = []
    for e in entries:
        a = ad.read_h5ad(e["path"])
        a.obs["cytokine"] = e["cytokine"]
        if "cell_type" not in a.obs.columns:
            raise KeyError(
                f"'cell_type' missing in obs of {e['path']}; cannot colour by cell type."
            )
        adatas.append(a)
    # var names align on the 4000 HVGs across all tubes.
    combined = ad.concat(adatas, join="inner", index_unique="-")
    return combined


def embed(adata: ad.AnnData, n_pcs: int) -> ad.AnnData:
    import scanpy as sc  # local import: scanpy is heavy

    n_comps = int(min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))
    sc.pp.pca(adata, n_comps=n_comps)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return adata


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest_path)
    entries = select_entries(manifest, args.donor, args.tubes_per_cytokine)
    print(
        f"Selected {len(entries)} pseudo-tubes for donor={args.donor} "
        f"(first {args.tubes_per_cytokine} tubes/cytokine)."
    )

    adata = load_and_concat(entries)
    print(f"Concatenated: {adata.n_obs} cells x {adata.n_vars} genes.")

    adata = embed(adata, args.n_pcs)

    coords = adata.obsm["X_umap"]
    df = pd.DataFrame(
        {
            "umap_x": np.asarray(coords[:, 0], dtype=float),
            "umap_y": np.asarray(coords[:, 1], dtype=float),
            "cell_type": adata.obs["cell_type"].astype(str).to_numpy(),
            "cytokine": adata.obs["cytokine"].astype(str).to_numpy(),
        }
    )

    out_path = out_dir / "umap_coords.parquet"
    df.to_parquet(out_path)

    print(f"Wrote {out_path} ({len(df)} rows).")
    print(f"  n_cells      = {len(df)}")
    print(f"  n_cytokines  = {df['cytokine'].nunique()}")
    print(f"  n_cell_types = {df['cell_type'].nunique()}")
    print("  cells per cell_type:")
    for ct, n in df["cell_type"].value_counts().items():
        print(f"    {ct:<24s} {n}")


if __name__ == "__main__":
    main()
