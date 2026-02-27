"""
Generate simulated demo data for local testing.

Creates a directory structure that mirrors the cluster data exactly:
  <base_dir>/
    manifest.json
    Donor1/
      IL-2/
        pseudotube_0.h5ad
      ...
    Donor2/
      ...

Spec:
  - 10 cytokines + PBS = 11 classes
  - 3 donors (Donor3 held out in val split tests)
  - 1 pseudo-tube per (donor, cytokine)
  - 5 cell types, 20 cells per cell type -> 100 cells per pseudo-tube
  - 200 simulated genes (log-normalised counts)
"""

import json
import os
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd


CYTOKINES = [
    "IL-2", "IL-4", "IL-6", "IL-10", "IL-12",
    "IL-15", "IFN-alpha", "IFN-gamma", "TNF", "TGF-beta",
]
DONORS = ["Donor1", "Donor2", "Donor3"]
CELL_TYPES = ["CD4_T", "CD8_T", "NK", "CD14_Mono", "B_cell"]
N_GENES = 200
N_CELLS_PER_TYPE = 20
N_PSEUDO_TUBES = 1


def make_demo_data(base_dir: str, seed: int = 42) -> str:
    """
    Write demo pseudo-tubes and manifest to base_dir.

    Args:
        base_dir: Directory to write data into (created if needed).
        seed: Random seed for reproducibility.
    Returns:
        Path to manifest.json.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    gene_names = [f"gene_{i:04d}" for i in range(N_GENES)]
    all_conditions = CYTOKINES + ["PBS"]
    manifest = []

    for donor in DONORS:
        for cytokine in all_conditions:
            safe_cyt = cytokine.replace(" ", "_").replace("/", "_")
            folder = base_dir / donor / safe_cyt
            folder.mkdir(parents=True, exist_ok=True)

            for tube_idx in range(N_PSEUDO_TUBES):
                adata = _make_pseudo_tube(
                    rng, gene_names, cytokine, donor, tube_idx
                )
                out_path = folder / f"pseudotube_{tube_idx}.h5ad"
                adata.write_h5ad(str(out_path))

                manifest.append(
                    {
                        "path": str(out_path),
                        "donor": donor,
                        "cytokine": cytokine,
                        "n_cells": adata.n_obs,
                        "cell_types_included": CELL_TYPES,
                        "tube_idx": tube_idx,
                    }
                )

    manifest_path = base_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return str(manifest_path)


def _make_pseudo_tube(
    rng: np.random.Generator,
    gene_names: list,
    cytokine: str,
    donor: str,
    tube_idx: int,
) -> ad.AnnData:
    """Create a single simulated pseudo-tube AnnData."""
    n_total = len(CELL_TYPES) * N_CELLS_PER_TYPE

    # Simulate log-normalised expression with mild cytokine-specific signal
    X = rng.lognormal(mean=0.5, sigma=1.0, size=(n_total, len(gene_names))).astype(
        np.float32
    )
    # Add a weak cytokine-specific shift on the first few genes
    cyt_idx = (CYTOKINES + ["PBS"]).index(cytokine)
    X[:, : cyt_idx + 1] += rng.normal(0.3, 0.1, size=(n_total, cyt_idx + 1)).astype(
        np.float32
    )

    # Build obs with cell_type labels
    cell_types = []
    for ct in CELL_TYPES:
        cell_types.extend([ct] * N_CELLS_PER_TYPE)

    obs = pd.DataFrame(
        {"cell_type": cell_types, "donor": donor, "cytokine": cytokine},
        index=[f"{donor}_{cytokine}_{tube_idx}_cell{i}" for i in range(n_total)],
    )

    # Shuffle cell order
    perm = rng.permutation(n_total)
    X = X[perm]
    obs = obs.iloc[perm].reset_index(drop=False).rename(columns={"index": "orig_index"})
    obs.index = [f"{donor}_{cytokine}_{tube_idx}_cell{i}" for i in range(n_total)]

    var = pd.DataFrame(index=gene_names)
    return ad.AnnData(X=X, obs=obs, var=var)


if __name__ == "__main__":
    import tempfile
    import sys

    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="cytokine_demo_")
    path = make_demo_data(out_dir)
    print(f"Demo data written to: {path}")
