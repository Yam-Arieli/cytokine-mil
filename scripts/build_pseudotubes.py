"""
Build pseudo-tubes from the raw PBMC cytokine dataset.

Run on the cluster from the working directory:
    python scripts/build_pseudotubes.py

Applies the full preprocessing pipeline before building tubes:
  1. Doublet removal (Scrublet)
  2. Total count normalisation
  3. Log1p transformation
  4. HVG selection (n_top_genes HVGs, default 4000)

The gene list is saved to <BASE_PATH>/hvg_list.json for reproducibility.
"""

import argparse
import json
import os
import sys

import numpy as np
import scanpy as sc


# ---------------------------------------------------------------------------
# Defaults (match configs/default.yaml)
# ---------------------------------------------------------------------------
FILE_PATH = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus/"
    "Parse_10M_PBMC_cytokines.h5ad"
)
BASE_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes"
N_PER_CELL_TYPE = 30
MIN_CELLS_THRESHOLD = 10
N_PSEUDO_TUBES = 10
N_HVGS = 4000
RANDOM_SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Build pseudo-tubes from raw PBMC data.")
    p.add_argument("--file_path", default=FILE_PATH)
    p.add_argument("--base_path", default=BASE_PATH)
    p.add_argument("--n_per_cell_type", type=int, default=N_PER_CELL_TYPE)
    p.add_argument("--min_cells_threshold", type=int, default=MIN_CELLS_THRESHOLD)
    p.add_argument("--n_pseudo_tubes", type=int, default=N_PSEUDO_TUBES)
    p.add_argument("--n_hvgs", type=int, default=N_HVGS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def remove_doublets(adata, expected_doublet_rate: float = 0.06):
    """
    Run Scrublet doublet detection and filter predicted doublets.

    Scrublet requires the raw count matrix. Run before normalisation.
    """
    try:
        import scrublet as scr
    except ImportError:
        print(
            "WARNING: scrublet not installed. Skipping doublet removal.\n"
            "Install with: pip install scrublet",
            file=sys.stderr,
        )
        return adata

    scrub = scr.Scrublet(adata.X, expected_doublet_rate=expected_doublet_rate)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
    adata.obs["doublet_score"] = doublet_scores
    adata.obs["predicted_doublet"] = predicted_doublets

    n_before = adata.n_obs
    adata = adata[~adata.obs["predicted_doublet"]].copy()
    n_after = adata.n_obs
    print(f"  Doublet removal: {n_before} -> {n_after} cells ({n_before - n_after} removed)")
    return adata


def preprocess(adata):
    """Apply normalisation + log1p. Operates in-place."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def select_hvgs(adata, n_top_genes: int = 4000):
    """Select highly variable genes on the full (preprocessed) dataset."""
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    hvg_list = adata.var_names[adata.var["highly_variable"]].tolist()
    print(f"  Selected {len(hvg_list)} HVGs")
    return hvg_list


# ---------------------------------------------------------------------------
# Pseudo-tube construction
# ---------------------------------------------------------------------------

def build_pseudo_tubes(
    adata,
    base_path: str,
    hvg_list: list,
    n_per_cell_type: int,
    min_cells_threshold: int,
    n_pseudo_tubes: int,
    rng: np.random.Generator,
) -> list:
    """
    Build pseudo-tubes from a preprocessed AnnData.

    Variable tube sizes are preserved: cytokines inducing apoptosis or
    proliferation will produce tubes with fewer / more cells. Do NOT
    equalise across cytokines — this is a meaningful biological signal.
    """
    manifest = []

    for donor_name in sorted(adata.obs["donor"].unique()):
        print(f"\n{'='*50}")
        print(f"Processing {donor_name}...")

        adata_donor = adata[adata.obs["donor"] == donor_name].copy()

        for cytokine in sorted(adata_donor.obs["cytokine"].unique()):
            adata_cyt = adata_donor[adata_donor.obs["cytokine"] == cytokine]

            eligible_cell_types = [
                ct
                for ct in adata_cyt.obs["cell_type"].unique()
                if (adata_cyt.obs["cell_type"] == ct).sum() >= min_cells_threshold
            ]

            if not eligible_cell_types:
                print(f"  Skipping {cytokine}: no eligible cell types")
                continue

            safe_cyt = str(cytokine).replace(" ", "_").replace("/", "_")
            folder = os.path.join(base_path, donor_name, safe_cyt)
            os.makedirs(folder, exist_ok=True)

            for tube_idx in range(n_pseudo_tubes):
                pseudo_tube = _sample_one_tube(
                    adata_cyt, eligible_cell_types, n_per_cell_type, hvg_list, rng
                )
                out_path = os.path.join(folder, f"pseudotube_{tube_idx}.h5ad")
                pseudo_tube.write_h5ad(out_path, compression="gzip")

                manifest.append(
                    {
                        "path": out_path,
                        "donor": donor_name,
                        "cytokine": cytokine,
                        "n_cells": pseudo_tube.n_obs,
                        "cell_types_included": eligible_cell_types,
                        "tube_idx": tube_idx,
                    }
                )

            print(
                f"  {cytokine}: {n_pseudo_tubes} tubes, "
                f"{len(eligible_cell_types)} cell types, "
                f"{pseudo_tube.n_obs} cells each"
            )

    return manifest


def _sample_one_tube(adata_cyt, eligible_cell_types, n_per_cell_type, hvg_list, rng):
    """Sample one pseudo-tube: stratified by cell type, subset to HVGs, shuffled."""
    sampled = []
    for ct in eligible_cell_types:
        ct_cells = adata_cyt[adata_cyt.obs["cell_type"] == ct]
        n_sample = min(n_per_cell_type, ct_cells.n_obs)
        chosen = rng.choice(ct_cells.n_obs, size=n_sample, replace=False)
        sampled.append(ct_cells[chosen].copy())

    tube = sc.concat(sampled, join="outer")

    # Subset to HVGs
    tube = tube[:, hvg_list].copy()

    # Shuffle cell order so cell type is not implicit in position
    perm = rng.permutation(tube.n_obs)
    return tube[perm].copy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.base_path, exist_ok=True)

    print(f"Loading dataset: {args.file_path}")
    adata_backed = sc.read_h5ad(args.file_path, backed="r")

    print("Loading full dataset into memory for preprocessing...")
    adata = adata_backed.to_memory()

    print("Step 1: Doublet removal")
    adata = remove_doublets(adata)

    print("Step 2: Normalisation + log1p")
    preprocess(adata)

    print(f"Step 3: HVG selection (n_top_genes={args.n_hvgs})")
    hvg_list = select_hvgs(adata, n_top_genes=args.n_hvgs)

    # Save HVG list for reproducibility
    hvg_path = os.path.join(args.base_path, "hvg_list.json")
    with open(hvg_path, "w") as f:
        json.dump(hvg_list, f, indent=2)
    print(f"HVG list saved to: {hvg_path}")

    print("Step 4: Building pseudo-tubes")
    manifest = build_pseudo_tubes(
        adata,
        base_path=args.base_path,
        hvg_list=hvg_list,
        n_per_cell_type=args.n_per_cell_type,
        min_cells_threshold=args.min_cells_threshold,
        n_pseudo_tubes=args.n_pseudo_tubes,
        rng=rng,
    )

    manifest_path = os.path.join(args.base_path, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} pseudo-tubes saved.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
