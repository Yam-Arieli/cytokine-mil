"""
Build pseudo-tubes from the Oelen et al. 2022 (1M-scBloodNL) dataset.

Run on the cluster from the working directory:
    python scripts/build_pseudotubes_oelen.py

Dataset: v3 chemistry only. v2 has ~900 genes/cell vs ~1860 for v3 —
mixing would introduce a strong technical confound.

Conditions used: 24h timepoints + unstimulated control only.
    UT     — unstimulated control (analog of PBS)
    24hCA  — C. albicans
    24hMTB — M. tuberculosis
    24hPA  — P. aeruginosa

Column mapping vs Oesinghaus:
    donor     -> assignment
    cytokine  -> timepoint  (manifest key still called 'cytokine' for compatibility)
    cell_type -> cell_type_lowerres

Preprocessing pipeline (applied before pseudo-tube construction):
  1. Filter conditions: keep only {UT, 24hCA, 24hMTB, 24hPA}
  2. Filter cells: min 200 genes detected, max 5% mitochondrial content
  3. Filter genes: min 10 cells expressing the gene
  4. Doublet removal (Scrublet)
  5. Total count normalisation (target_sum=1e4)
  6. Log1p transformation
  7. HVG selection (n_top_genes=4000, seurat_v3)

The gene list is saved to <BASE_PATH>/hvg_list.json for reproducibility.

The manifest structure is identical to the Oesinghaus manifest so that
the existing PseudoTubeDataset and CytokineLabel classes can consume it
without any changes.
"""

import argparse
import json
import os
import sys

import numpy as np
import scanpy as sc


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
FILE_PATH = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oelen/1m_scbloodnl_v3.h5ad"
)
BASE_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oelen_pseudotubes"
N_PER_CELL_TYPE = 30
MIN_CELLS_THRESHOLD = 10
N_PSEUDO_TUBES = 10
N_HVGS = 4000
RANDOM_SEED = 42

# Only 24h conditions + unstimulated control
CONDITIONS_TO_USE = {"UT", "24hCA", "24hMTB", "24hPA"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Build pseudo-tubes from the Oelen 2022 (v3) dataset."
    )
    p.add_argument("--file_path", default=FILE_PATH,
                   help="Path to 1m_scbloodnl_v3.h5ad (default: %(default)s)")
    p.add_argument("--base_path", default=BASE_PATH,
                   help="Output directory for pseudo-tubes (default: %(default)s)")
    p.add_argument("--n_per_cell_type", type=int, default=N_PER_CELL_TYPE,
                   help="Max cells sampled per cell type per tube (default: %(default)s)")
    p.add_argument("--min_cells_threshold", type=int, default=MIN_CELLS_THRESHOLD,
                   help="Min cells for a cell type to be eligible (default: %(default)s)")
    p.add_argument("--n_pseudo_tubes", type=int, default=N_PSEUDO_TUBES,
                   help="Number of pseudo-tubes per donor/condition (default: %(default)s)")
    p.add_argument("--n_hvgs", type=int, default=N_HVGS,
                   help="Number of highly variable genes (default: %(default)s)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help="Random seed (default: %(default)s)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def filter_conditions(adata):
    """
    Keep only the 24h timepoints and unstimulated control.

    Removes 3h timepoints (too early for stable transcriptional cascade
    signatures) and any other unexpected conditions.
    """
    mask = adata.obs["timepoint"].isin(CONDITIONS_TO_USE)
    n_before = adata.n_obs
    adata = adata[mask].copy()
    n_after = adata.n_obs
    dropped = n_before - n_after
    print(f"  Condition filter: kept {sorted(CONDITIONS_TO_USE)}")
    print(f"  Cells: {n_before} -> {n_after} ({dropped} removed)")
    return adata


def filter_cells_qc(adata):
    """
    Filter cells: min 200 genes, max 5% mitochondrial content.

    The Oelen data is raw unnormalized counts. QC filtering must happen
    before normalisation.
    """
    # Mitochondrial gene flag
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    n_after_genes = adata.n_obs

    mask_mito = adata.obs["pct_counts_mt"] <= 5.0
    adata = adata[mask_mito].copy()
    n_after_mito = adata.n_obs

    print(f"  Cell QC: {n_before} -> {n_after_genes} (min_genes=200) "
          f"-> {n_after_mito} (max_mito=5%)")
    return adata


def filter_genes_qc(adata):
    """Filter genes: min 10 cells expressing the gene."""
    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=10)
    n_after = adata.n_vars
    print(f"  Gene QC: {n_before} -> {n_after} genes (min_cells=10)")
    return adata


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
    print(f"  Doublet removal: {n_before} -> {n_after} cells "
          f"({n_before - n_after} removed)")
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


def check_complete_donors(adata):
    """
    Retain only donors that have all 4 conditions with at least one cell.

    A donor missing any condition would have tubes with no valid labels for
    that class, biasing the learning dynamics analysis.
    """
    required = CONDITIONS_TO_USE
    donor_col = "assignment"
    cond_col = "timepoint"

    all_donors = adata.obs[donor_col].unique()
    complete = []
    for donor in all_donors:
        donor_mask = adata.obs[donor_col] == donor
        conditions_present = set(adata.obs.loc[donor_mask, cond_col].unique())
        if required.issubset(conditions_present):
            complete.append(donor)

    n_before = adata.n_obs
    adata = adata[adata.obs[donor_col].isin(complete)].copy()
    n_after = adata.n_obs
    print(f"  Complete-donor filter: {len(complete)}/{len(all_donors)} donors "
          f"have all 4 conditions")
    print(f"  Cells: {n_before} -> {n_after}")
    print(f"  Donors retained: {sorted(complete)}")
    return adata


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
    Build pseudo-tubes from a preprocessed AnnData (Oelen dataset).

    Column mapping:
        assignment       -> donor identifier
        timepoint        -> condition / class (stored as 'cytokine' in manifest
                            for compatibility with PseudoTubeDataset)
        cell_type_lowerres -> stratification unit (coarse cell type)

    Variable tube sizes are preserved — same design decision as Oesinghaus.
    """
    manifest = []
    donor_col = "assignment"
    cond_col = "timepoint"
    ct_col = "cell_type_lowerres"

    for donor_name in sorted(adata.obs[donor_col].unique()):
        print(f"\n{'='*50}")
        print(f"Processing donor {donor_name}...")

        adata_donor = adata[adata.obs[donor_col] == donor_name].copy()

        for timepoint in sorted(adata_donor.obs[cond_col].unique()):
            adata_cond = adata_donor[adata_donor.obs[cond_col] == timepoint]

            eligible_cell_types = [
                ct
                for ct in adata_cond.obs[ct_col].unique()
                if (adata_cond.obs[ct_col] == ct).sum() >= min_cells_threshold
            ]

            if not eligible_cell_types:
                print(f"  Skipping {timepoint}: no eligible cell types")
                continue

            safe_tp = str(timepoint).replace(" ", "_").replace("/", "_")
            folder = os.path.join(base_path, f"donor_{donor_name}", safe_tp)
            os.makedirs(folder, exist_ok=True)

            for tube_idx in range(n_pseudo_tubes):
                pseudo_tube = _sample_one_tube(
                    adata_cond, eligible_cell_types, ct_col,
                    n_per_cell_type, hvg_list, rng,
                )
                out_path = os.path.join(folder, f"pseudotube_{tube_idx}.h5ad")
                pseudo_tube.write_h5ad(out_path, compression="gzip")

                # Store timepoint as 'cytokine' so existing dataset/label
                # encoder classes work without modification.
                manifest.append(
                    {
                        "path": out_path,
                        "donor": f"donor_{donor_name}",
                        "cytokine": timepoint,
                        "n_cells": pseudo_tube.n_obs,
                        "cell_types_included": eligible_cell_types,
                        "tube_idx": tube_idx,
                    }
                )

            print(
                f"  {timepoint}: {n_pseudo_tubes} tubes, "
                f"{len(eligible_cell_types)} cell types, "
                f"{pseudo_tube.n_obs} cells each"
            )

    return manifest


def _sample_one_tube(adata_cond, eligible_cell_types, ct_col,
                     n_per_cell_type, hvg_list, rng):
    """Sample one pseudo-tube: stratified by cell type, subset to HVGs, shuffled."""
    sampled = []
    for ct in eligible_cell_types:
        ct_cells = adata_cond[adata_cond.obs[ct_col] == ct]
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

    print(f"Loading dataset (backed): {args.file_path}")
    adata_backed = sc.read_h5ad(args.file_path, backed="r")

    print("Loading full dataset into memory...")
    adata = adata_backed.to_memory()
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Conditions present: {sorted(str(x) for x in adata.obs['timepoint'].unique() if x == x)}")
    print(f"Donors present:     {len(adata.obs['assignment'].unique())}")

    print("\nStep 1: Filter to 24h + UT conditions")
    adata = filter_conditions(adata)

    print("\nStep 2: QC — filter cells (min_genes=200, max_mito=5%)")
    adata = filter_cells_qc(adata)

    print("\nStep 3: QC — filter genes (min_cells=10)")
    adata = filter_genes_qc(adata)

    print("\nStep 4: Filter to donors with all 4 conditions")
    adata = check_complete_donors(adata)

    print("\nStep 5: Doublet removal (Scrublet)")
    adata = remove_doublets(adata)

    print("\nStep 6: Normalisation + log1p")
    preprocess(adata)

    print(f"\nStep 7: HVG selection (n_top_genes={args.n_hvgs})")
    hvg_list = select_hvgs(adata, n_top_genes=args.n_hvgs)

    # Save HVG list for reproducibility
    hvg_path = os.path.join(args.base_path, "hvg_list.json")
    with open(hvg_path, "w") as f:
        json.dump(hvg_list, f, indent=2)
    print(f"HVG list saved to: {hvg_path}")

    print("\nStep 8: Building pseudo-tubes")
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

    # Summary
    conditions = sorted({e["cytokine"] for e in manifest})
    donors = sorted({e["donor"] for e in manifest})
    print(f"Conditions: {conditions}")
    print(f"Donors:     {len(donors)}")


if __name__ == "__main__":
    main()
