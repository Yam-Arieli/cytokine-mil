"""
Build pseudo-tubes from the Sheu et al. 2024 BMDM time-course dataset (GSE224518).

Run on the cluster from the working directory:
    python scripts/build_pseudotubes_sheu2024.py

Dataset structure (see CLAUDE.md §2.5):
  - Platform: BD Rhapsody targeted scRNA-seq (500 immune-response mouse genes)
  - 13 GSM accessions multiplexed via MULTI-seq sample-tag hashing; the
    global `GSE224518_samptag.all_cellannotations_metadata.txt.gz` file
    demultiplexes them into the full design grid.
  - 12 biological contexts (M0/M1_IFNg/M2_IL4 BMDMs, BMDM strain variants,
    PM strain backgrounds); 2 replicates per condition; 8 time points
    (0/0.25/0.5/1/3/5/8/24 hr).

Phase 1 protocol (kept fixed):
  - Time-point subset: {0hr, 3hr}.
  - Active stimuli: LPS, LPSlo, Pam3CSK4 (P3CSK), polyIC (PIC), TNF, CpG, IFNb,
    plus all Unstim and all 0hr cells relabeled to "PBS".
  - Pseudo-donor = f"{type}_{replicate}". 7 pseudo-donors at 3hr (see §2.5).
  - val pseudo-donors: ["M2_IL4_rep1", "PM_B6.old_rep1"] (see CLAUDE.md §15).
  - Cell types: global Leiden on all 0hr Unstim cells pooled across pseudo-donors,
    labels `mac_c0`, `mac_c1`, ...; post-stim cells assigned to nearest 0h
    cluster centroid in PCA space.
  - No HVG selection: keep all 500 targeted-panel genes.

The manifest structure mirrors the existing PseudoTubeDataset contract so
all downstream code (Stage 1/2 training, dynamics, latent_geometry) works
unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults (cluster paths; override via CLI)
# ---------------------------------------------------------------------------
RAW_DIR = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw"
BASE_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes"

N_PER_CELL_TYPE = 30
MIN_CELLS_THRESHOLD = 10
N_PSEUDO_TUBES = 10
N_HVGS = 500  # = full Sheu targeted panel; HVG selection is a no-op
RANDOM_SEED = 42

# Phase 1: keep only these time points; all cells with 0hr or Unstim become PBS
TIME_POINTS_KEEP = {"0hr", "3hr"}

# Active stimuli at 3hr (PBS class produced via relabeling, not present as a literal stimulus)
ACTIVE_STIMULI_3HR = {"LPS", "LPSlo", "P3CSK", "PIC", "TNF", "CpG", "IFNb"}

# Default val pseudo-donors per CLAUDE.md §15
VAL_PSEUDO_DONORS_DEFAULT = ["M2_IL4_rep1", "PM_B6.old_rep1"]

SAMPTAG_METADATA_FILENAME = "GSE224518_samptag.all_cellannotations_metadata.txt.gz"


# ===========================================================================
# BD Rhapsody CSV parser (used by main(); not exercised by demo tests)
# ===========================================================================

_BD_HEADER_COMMENT_RE = re.compile(r"^##")


def _read_bd_rhapsody_csv(path: str) -> pd.DataFrame:
    """
    Read one BD Rhapsody DBEC_MolsPerCell.csv.gz file.

    Format:
      - 7 leading lines starting with `##` (pipeline metadata)
      - then a header row: `Cell_Index,gene1,gene2,...`
      - then integer count rows
    """
    with gzip.open(path, "rt") as fh:
        lines = []
        header_seen = False
        for line in fh:
            if not header_seen and _BD_HEADER_COMMENT_RE.match(line):
                continue
            lines.append(line)
            header_seen = True
    df = pd.read_csv(io.StringIO("".join(lines)))
    return df


def _infer_batch_id_from_overlap(gsm_cell_indices: set, batch_cell_sets: dict) -> int:
    """
    Infer the batch_id of a BD Rhapsody count file by overlapping its
    Cell_Index set with each metadata batch's Cell_Index set.

    Returns the batch_id with the largest overlap. Raises if the best overlap
    is below `min_overlap_fraction` of the GSM's cell count, indicating
    a missing or wrong-format mapping.

    This replaces the previous regex-based filename parser, which failed on
    GSE224518's `-2019sample-Rev-Primer*` and `_2_` GSM filenames. The
    metadata's `batch` column is the source of truth; filenames are unreliable.

    Args:
        gsm_cell_indices: Set of integer Cell_Index values in the GSM file
                         (without any batch prefix).
        batch_cell_sets: Dict {batch_id: set of int Cell_Index values present
                         in the metadata for that batch, with prefix stripped}.
    """
    best_batch = None
    best_overlap = 0
    for batch_id, batch_set in batch_cell_sets.items():
        overlap = len(gsm_cell_indices & batch_set)
        if overlap > best_overlap:
            best_batch = batch_id
            best_overlap = overlap
    if best_batch is None or best_overlap < max(1, len(gsm_cell_indices) * 0.5):
        raise ValueError(
            f"Could not infer batch_id (best overlap {best_overlap} / "
            f"{len(gsm_cell_indices)} cells). Either the file is from a "
            f"batch not present in metadata or the prefix scheme has changed."
        )
    return best_batch


def load_sheu_anndata(raw_dir: str) -> ad.AnnData:
    """
    Load all BD Rhapsody count files in `raw_dir` and the global samptag
    metadata into a single AnnData with `obs[pseudo_donor, cytokine,
    cell_type=NA placeholder, time_point, batch, sample_tag, replicate, type]`.

    Cell typing (Leiden) is applied separately in `assign_cell_types_global_leiden`.

    Returns the assembled AnnData. Cells with NA design factors, Multiplets,
    and Undetermined sample tags are dropped at this stage.
    """
    raw_dir = Path(raw_dir)

    # 1) Read global samptag metadata
    meta_path = raw_dir / SAMPTAG_METADATA_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing samptag metadata: {meta_path}\n"
            f"Expected the global cell-annotation file from GSE224518."
        )
    meta = pd.read_csv(meta_path, sep="\t")
    # Drop cells with NA on any required column
    required_meta_cols = ["Cell_Index", "Sample_Tag", "timept", "stimulus", "type", "batch", "replicate"]
    for col in required_meta_cols:
        if col not in meta.columns:
            raise KeyError(f"Samptag metadata missing column: {col}")
    meta = meta.dropna(subset=required_meta_cols).copy()
    # Drop Multiplet and Undetermined sample tags
    meta = meta[~meta["Sample_Tag"].isin(["Multiplet", "Undetermined"])].copy()

    # 2) Locate all *_DBEC_MolsPerCell.csv.gz files. Glob is broader than
    #    `SampleTag*` to also catch GSE224518's `_2_` re-runs and `-2019sample-`
    #    re-deposits whose filenames don't follow the SampleTagN-Rev-PrimerN
    #    convention used by the first 10 GSMs.
    count_files = sorted(raw_dir.glob("GSM*_Combined_*_DBEC_MolsPerCell.csv.gz"))
    if not count_files:
        raise FileNotFoundError(f"No BD Rhapsody count files found under {raw_dir}")

    # 3) Build a per-batch Cell_Index set from metadata for content-based
    #    batch_id inference (replaces the brittle filename regex).
    #    Metadata's Cell_Index is `f"{batch}_{cell_int}"`; we split off the int.
    batch_cell_sets: dict[int, set] = {}
    for bid, sub in meta.groupby("batch"):
        suffixes = sub["Cell_Index"].astype(str).str.rsplit("_", n=1).str[-1]
        batch_cell_sets[int(bid)] = set(suffixes.astype(int))

    # 4) Read each GSM count file, infer its batch_id by overlap, prefix
    #    Cell_Index with batch id, concat.
    parts = []
    var_names: Optional[list] = None
    for fpath in count_files:
        df = _read_bd_rhapsody_csv(str(fpath))
        if var_names is None:
            var_names = [c for c in df.columns if c != "Cell_Index"]
        cell_ints = set(df["Cell_Index"].astype(int))
        batch_id = _infer_batch_id_from_overlap(cell_ints, batch_cell_sets)
        df["Cell_Index"] = df["Cell_Index"].astype(int).astype(str)
        df["Cell_Index"] = f"{batch_id}_" + df["Cell_Index"]
        df["__batch__"] = batch_id
        parts.append(df)
        print(f"  {fpath.name} -> batch {batch_id} ({len(df)} cells)")
    full = pd.concat(parts, axis=0, ignore_index=True)

    # 4) Join with metadata on Cell_Index
    #    Metadata's Cell_Index is already prefixed like "1_584933"
    meta_int = meta.copy()
    meta_int["Cell_Index"] = meta_int["Cell_Index"].astype(str)
    joined = full.merge(meta_int, on="Cell_Index", how="inner")

    # 5) Build AnnData
    obs = joined[["Cell_Index", "Sample_Tag", "timept", "stimulus", "type", "batch", "replicate"]].copy()
    obs.rename(
        columns={
            "Sample_Tag": "sample_tag",
            "timept": "time_point",
            "stimulus": "cytokine",
        },
        inplace=True,
    )
    obs["pseudo_donor"] = obs["type"].astype(str) + "_" + obs["replicate"].astype(str)
    # cell_type placeholder; will be filled by global Leiden in a later step
    obs["cell_type"] = "unassigned"
    obs.index = joined["Cell_Index"].values

    X = joined[var_names].to_numpy(dtype=np.float32, copy=False)
    var = pd.DataFrame(index=var_names)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


def filter_phase1_subset(adata: ad.AnnData) -> ad.AnnData:
    """
    Phase 1: keep only `time_point in {0hr, 3hr}` and active stimuli (+ Unstim).
    Cells with stimuli outside the active set are dropped.
    """
    keep_stimuli = ACTIVE_STIMULI_3HR | {"Unstim"}
    mask = adata.obs["time_point"].isin(TIME_POINTS_KEEP) & adata.obs["cytokine"].isin(keep_stimuli)
    return adata[mask.values].copy()


def assign_cell_types_global_leiden(
    adata: ad.AnnData, resolution: float = 0.2, n_pcs: int = 30
) -> ad.AnnData:
    """
    Global Leiden cell typing.

    1. Subset to all 0hr Unstim cells (pooled across pseudo-donors).
    2. PCA on that subset.
    3. Leiden cluster on KNN graph in PCA space.
    4. Compute per-cluster centroids in PCA space.
    5. For every cell in the full AnnData, project to the same PCA basis and
       assign the nearest cluster centroid.

    Writes `adata.obs["cell_type"]` as `f"mac_c{cluster_id}"`.
    """
    import scanpy as sc  # local import: scanpy is heavy

    if (adata.obs["cytokine"] == "Unstim").sum() == 0:
        # Fall back: cluster on all cells if no Unstim subset exists (test fixture)
        sub = adata
    else:
        sub_mask = (adata.obs["cytokine"] == "Unstim") & (adata.obs["time_point"] == "0hr")
        sub = adata[sub_mask.values].copy()

    # PCA on the reference subset
    sc.pp.pca(sub, n_comps=min(n_pcs, sub.n_vars - 1, sub.n_obs - 1))
    sc.pp.neighbors(sub)
    sc.tl.leiden(sub, resolution=resolution)

    # Cluster centroids in PCA space
    pca_components = sub.varm["PCs"]                 # (n_genes, n_pcs)
    pca_mean = sub.X.mean(axis=0)                    # (n_genes,)
    pca_mean = np.asarray(pca_mean).ravel()
    sub_pca = (np.asarray(sub.X) - pca_mean) @ pca_components

    cluster_labels = sub.obs["leiden"].astype(int).values
    centroids = np.stack(
        [sub_pca[cluster_labels == k].mean(axis=0) for k in sorted(set(cluster_labels))]
    )

    # Project every cell in the full AnnData onto the same basis
    X_full = np.asarray(adata.X)
    X_centered = X_full - pca_mean
    full_pca = X_centered @ pca_components

    # Assign nearest cluster centroid
    dists = ((full_pca[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = dists.argmin(axis=1)
    adata.obs["cell_type"] = [f"mac_c{int(k)}" for k in nearest]
    return adata


# ===========================================================================
# Relabeling (PBS contract)
# ===========================================================================

def relabel_to_pbs(adata: ad.AnnData) -> ad.AnnData:
    """
    Relabel `cytokine` to `"PBS"` for any cell that is Unstim or at 0hr.

    The cytokine-MIL pipeline pins PBS to index 90 (`label_encoder.py:11`)
    and `analysis/pbs_rc.py:59` hard-checks `cytokine == "PBS"`. By relabeling
    at the adapter boundary we keep the package code untouched.

    Operates in-place on `adata.obs["cytokine"]` and also returns the same
    AnnData for convenience.
    """
    cyt = adata.obs["cytokine"].astype(str).copy()
    tp = adata.obs["time_point"].astype(str)
    is_unstim = cyt == "Unstim"
    is_0hr = tp == "0hr"
    relabel_mask = (is_unstim | is_0hr).values
    cyt.loc[relabel_mask] = "PBS"
    adata.obs["cytokine"] = cyt.values
    return adata


# ===========================================================================
# Pseudo-tube construction (testable function exercised by tests/test_demo_sheu.py)
# ===========================================================================

def build_pseudo_tubes_sheu(
    adata: ad.AnnData,
    base_path: str,
    n_per_cell_type: int = N_PER_CELL_TYPE,
    min_cells_threshold: int = MIN_CELLS_THRESHOLD,
    n_pseudo_tubes: int = N_PSEUDO_TUBES,
    rng: Optional[np.random.Generator] = None,
) -> list:
    """
    Build pseudo-tubes from a Sheu-style AnnData and write them under
    `base_path` with the same directory/manifest schema as the Oesinghaus and
    Oelen pipelines.

    Required obs columns on `adata`:
      - pseudo_donor : str  (e.g., "M0_rep1") — used as the manifest "donor" key
      - cytokine     : str  (already PBS-relabeled; this function does NOT relabel)
      - cell_type    : str  (e.g., "mac_c0") — used for stratified sampling

    Variable tube sizes are preserved (same design as Oesinghaus and Oelen).
    The manifest schema is exactly:
      {path, donor, cytokine, n_cells, cell_types_included, tube_idx}

    Returns the manifest list (also written to `<base_path>/manifest.json`).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    donor_col = "pseudo_donor"
    cond_col = "cytokine"
    ct_col = "cell_type"

    manifest = []
    donors = sorted(adata.obs[donor_col].unique())
    for donor_name in donors:
        adata_donor = adata[adata.obs[donor_col] == donor_name].copy()

        for cyt in sorted(adata_donor.obs[cond_col].unique()):
            adata_cond = adata_donor[adata_donor.obs[cond_col] == cyt]

            eligible_cell_types = [
                ct
                for ct in adata_cond.obs[ct_col].unique()
                if (adata_cond.obs[ct_col] == ct).sum() >= min_cells_threshold
            ]
            if not eligible_cell_types:
                print(f"  Skipping {donor_name}/{cyt}: no eligible cell types")
                continue

            safe_cyt = str(cyt).replace(" ", "_").replace("/", "_")
            folder = base_path / donor_name / safe_cyt
            folder.mkdir(parents=True, exist_ok=True)

            for tube_idx in range(n_pseudo_tubes):
                pseudo_tube = _sample_one_tube(
                    adata_cond, eligible_cell_types, ct_col, n_per_cell_type, rng
                )
                # Stamp the metadata onto the tube obs so downstream code
                # can read it without re-joining manifest.
                pseudo_tube.obs["donor"] = donor_name
                pseudo_tube.obs["cytokine"] = cyt

                out_path = folder / f"pseudotube_{tube_idx}.h5ad"
                pseudo_tube.write_h5ad(str(out_path), compression="gzip")

                manifest.append(
                    {
                        "path": str(out_path),
                        "donor": donor_name,
                        "cytokine": cyt,
                        "n_cells": int(pseudo_tube.n_obs),
                        "cell_types_included": list(map(str, eligible_cell_types)),
                        "tube_idx": tube_idx,
                    }
                )

    manifest_path = base_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _sample_one_tube(adata_cond, eligible_cell_types, ct_col, n_per_cell_type, rng):
    """Sample one pseudo-tube: stratified by cell type, shuffled, dtype-stable."""
    import scanpy as sc  # local: only needed for sc.concat in some paths

    sampled = []
    for ct in eligible_cell_types:
        ct_cells = adata_cond[adata_cond.obs[ct_col] == ct]
        n_sample = min(n_per_cell_type, ct_cells.n_obs)
        chosen = rng.choice(ct_cells.n_obs, size=n_sample, replace=False)
        sampled.append(ct_cells[chosen].copy())

    tube = sc.concat(sampled, join="outer")

    # Shuffle cell order so cell type is not implicit in row position
    perm = rng.permutation(tube.n_obs)
    tube = tube[perm].copy()

    # Ensure dtype is float32 (downstream PyTorch consumers expect float32)
    if hasattr(tube.X, "toarray"):
        X = tube.X.toarray().astype(np.float32)
    else:
        X = np.asarray(tube.X, dtype=np.float32)
    tube.X = X
    return tube


# ===========================================================================
# CLI entry point (cluster-only; not exercised by tests)
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Build pseudo-tubes from the Sheu et al. 2024 (GSE224518) dataset."
    )
    p.add_argument("--raw_dir", default=RAW_DIR, help="Dir with extracted GSE224518 raw files")
    p.add_argument("--base_path", default=BASE_PATH, help="Output directory for pseudo-tubes")
    p.add_argument("--n_per_cell_type", type=int, default=N_PER_CELL_TYPE)
    p.add_argument("--min_cells_threshold", type=int, default=MIN_CELLS_THRESHOLD)
    p.add_argument("--n_pseudo_tubes", type=int, default=N_PSEUDO_TUBES)
    p.add_argument("--n_hvgs", type=int, default=N_HVGS, help="No-op for Sheu (panel is 500 genes)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main():
    import scanpy as sc

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.base_path, exist_ok=True)

    print(f"Step 1: Load BD Rhapsody files from {args.raw_dir}")
    adata = load_sheu_anndata(args.raw_dir)
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  Pseudo-donors present: {sorted(adata.obs['pseudo_donor'].unique())}")

    print("\nStep 2: Phase-1 filter (time_point in {0hr, 3hr}, active stimuli + Unstim)")
    adata = filter_phase1_subset(adata)
    print(f"  After phase-1 filter: {adata.n_obs} cells")

    print("\nStep 3: QC (min_genes=20 — targeted panel is small)")
    sc.pp.filter_cells(adata, min_genes=20)
    print(f"  After cell QC: {adata.n_obs} cells")

    print("\nStep 4: Normalize + log1p")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("\nStep 5: Global Leiden cell typing on 0hr Unstim subset")
    adata = assign_cell_types_global_leiden(adata)
    print(f"  Cell types assigned: {sorted(adata.obs['cell_type'].unique())}")

    print("\nStep 6: Relabel Unstim/0hr -> PBS")
    adata = relabel_to_pbs(adata)
    print(f"  Active classes after relabel: {sorted(adata.obs['cytokine'].unique())}")

    print("\nStep 7: Save HVG list (= full 500-gene targeted panel, no HVG selection)")
    hvg_path = os.path.join(args.base_path, "hvg_list.json")
    with open(hvg_path, "w") as f:
        json.dump(list(adata.var_names), f, indent=2)
    print(f"  Wrote {hvg_path}")

    print("\nStep 8: Build pseudo-tubes")
    manifest = build_pseudo_tubes_sheu(
        adata,
        base_path=args.base_path,
        n_per_cell_type=args.n_per_cell_type,
        min_cells_threshold=args.min_cells_threshold,
        n_pseudo_tubes=args.n_pseudo_tubes,
        rng=rng,
    )
    print(f"\nDone. {len(manifest)} pseudo-tubes written under {args.base_path}")


if __name__ == "__main__":
    main()
