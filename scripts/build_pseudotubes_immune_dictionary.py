"""
Build pseudo-tubes from the Immune Dictionary — Cui et al., Nature 2024 (GSE202186 /
SCP2554).

Run on the cluster from the working directory:
    python scripts/build_pseudotubes_immune_dictionary.py

Dataset structure (see CLAUDE.md §2.7):
  - Source: Cui A. et al., Nature 625, 377–384 (2024). DOI: 10.1038/s41586-023-06816-9
  - Data deposit: Broad Institute Single Cell Portal SCP2554; also GEO GSE202186.
  - Platform: 10x Genomics Chromium 3' v3, whole transcriptome (~31,053 genes).
  - Design: 86 cytokines + PBS-injected controls; 3 C57BL/6 mice per cytokine;
    single 4h time point (in vivo subcutaneous/intradermal injection).
  - ~386,703 total cells.

Pipeline (fixed):
  - Each mouse is one donor (pseudo-donor convention collapses to `mouse_id`).
  - PBS-injected control cells relabeled to literal string "PBS" at adapter boundary.
  - Cell types: global Leiden on all PBS control cells pooled across mice.
    Labels `id_c0`, `id_c1`, …; post-stim cells assigned to nearest PBS centroid
    in PCA space. Target: 4–8 clusters (immune family granularity).
  - Val mouse: the mouse with the most outlier PBS PCA centroid (computed at build
    time; documented in manifest). Train: 2 mice per cytokine.
  - HVG selection: 4000 HVGs (Seurat v3, batch_key='mouse_id'), same as Oesinghaus.
  - Raw counts preserved; both raw and preprocessed .h5ad written per tube.

Manifest structure mirrors PseudoTubeDataset contract:
  {path, donor, cytokine, n_cells, cell_types_included, tube_idx}
  where `donor` = mouse_id.

TODO (after raw data download completes):
  - Verify GSE202186_RAW.tar extraction layout — expected: one subdirectory per
    sample named by GSM accession, each containing MTX + barcodes + features
    (standard 10x output). Adjust `_inspect_raw_layout()` accordingly if the
    structure differs.
  - Verify column names in GSE202186_map-scRNAseq-cytokines-dictionary.xlsx —
    spec expects `GSM_id`, `cytokine`, `mouse_id`; raise with clear error if absent.
  - Confirm the SOFT file (GSE202186_family.soft.gz) is needed at all or if the xlsx
    is fully authoritative; update `_parse_soft_file()` accordingly.
  - Check whether the cytokine column uses the string "PBS" for PBS-injected controls
    or something like "vehicle" / "untreated" / "PBS"; adjust PBS_CONTROL_STRINGS.
  - Confirm that the extracted MTX directories follow the standard 10x layout
    (matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz) vs. AnnData h5ad directly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults (cluster paths; override via CLI)
# ---------------------------------------------------------------------------
RAW_DIR = "/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary/raw"
BASE_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes"

N_PER_CELL_TYPE = 30
MIN_CELLS_THRESHOLD = 10
N_PSEUDO_TUBES = 10
N_HVGS = 4000
RANDOM_SEED = 42

# Leiden resolution for PBS cell clustering.
# N_LEIDEN_CLUSTERS_TARGET = 6 (expected 4–8 clusters at resolution=0.5).
LEIDEN_RESOLUTION = 0.5

# Val mouse selection strategy at build time.
VAL_MOUSE_SELECTION = "outlier_pbs_pca"

# Strings in the cytokine/stimulus column that indicate PBS-injected controls.
# TODO: verify against actual xlsx after download.
PBS_CONTROL_STRINGS = {"PBS", "vehicle", "untreated", "control", "pbs"}

# xlsx mapping file (within RAW_DIR)
XLSX_FILENAME = "GSE202186_map-scRNAseq-cytokines-dictionary.xlsx"
SOFT_FILENAME = "GSE202186_family.soft.gz"

# Extracted MTX directory (within RAW_DIR)
EXTRACTED_SUBDIR = "extracted"


# ===========================================================================
# Raw data inspection helper
# ===========================================================================

def _inspect_raw_layout(raw_dir: str) -> None:
    """
    Print a structured summary of what is present in raw_dir and its
    extracted/ subdir. Dies with a clear message if critical files are absent.

    TODO: update expected layout once GSE202186_RAW.tar has been extracted
    and the actual directory structure is known.
    """
    raw_dir = Path(raw_dir)
    print("=== Raw directory layout inspection ===")
    top_level = sorted(raw_dir.iterdir()) if raw_dir.exists() else []
    print(f"  raw_dir: {raw_dir} ({'exists' if raw_dir.exists() else 'MISSING'})")
    for p in top_level:
        print(f"    {p.name}/") if p.is_dir() else print(f"    {p.name}")

    extracted_dir = raw_dir / EXTRACTED_SUBDIR
    if extracted_dir.exists():
        gsm_dirs = sorted(p for p in extracted_dir.iterdir() if p.is_dir())
        print(f"\n  extracted/ contains {len(gsm_dirs)} subdirectories:")
        for d in gsm_dirs[:5]:
            contents = [f.name for f in d.iterdir()]
            print(f"    {d.name}/: {contents[:6]}")
        if len(gsm_dirs) > 5:
            print(f"    ... ({len(gsm_dirs) - 5} more)")
    else:
        raise FileNotFoundError(
            f"Expected extracted subdirectory at {extracted_dir}. "
            f"Extract GSE202186_RAW.tar first:\n"
            f"  mkdir -p {extracted_dir} && tar -xf {raw_dir}/GSE202186_RAW.tar -C {extracted_dir}"
        )

    xlsx_path = raw_dir / XLSX_FILENAME
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Missing xlsx mapping file: {xlsx_path}\n"
            f"Expected: {XLSX_FILENAME} in {raw_dir}"
        )
    print(f"\n  xlsx mapping: {xlsx_path} (exists)")
    print("=== End layout inspection ===\n")


# ===========================================================================
# Metadata loading (xlsx + SOFT)
# ===========================================================================

def _load_xlsx_metadata(raw_dir: str) -> pd.DataFrame:
    """
    Load the cytokine-to-GSM mapping from the xlsx file.

    Expected columns (TODO: verify after download):
      - GSM_id   : str   — e.g., "GSM6085620"
      - cytokine : str   — e.g., "IL-2", "PBS"
      - mouse_id : str   — e.g., "mouse_1", "mouse_2", "mouse_3"

    If the actual column names differ, raises a clear KeyError with the
    columns present so the TODO can be resolved.
    """
    xlsx_path = Path(raw_dir) / XLSX_FILENAME
    df = pd.read_excel(str(xlsx_path))
    print(f"  xlsx columns found: {list(df.columns)}")

    # TODO: replace with actual column names once download is verified.
    required = {"GSM_id", "cytokine", "mouse_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"xlsx mapping file is missing required columns: {missing}\n"
            f"Columns present: {list(df.columns)}\n"
            f"TODO: update column-name mapping in _load_xlsx_metadata() once "
            f"actual file structure is known."
        )
    return df[["GSM_id", "cytokine", "mouse_id"]].copy()


def _parse_soft_file(raw_dir: str) -> pd.DataFrame:
    """
    Parse GSE202186_family.soft.gz for per-sample metadata.

    The SOFT file is a GEO-standard text format. We extract
    !Sample_geo_accession and !Sample_characteristics_ch1 lines.

    Returns a DataFrame indexed by GSM_id with any additional metadata
    columns extracted. May be redundant with the xlsx; the xlsx is
    authoritative. This function is called only when --use_soft is passed.

    TODO: verify SOFT field names and characteristics structure after download.
    """
    soft_path = Path(raw_dir) / SOFT_FILENAME
    if not soft_path.exists():
        print(f"  SOFT file not found at {soft_path}; skipping SOFT parse.")
        return pd.DataFrame(columns=["GSM_id"])

    records = []
    current: Dict[str, str] = {}
    with gzip.open(str(soft_path), "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE"):
                if current:
                    records.append(current)
                current = {"GSM_id": line.split("=")[-1].strip()}
            elif line.startswith("!Sample_geo_accession"):
                current["GSM_id"] = line.split("=")[-1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                val = line.split("=", 1)[-1].strip()
                if ":" in val:
                    key, v = val.split(":", 1)
                    current[key.strip()] = v.strip()
    if current:
        records.append(current)
    return pd.DataFrame(records)


# ===========================================================================
# Loading 10x MTX data (one GSM = one sample directory)
# ===========================================================================

def _load_one_gsm_10x(gsm_dir: Path) -> ad.AnnData:
    """
    Load one 10x Genomics output directory (standard MTX layout).

    Expected contents (TODO: confirm after extraction):
      matrix.mtx.gz    — sparse count matrix (genes x barcodes)
      barcodes.tsv.gz  — cell barcodes (one per line)
      features.tsv.gz  — gene symbols (two-column: Ensembl id, symbol)

    Returns an AnnData with:
      obs.index = barcodes
      var.index = gene symbols (second column of features.tsv.gz)
      X = raw integer counts (float32)
    """
    import scipy.io
    import scipy.sparse

    # Locate files — filenames may vary slightly across GEO deposits.
    def _find(stem: str) -> Path:
        candidates = list(gsm_dir.glob(f"*{stem}*"))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find '{stem}' file in {gsm_dir}.\n"
                f"Contents: {[p.name for p in gsm_dir.iterdir()]}\n"
                f"TODO: inspect actual 10x output layout after extraction."
            )
        return candidates[0]

    matrix_path = _find("matrix.mtx")
    barcodes_path = _find("barcodes.tsv")
    features_path = _find("features.tsv")

    # Read barcodes
    with (gzip.open(str(barcodes_path), "rt") if str(barcodes_path).endswith(".gz")
          else open(str(barcodes_path))) as f:
        barcodes = [line.strip() for line in f if line.strip()]

    # Read features
    with (gzip.open(str(features_path), "rt") if str(features_path).endswith(".gz")
          else open(str(features_path))) as f:
        feature_rows = [line.strip().split("\t") for line in f if line.strip()]
    # Standard 10x: col0=Ensembl id, col1=gene symbol, col2=feature_type (optional)
    gene_symbols = [row[1] if len(row) > 1 else row[0] for row in feature_rows]

    # Read matrix (genes x barcodes in MTX; transpose to cells x genes)
    with (gzip.open(str(matrix_path), "rb") if str(matrix_path).endswith(".gz")
          else open(str(matrix_path), "rb")) as f:
        mat = scipy.io.mmread(f).T.tocsr()  # cells x genes

    X = mat.toarray().astype(np.float32)
    obs = pd.DataFrame(index=barcodes)
    obs.index.name = None
    var = pd.DataFrame(index=gene_symbols)
    var.index.name = None
    return ad.AnnData(X=X, obs=obs, var=var)


def load_id_anndata(raw_dir: str, gsm_metadata: pd.DataFrame) -> ad.AnnData:
    """
    Load all GSM samples from the extracted raw directory and assemble into
    a single AnnData with obs[GSM_id, cytokine, mouse_id, cell_barcode].

    PBS-injected controls are identified by the cytokine column (strings in
    PBS_CONTROL_STRINGS). They are NOT relabeled here — relabeling happens in
    `relabel_to_pbs()` so the boundary is explicit.

    Args:
        raw_dir:      Path to the raw directory (contains extracted/ subdir).
        gsm_metadata: DataFrame from _load_xlsx_metadata with columns
                      [GSM_id, cytokine, mouse_id].

    Returns assembled AnnData (raw counts, float32).
    """
    extracted_dir = Path(raw_dir) / EXTRACTED_SUBDIR
    gsm_dirs = sorted(p for p in extracted_dir.iterdir() if p.is_dir())

    parts = []
    for gsm_dir in gsm_dirs:
        gsm_id = gsm_dir.name
        meta_row = gsm_metadata[gsm_metadata["GSM_id"] == gsm_id]
        if meta_row.empty:
            print(f"  WARNING: {gsm_id} not found in xlsx metadata; skipping.")
            continue
        cyt = str(meta_row["cytokine"].iloc[0])
        mouse_id = str(meta_row["mouse_id"].iloc[0])

        print(f"  Loading {gsm_id}: cytokine={cyt}, mouse_id={mouse_id}")
        adata_gsm = _load_one_gsm_10x(gsm_dir)

        # Prefix barcodes with GSM_id to ensure global uniqueness
        adata_gsm.obs.index = [f"{gsm_id}_{bc}" for bc in adata_gsm.obs.index]
        adata_gsm.obs["GSM_id"] = gsm_id
        adata_gsm.obs["cytokine"] = cyt
        adata_gsm.obs["mouse_id"] = mouse_id
        parts.append(adata_gsm)

    if not parts:
        raise RuntimeError(
            "No GSM directories matched the xlsx metadata. "
            "Check that EXTRACTED_SUBDIR names match GSM_id values in the xlsx."
        )

    adata = ad.concat(parts, join="inner", label=None)
    adata.obs_names_make_unique()
    print(f"  Assembled: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


# ===========================================================================
# Preprocessing
# ===========================================================================

def preprocess_id(adata: ad.AnnData, n_hvgs: int = N_HVGS) -> ad.AnnData:
    """
    Apply the standard preprocessing pipeline to the ID AnnData.

    Order (per CLAUDE.md §3):
      1. Preserve raw counts in adata.layers["counts"] (required by seurat_v3 HVG).
      2. normalize_total (target_sum=1e4).
      3. log1p.
      4. HVG selection: 4000, seurat_v3, batch_key='mouse_id'.

    Returns filtered AnnData (only HVG columns retained).
    """
    import scanpy as sc

    # 1) Preserve raw integer counts before normalization (seurat_v3 needs them)
    adata.layers["counts"] = adata.X.copy()

    # 2) Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)

    # 3) log1p
    sc.pp.log1p(adata)

    # 4) HVG selection using raw counts stored in the layer
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvgs,
        flavor="seurat_v3",
        batch_key="mouse_id",
        layer="counts",   # seurat_v3 expects raw counts
    )
    n_hvg = adata.var["highly_variable"].sum()
    print(f"  HVG selection: {n_hvg} / {adata.n_vars} genes marked as highly variable")
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


# ===========================================================================
# Cell-type clustering (global Leiden on PBS cells)
# ===========================================================================

def assign_cell_types_global_leiden(
    adata: ad.AnnData,
    resolution: float = LEIDEN_RESOLUTION,
    n_pcs: int = 50,
) -> Tuple[ad.AnnData, np.ndarray]:
    """
    Global Leiden cell typing using PBS-injected control cells as reference.

    1. Subset to cells with cytokine in PBS_CONTROL_STRINGS (post-relabel,
       these will all be "PBS"; if called before relabel, still works by
       matching the control strings).
    2. PCA on PBS subset.
    3. Leiden cluster on KNN graph.
    4. Compute per-cluster centroids in PCA space.
    5. Assign every cell in the full AnnData to the nearest cluster centroid.

    Writes adata.obs["cell_type"] as "id_c{k}".
    Returns (adata_with_cell_types, pca_centroids) where pca_centroids has
    shape (n_clusters, n_pcs).
    """
    import scanpy as sc

    pbs_mask = adata.obs["cytokine"].isin(PBS_CONTROL_STRINGS) | (adata.obs["cytokine"] == "PBS")
    if pbs_mask.sum() == 0:
        raise RuntimeError(
            "No PBS/control cells found for Leiden clustering. "
            "Check that PBS_CONTROL_STRINGS matches the cytokine column values "
            f"(unique values seen: {adata.obs['cytokine'].unique()[:10]})"
        )

    pbs_subset = adata[pbs_mask.values].copy()
    print(f"  PBS subset for Leiden: {pbs_subset.n_obs} cells")

    n_comps = min(n_pcs, pbs_subset.n_vars - 1, pbs_subset.n_obs - 1)
    sc.pp.pca(pbs_subset, n_comps=n_comps)
    sc.pp.neighbors(pbs_subset, n_neighbors=15)
    sc.tl.leiden(pbs_subset, resolution=resolution, key_added="leiden_pbs")

    cluster_ids = pbs_subset.obs["leiden_pbs"].astype(int).values
    n_clusters = len(set(cluster_ids))
    print(f"  Leiden clusters at resolution={resolution}: {n_clusters} clusters")

    # Centroids in PCA space
    pca_components = pbs_subset.varm["PCs"]           # (n_genes, n_comps)
    pca_mean = np.asarray(pbs_subset.X).mean(axis=0)  # (n_genes,)
    pca_mean = np.asarray(pca_mean).ravel()
    sub_pca = (np.asarray(pbs_subset.X) - pca_mean) @ pca_components

    centroids = np.stack(
        [sub_pca[cluster_ids == k].mean(axis=0) for k in sorted(set(cluster_ids))]
    )  # (n_clusters, n_comps)

    # Project every cell onto the same PCA basis
    X_full = np.asarray(adata.X)
    X_centered = X_full - pca_mean
    full_pca = X_centered @ pca_components  # (n_cells, n_comps)

    # Nearest centroid assignment
    dists = ((full_pca[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = dists.argmin(axis=1)
    adata.obs["cell_type"] = [f"id_c{int(k)}" for k in nearest]

    cluster_sizes = {f"id_c{k}": int((nearest == k).sum()) for k in sorted(set(cluster_ids))}
    print(f"  Cluster sizes (all cells): {cluster_sizes}")
    return adata, centroids


# ===========================================================================
# PBS relabeling (adapter boundary contract)
# ===========================================================================

def relabel_to_pbs(adata: ad.AnnData) -> ad.AnnData:
    """
    Relabel all PBS-injected control cells to the literal string "PBS".

    The cytokine-MIL pipeline pins PBS to index 90 (label_encoder.py:11) and
    analysis/pbs_rc.py:59 hard-checks `cytokine == "PBS"`. Relabeling at the
    adapter boundary keeps the package code untouched.

    PBS cells are identified by the cytokine column matching PBS_CONTROL_STRINGS.
    Raises if no PBS cells are found.
    """
    cyt = adata.obs["cytokine"].astype(str).copy()
    is_pbs = cyt.str.strip().str.lower().isin(
        {s.lower() for s in PBS_CONTROL_STRINGS}
    )
    if is_pbs.sum() == 0:
        unique_vals = adata.obs["cytokine"].unique()[:20]
        raise RuntimeError(
            f"No PBS-control cells detected. "
            f"PBS_CONTROL_STRINGS={PBS_CONTROL_STRINGS}; "
            f"unique cytokine values: {unique_vals}\n"
            f"TODO: update PBS_CONTROL_STRINGS to match the actual control label."
        )
    cyt.loc[is_pbs.values] = "PBS"
    adata.obs["cytokine"] = cyt.values
    n_pbs = int(is_pbs.sum())
    print(f"  Relabeled {n_pbs} cells to PBS")
    return adata


# ===========================================================================
# Val mouse selection
# ===========================================================================

def select_val_mouse(
    adata: ad.AnnData,
    pbs_centroids: np.ndarray,
    strategy: str = VAL_MOUSE_SELECTION,
) -> str:
    """
    Select the validation mouse using the specified strategy.

    Strategy 'outlier_pbs_pca':
      1. Compute PCA on all PBS cells pooled (using centroids already computed).
      2. For each mouse, take the mean PBS cell embedding in PCA space.
      3. Pick the mouse with the largest Euclidean distance from the cross-mouse
         mean PBS embedding.

    Returns the mouse_id string of the chosen val mouse.
    """
    pbs_mask = adata.obs["cytokine"] == "PBS"
    pbs_adata = adata[pbs_mask.values]

    # Project PBS cells onto the same PCA basis (already encoded in X as log-norm)
    # We re-use the per-cluster centroid coordinates stored in pbs_centroids to
    # compute a per-mouse summary: mean cluster assignment in PCA space.
    # Simpler: project each PBS cell using the already-fit PCA (stored in adata.obsm
    # if scanpy was run; otherwise fall back to raw X mean).
    if "X_pca" in pbs_adata.obsm:
        pca_coords = pbs_adata.obsm["X_pca"]  # (n_pbs_cells, n_pcs)
    else:
        # Fall back: use raw gene expression mean per mouse as summary
        pca_coords = np.asarray(pbs_adata.X)

    mouse_ids = pbs_adata.obs["mouse_id"].values
    unique_mice = sorted(set(mouse_ids))

    per_mouse_mean = {
        m: pca_coords[mouse_ids == m].mean(axis=0) for m in unique_mice
    }
    global_mean = np.stack(list(per_mouse_mean.values())).mean(axis=0)
    distances = {
        m: float(np.linalg.norm(per_mouse_mean[m] - global_mean))
        for m in unique_mice
    }
    val_mouse = max(distances, key=distances.__getitem__)
    print(f"  Val mouse selection ({strategy}):")
    for m, d in sorted(distances.items(), key=lambda x: -x[1]):
        marker = " <- VAL" if m == val_mouse else ""
        print(f"    {m}: dist={d:.4f}{marker}")
    return val_mouse


# ===========================================================================
# Pseudo-tube construction
# ===========================================================================

def build_pseudo_tubes_id(
    adata: ad.AnnData,
    base_path: str,
    n_per_cell_type: int = N_PER_CELL_TYPE,
    min_cells_threshold: int = MIN_CELLS_THRESHOLD,
    n_pseudo_tubes: int = N_PSEUDO_TUBES,
    limit_cytokines: Optional[List[str]] = None,
    limit_mice: Optional[List[str]] = None,
    skip_existing: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> list:
    """
    Build pseudo-tubes from the ID AnnData and write them under base_path.

    Required obs columns on `adata`:
      - mouse_id  : str  — used as manifest "donor" key
      - cytokine  : str  — PBS-relabeled; this function does NOT relabel
      - cell_type : str  — e.g., "id_c0" — used for stratified sampling

    Manifest schema (per PseudoTubeDataset contract):
      {path, donor, cytokine, n_cells, cell_types_included, tube_idx}

    Also writes both a preprocessed .h5ad (HVG-filtered, log-normalized)
    and a raw-count .h5ad per tube (preserving adata.layers["counts"] if
    present). Raw .h5ad written as pseudotube_{tube_idx}_raw.h5ad.

    Returns the manifest list (also written to <base_path>/manifest.json).
    """
    import scanpy as sc

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    donor_col = "mouse_id"
    cond_col = "cytokine"
    ct_col = "cell_type"

    mice = sorted(adata.obs[donor_col].unique())
    if limit_mice:
        mice = [m for m in mice if m in limit_mice]

    cytokines = sorted(adata.obs[cond_col].unique())
    if limit_cytokines:
        cytokines = [c for c in cytokines if c in limit_cytokines]

    has_raw_layer = "counts" in adata.layers

    manifest = []
    for mouse_id in mice:
        adata_mouse = adata[adata.obs[donor_col] == mouse_id].copy()

        for cyt in cytokines:
            adata_cond = adata_mouse[adata_mouse.obs[cond_col] == cyt]
            if adata_cond.n_obs == 0:
                continue

            eligible_cell_types = [
                ct
                for ct in adata_cond.obs[ct_col].unique()
                if (adata_cond.obs[ct_col] == ct).sum() >= min_cells_threshold
            ]
            if not eligible_cell_types:
                print(f"  Skipping {mouse_id}/{cyt}: no eligible cell types")
                continue

            safe_cyt = str(cyt).replace(" ", "_").replace("/", "_")
            folder = base_path / mouse_id / safe_cyt
            folder.mkdir(parents=True, exist_ok=True)

            for tube_idx in range(n_pseudo_tubes):
                out_path = folder / f"pseudotube_{tube_idx}.h5ad"
                raw_path = folder / f"pseudotube_{tube_idx}_raw.h5ad"

                if skip_existing and out_path.exists():
                    print(f"  Skipping (exists): {out_path}")
                    # Still need to add to manifest
                    existing = ad.read_h5ad(str(out_path))
                    manifest.append({
                        "path": str(out_path),
                        "donor": mouse_id,
                        "cytokine": cyt,
                        "n_cells": int(existing.n_obs),
                        "cell_types_included": list(existing.obs[ct_col].unique().astype(str)),
                        "tube_idx": tube_idx,
                    })
                    continue

                pseudo_tube = _sample_one_tube(
                    adata_cond, eligible_cell_types, ct_col, n_per_cell_type, rng
                )
                pseudo_tube.obs["donor"] = mouse_id
                pseudo_tube.obs["cytokine"] = cyt

                # Write preprocessed (HVG-filtered, log-normalized) tube
                pseudo_tube.write_h5ad(str(out_path), compression="gzip")

                # Write raw-count tube (using layers["counts"] if available)
                if has_raw_layer:
                    raw_tube = pseudo_tube.copy()
                    raw_tube.X = raw_tube.layers["counts"].copy()
                    raw_tube.write_h5ad(str(raw_path), compression="gzip")

                manifest.append({
                    "path": str(out_path),
                    "donor": mouse_id,
                    "cytokine": cyt,
                    "n_cells": int(pseudo_tube.n_obs),
                    "cell_types_included": list(map(str, eligible_cell_types)),
                    "tube_idx": tube_idx,
                })

    manifest_path = base_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _sample_one_tube(
    adata_cond: ad.AnnData,
    eligible_cell_types: list,
    ct_col: str,
    n_per_cell_type: int,
    rng: np.random.Generator,
) -> ad.AnnData:
    """Sample one pseudo-tube: stratified by cell type, shuffled, float32."""
    import scanpy as sc

    sampled = []
    for ct in eligible_cell_types:
        ct_cells = adata_cond[adata_cond.obs[ct_col] == ct]
        n_sample = min(n_per_cell_type, ct_cells.n_obs)
        chosen = rng.choice(ct_cells.n_obs, size=n_sample, replace=False)
        sampled.append(ct_cells[chosen].copy())

    tube = sc.concat(sampled, join="outer")
    perm = rng.permutation(tube.n_obs)
    tube = tube[perm].copy()

    if hasattr(tube.X, "toarray"):
        X = tube.X.toarray().astype(np.float32)
    else:
        X = np.asarray(tube.X, dtype=np.float32)
    tube.X = X
    return tube


# ===========================================================================
# Run summary
# ===========================================================================

def _print_run_summary(
    adata: ad.AnnData,
    manifest: list,
    val_mouse: str,
    n_cells_before_hvg: int,
) -> None:
    """Print a structured run summary after build completes."""
    mice = sorted(adata.obs["mouse_id"].unique())
    cytokines = sorted(adata.obs["cytokine"].unique())
    cell_types = sorted(adata.obs["cell_type"].unique())

    print("\n" + "=" * 60)
    print("RUN SUMMARY — Immune Dictionary pseudo-tube build")
    print("=" * 60)
    print(f"  Total mice:            {len(mice)}")
    print(f"  Total cytokines:       {len(cytokines)} (including PBS)")
    print(f"  Total pseudo-tubes:    {len(manifest)}")
    print(f"  Cells before HVG:      {n_cells_before_hvg}")
    print(f"  Cells after HVG:       {adata.n_obs} x {adata.n_vars} genes")
    print(f"  Leiden clusters:       {cell_types}")
    print(f"  Val mouse:             {val_mouse}  (strategy: {VAL_MOUSE_SELECTION})")

    # Per-cell-type distribution across cytokines
    print("\n  Per-cluster cell counts across cytokines (train donors):")
    train_mice = [m for m in mice if m != val_mouse]
    train_mask = adata.obs["mouse_id"].isin(train_mice)
    for ct in cell_types:
        ct_mask = adata.obs["cell_type"] == ct
        for cyt in cytokines:
            cyt_mask = adata.obs["cytokine"] == cyt
            n = int((train_mask & ct_mask & cyt_mask).sum())
            if n < MIN_CELLS_THRESHOLD:
                print(f"    WARNING: {ct} x {cyt} has only {n} cells in train "
                      f"(< MIN_CELLS_THRESHOLD={MIN_CELLS_THRESHOLD})")
    print("=" * 60)


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build pseudo-tubes from the Immune Dictionary (Cui et al. 2024)."
    )
    p.add_argument("--raw_dir", default=RAW_DIR,
                   help="Dir containing GSE202186_RAW.tar (extracted/) and xlsx mapping")
    p.add_argument("--base_path", default=BASE_PATH,
                   help="Output directory for pseudo-tubes and manifest")
    p.add_argument("--n_per_cell_type", type=int, default=N_PER_CELL_TYPE)
    p.add_argument("--min_cells_threshold", type=int, default=MIN_CELLS_THRESHOLD)
    p.add_argument("--n_pseudo_tubes", type=int, default=N_PSEUDO_TUBES)
    p.add_argument("--n_hvgs", type=int, default=N_HVGS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--limit_cytokines", type=str, default=None,
                   help="Comma-separated cytokine subset for testing, e.g. 'IL-2,IL-6,PBS'")
    p.add_argument("--limit_mice", type=str, default=None,
                   help="Comma-separated mouse_id subset for testing")
    p.add_argument("--skip_existing", action="store_true",
                   help="Do not overwrite existing pseudo-tube .h5ad files")
    p.add_argument("--use_soft", action="store_true",
                   help="Also parse the SOFT file (usually redundant with xlsx)")
    p.add_argument("--leiden_resolution", type=float, default=LEIDEN_RESOLUTION)
    return p.parse_args()


def main() -> None:
    import scanpy as sc

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.base_path, exist_ok=True)

    limit_cytokines = [c.strip() for c in args.limit_cytokines.split(",")] \
        if args.limit_cytokines else None
    limit_mice = [m.strip() for m in args.limit_mice.split(",")] \
        if args.limit_mice else None

    print("Step 0: Inspect raw directory layout")
    _inspect_raw_layout(args.raw_dir)

    print("Step 1: Load xlsx metadata")
    gsm_metadata = _load_xlsx_metadata(args.raw_dir)
    print(f"  {len(gsm_metadata)} GSM entries in xlsx")

    if args.use_soft:
        print("Step 1b: Parse SOFT file (supplementary; xlsx is authoritative)")
        _parse_soft_file(args.raw_dir)

    print("\nStep 2: Load raw 10x count data from extracted/ subdir")
    adata = load_id_anndata(args.raw_dir, gsm_metadata)
    n_cells_before_hvg = adata.n_obs

    print("\nStep 3: QC (min_genes=200 — whole-transcriptome 10x)")
    sc.pp.filter_cells(adata, min_genes=200)
    print(f"  After cell QC: {adata.n_obs} cells")

    print("\nStep 4: Relabel PBS-control cells to 'PBS'")
    adata = relabel_to_pbs(adata)
    print(f"  Active classes after relabel: {sorted(adata.obs['cytokine'].unique())}")

    if not (adata.obs["cytokine"] == "PBS").any():
        raise RuntimeError("No PBS cells after relabeling — cannot proceed.")

    print("\nStep 5: Preprocess (normalize_total → log1p → HVG 4000, seurat_v3)")
    adata = preprocess_id(adata, n_hvgs=args.n_hvgs)
    print(f"  After HVG filter: {adata.n_obs} cells x {adata.n_vars} genes")

    print("\nStep 6: Save HVG list")
    hvg_path = os.path.join(args.base_path, "hvg_list.json")
    with open(hvg_path, "w") as f:
        json.dump(list(adata.var_names), f, indent=2)
    print(f"  Wrote {hvg_path}")

    print("\nStep 7: Global Leiden cell typing on PBS cells")
    adata, pbs_centroids = assign_cell_types_global_leiden(
        adata, resolution=args.leiden_resolution
    )
    # Save PBS centroids for downstream PBS-RC computation
    centroids_path = os.path.join(args.base_path, "pbs_centroids.npy")
    np.save(centroids_path, pbs_centroids)
    print(f"  Saved PBS PCA centroids to {centroids_path}")

    print("\nStep 8: Select val mouse")
    val_mouse = select_val_mouse(adata, pbs_centroids)
    # Record val mouse in a manifest metadata file
    meta_path = os.path.join(args.base_path, "build_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"val_mouse": val_mouse, "val_mouse_strategy": VAL_MOUSE_SELECTION}, f, indent=2)
    print(f"  Val mouse: {val_mouse}. Wrote to {meta_path}")

    print("\nStep 9: Build pseudo-tubes")
    manifest = build_pseudo_tubes_id(
        adata,
        base_path=args.base_path,
        n_per_cell_type=args.n_per_cell_type,
        min_cells_threshold=args.min_cells_threshold,
        n_pseudo_tubes=args.n_pseudo_tubes,
        limit_cytokines=limit_cytokines,
        limit_mice=limit_mice,
        skip_existing=args.skip_existing,
        rng=rng,
    )

    _print_run_summary(adata, manifest, val_mouse, n_cells_before_hvg)
    print(f"\nDone. {len(manifest)} pseudo-tubes written under {args.base_path}")


if __name__ == "__main__":
    main()
