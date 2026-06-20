"""
Build pseudo-tubes for Cano-Gamez et al. 2020 (Nat Commun) CD4+ T-cell cytokine
scRNA-seq — the 4th validation dataset (human, cascade-rich).

Source: BioStudies S-BSST2978 (processed UMI counts; free). 10X 3' v2.
  extracted/scRNAseq/
    NCOMMS-19-7936188_scRNAseq_raw_UMIs.mtx   (genes x cells, raw UMI)
    NCOMMS-19-7936188_scRNAseq_barcodes.tsv   (cells; e.g. N_resting_AAACCTGA...)
    NCOMMS-19-7936188_scRNAseq_genes.tsv      (gene symbols)
    NCOMMS-19-7936188_metadata.txt            (TSV; index=barcode; 12 cols incl
       cytokine.condition, cell.type {Naive,Memory}, donor.id, cluster.id, effectorness)

Knob mapping (method data contract; mirrors Sheu/ID adapters):
  - cytokine = cytokine.condition; the resting/unstim level "UNS" -> literal "PBS"
    (keeps the PBS-index-90 / PBS-RC contract; the cytokine effect sits on top of
    TCR activation, so the shared post-activation program is handled by the §28
    IG_vsPanel signature, not here).
  - cell_type = cell.type (Naive / Memory) -- exists across ALL conditions, so it is
    the valid per-cell-type stratification key (cluster.id is condition-specific and
    is NOT used for stratification).
  - donor = donor.id.
  - 4000 HVG (seurat_v3, batch_key=donor); normalize_total -> log1p.
  - N_PER_CELL_TYPE=50 (only 2 cell types -> bump from 30 to keep tubes ~100 cells),
    N_PSEUDO_TUBES=10, MIN_CELLS_THRESHOLD=10.

Run on the cluster:
    python scripts/build_pseudotubes_cano_gamez.py --inspect_only   # metadata summary
    python scripts/build_pseudotubes_cano_gamez.py                   # full build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# Reuse the generic stratified pseudo-tube builder from the ID adapter.
from build_pseudotubes_immune_dictionary import build_pseudo_tubes_id  # noqa: E402

RAW = "/cs/labs/mornitzan/yam.arieli/datasets/CanoGamez/raw/extracted/scRNAseq"
BASE_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/CanoGamez_pseudotubes"

N_PER_CELL_TYPE = 50
MIN_CELLS_THRESHOLD = 10
N_PSEUDO_TUBES = 10
N_HVGS = 4000
RANDOM_SEED = 42
CONTROL_IN = "UNS"          # resting/unstim label in cytokine.condition
CONTROL_OUT = "PBS"


def _log(m=""):
    print(m, flush=True)


def _load_metadata(raw: Path) -> pd.DataFrame:
    meta = pd.read_csv(raw / "NCOMMS-19-7936188_metadata.txt", sep="\t", index_col=0)
    meta.columns = [c.strip() for c in meta.columns]
    return meta


def _summarize(meta: pd.DataFrame) -> None:
    _log(f"metadata: {meta.shape[0]} cells x {meta.shape[1]} cols")
    _log(f"columns: {list(meta.columns)}")
    for col in ["cytokine.condition", "cell.type", "donor.id", "Phase"]:
        if col in meta.columns:
            _log(f"\n== {col} value_counts ==")
            _log(meta[col].value_counts().to_string())
    if "cluster.id" in meta.columns:
        _log(f"\n== cluster.id ({meta['cluster.id'].nunique()} unique) ==")
        _log(meta["cluster.id"].value_counts().to_string())
    if {"cytokine.condition", "donor.id"} <= set(meta.columns):
        _log("\n== crosstab cytokine.condition x donor.id ==")
        _log(pd.crosstab(meta["cytokine.condition"], meta["donor.id"]).to_string())
    if {"cytokine.condition", "cell.type"} <= set(meta.columns):
        _log("\n== crosstab cytokine.condition x cell.type ==")
        _log(pd.crosstab(meta["cytokine.condition"], meta["cell.type"]).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--base_path", default=BASE_PATH)
    ap.add_argument("--n_per_cell_type", type=int, default=N_PER_CELL_TYPE)
    ap.add_argument("--n_pseudo_tubes", type=int, default=N_PSEUDO_TUBES)
    ap.add_argument("--n_hvgs", type=int, default=N_HVGS)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--val_donor", default=None,
                    help="Held-out donor; default = donor with fewest cells (set at runtime).")
    ap.add_argument("--inspect_only", action="store_true",
                    help="Print the metadata summary and exit (no mtx load / build).")
    args = ap.parse_args()

    raw = Path(args.raw)
    base = Path(args.base_path)
    rng = np.random.default_rng(args.seed)

    _log("=== Cano-Gamez CD4 T-cell pseudo-tube build ===")
    _log(f"raw:       {raw}")
    _log(f"base_path: {base}")

    meta = _load_metadata(raw)
    _summarize(meta)
    if args.inspect_only:
        _log("\n--inspect_only set; exiting before mtx load.")
        return

    # ------------------------------------------------------------------
    # Load mtx (genes x cells) + barcodes + genes -> AnnData (cells x genes)
    # ------------------------------------------------------------------
    _log("\nLoading mtx (genes x cells)...")
    mat = scipy.io.mmread(str(raw / "NCOMMS-19-7936188_scRNAseq_raw_UMIs.mtx")).tocsr()
    with open(raw / "NCOMMS-19-7936188_scRNAseq_barcodes.tsv") as fh:
        barcodes = [ln.strip() for ln in fh if ln.strip()]
    with open(raw / "NCOMMS-19-7936188_scRNAseq_genes.tsv") as fh:
        genes = [ln.strip() for ln in fh if ln.strip()]
    _log(f"  mtx shape {mat.shape}; barcodes {len(barcodes)}; genes {len(genes)}")
    # cellranger mtx is genes x cells -> transpose to cells x genes
    if mat.shape == (len(genes), len(barcodes)):
        X = mat.T.tocsr()
    elif mat.shape == (len(barcodes), len(genes)):
        X = mat.tocsr()
    else:
        _log(f"FATAL: mtx shape {mat.shape} matches neither (genes,cells) nor (cells,genes)")
        sys.exit(2)
    X = X.astype(np.float32)

    # Align metadata to the barcode order; keep only cells present in both.
    meta = meta.reindex(barcodes)
    keep = meta["cytokine.condition"].notna().values
    if not keep.all():
        _log(f"  dropping {int((~keep).sum())} cells absent from metadata")
        X = X[keep]
        barcodes = [b for b, k in zip(barcodes, keep) if k]
        meta = meta.iloc[keep]

    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.var_names_make_unique()
    adata.obs_names = barcodes
    adata.obs["cytokine"] = (meta["cytokine.condition"].astype(str)
                             .replace({CONTROL_IN: CONTROL_OUT}).values)
    adata.obs["cell_type"] = meta["cell.type"].astype(str).values
    adata.obs["donor"] = meta["donor.id"].astype(str).values
    _log(f"\nAnnData: {adata.n_obs} cells x {adata.n_vars} genes")
    _log(f"  conditions: {sorted(adata.obs['cytokine'].unique())}")
    _log(f"  cell types: {sorted(adata.obs['cell_type'].unique())}")
    _log(f"  donors:     {sorted(adata.obs['donor'].unique())}")

    # ------------------------------------------------------------------
    # Preprocess: HVG (seurat_v3 on raw counts) -> normalize -> log1p
    # ------------------------------------------------------------------
    adata.layers["counts"] = adata.X.copy()
    _log("\nSelecting HVGs (seurat_v3, batch_key=donor)...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs, flavor="seurat_v3",
                                    batch_key="donor", layer="counts")
    except Exception as e:
        _log(f"  seurat_v3 batch_key failed ({e}); retry without batch_key")
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs, flavor="seurat_v3",
                                    layer="counts")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    hvg_mask = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask].tolist()
    adata_hvg = adata[:, hvg_mask].copy()
    _log(f"  HVGs selected: {len(hvg_genes)}")

    base.mkdir(parents=True, exist_ok=True)
    with open(base / "hvg_list.json", "w") as fh:
        json.dump(hvg_genes, fh)

    # val donor: smallest by cell count unless overridden
    counts_per_donor = adata_hvg.obs["donor"].value_counts()
    val_donor = args.val_donor or counts_per_donor.idxmin()
    _log(f"\nval_donor = {val_donor}  (per-donor cells: {counts_per_donor.to_dict()})")

    # ------------------------------------------------------------------
    # Build pseudo-tubes (reuse generic ID builder)
    # ------------------------------------------------------------------
    _log("\nBuilding pseudo-tubes per (cytokine, donor)...")
    manifest = build_pseudo_tubes_id(
        adata_hvg, base,
        n_per_cell_type=args.n_per_cell_type,
        min_cells_threshold=MIN_CELLS_THRESHOLD,
        n_pseudo_tubes=args.n_pseudo_tubes,
        rng=rng,
        donor_col="donor", cytokine_col="cytokine", celltype_col="cell_type",
    )
    cyts = sorted({m["cytokine"] for m in manifest})
    donors = sorted({m["donor"] for m in manifest})
    build_meta = {
        "dataset": "CanoGamez2020_CD4Tcell",
        "source": "BioStudies S-BSST2978 (processed UMI; EGA raw EGAS00001003215)",
        "control_label": CONTROL_OUT,
        "val_donor": val_donor,
        "n_tubes": len(manifest),
        "n_cytokines": len(cyts),
        "donors": donors,
        "n_hvgs": len(hvg_genes),
        "cytokines": cyts,
    }
    with open(base / "build_metadata.json", "w") as fh:
        json.dump(build_meta, fh, indent=2)

    _log("\n=== RUN SUMMARY ===")
    _log(f"  tubes:      {len(manifest)}")
    _log(f"  conditions: {len(cyts)} -> {cyts} (PBS in: {'PBS' in cyts})")
    _log(f"  donors:     {donors}")
    _log(f"  HVGs:       {len(hvg_genes)}")
    _log(f"  val donor:  {val_donor}")
    _log("=== DONE ===")


if __name__ == "__main__":
    main()
