"""
Build pseudo-tubes for Oelen et al. 2022 (1M-scBloodNL) human PBMC — the 5th
validation dataset; 121 donors (finally powers the DONOR-LEVEL coupling gate).

Source: eQTLGen/molgenis (free; raw reads/genotypes EGA-gated). raw/:
  10x_v2_RNA_matrix.mtx.gz / 10x_v2_RNA_features.tsv.gz / 10x_v2_barcodes.tsv.gz
  10x_v3_RNA_matrix.mtx.gz / 10x_v3_RNA_features.tsv.gz / 10x_v3_barcodes.tsv.gz
  1M_assignments_conditions_expid.tsv  (barcode, assignment, timepoint, chem)
  1M_cell_types.tsv                    (barcode, cell_type, cell_type_lowerres)

Knob mapping (method data contract):
  - donor     = "<chem>_<assignment>"  (assignment is genotype-demuxed PER chem, so
                the composite is the true individual; ~121 donors).
  - cytokine  = the `timepoint` label (3hCA/24hCA/3hPA/24hPA/3hMTB/24hMTB); UT -> "PBS".
                (Keeping pathogen x time as the class lets BOTH the coupling gate and
                 the 3h->24h progression-direction run off one build.) obs also carries
                'pathogen' and 'time' for the progression analysis.
  - cell_type = cell_type_lowerres (CD4T/CD8T/monocyte/NK/B/DC/...); drop 'unknown'.
  - chem      = V2/V3 (HVG batch_key).
  - 4000 HVG (seurat_v3, batch_key=chem); normalize_total -> log1p.

Scale: ~928k cells, 121 donors. To keep the downstream binary AB-MIL + per-donor
coupling tractable, N_PSEUDO_TUBES is capped low (default 3) and cells are NOT
otherwise subsampled (the per-(donor,cond,celltype) pools are already modest).

Run:  python scripts/build_pseudotubes_oelen2022.py [--inspect_only]
"""
from __future__ import annotations

import argparse
import gzip
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
from build_pseudotubes_immune_dictionary import build_pseudo_tubes_id  # noqa: E402

RAW = "/cs/labs/mornitzan/yam.arieli/datasets/Oelen2022/raw"
BASE_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oelen2022_pseudotubes"

N_PER_CELL_TYPE = 30
MIN_CELLS_THRESHOLD = 10
N_PSEUDO_TUBES = 3
N_HVGS = 4000
RANDOM_SEED = 42
CONTROL_IN = "UT"
CONTROL_OUT = "PBS"
DROP_CELLTYPES = {"unknown"}


def _log(m=""):
    print(m, flush=True)


def _read_lines_gz(path):
    with gzip.open(path, "rt") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _load_chem(raw: Path, chem: str, meta: pd.DataFrame, log):
    """Load one chemistry's mtx (genes x cells) -> AnnData (cells x genes), subset to
    barcodes present in `meta` (this chem's annotated cells)."""
    tag = f"10x_{chem.lower()}"
    mtx = scipy.io.mmread(str(raw / f"{tag}_RNA_matrix.mtx.gz")).tocsr()
    feats = _read_lines_gz(raw / f"{tag}_RNA_features.tsv.gz")
    barcodes = _read_lines_gz(raw / f"{tag}_barcodes.tsv.gz")
    # genes x cells -> cells x genes
    if mtx.shape == (len(feats), len(barcodes)):
        X = mtx.T.tocsr()
    elif mtx.shape == (len(barcodes), len(feats)):
        X = mtx.tocsr()
    else:
        raise ValueError(f"{chem}: mtx {mtx.shape} != (genes,cells)/(cells,genes)")
    X = X.astype(np.float32)

    meta_c = meta.reindex(barcodes)
    keep = meta_c["assignment"].notna().values
    X = X[keep]
    bc = [b for b, k in zip(barcodes, keep) if k]
    meta_c = meta_c.iloc[keep]
    a = ad.AnnData(X=X)
    a.var_names = [f.split("\t")[-1] for f in feats]  # symbol if tab-sep, else raw
    a.var_names_make_unique()
    a.obs_names = [f"{chem}_{b}" for b in bc]
    a.obs["assignment"] = meta_c["assignment"].astype(str).values
    a.obs["timepoint"] = meta_c["timepoint"].astype(str).values
    a.obs["cell_type"] = meta_c["cell_type_lowerres"].astype(str).values
    a.obs["chem"] = chem
    log(f"  {chem}: {a.n_obs} annotated cells x {a.n_vars} genes")
    return a


def _parse_conditions(adata):
    """timepoint -> cytokine (UT->PBS), pathogen, time."""
    tp = adata.obs["timepoint"].astype(str)
    cyt, path, tm = [], [], []
    for v in tp:
        if v == CONTROL_IN:
            cyt.append(CONTROL_OUT); path.append("PBS"); tm.append("UT")
        elif v.startswith("3h"):
            cyt.append(v); path.append(v[2:]); tm.append("3h")
        elif v.startswith("24h"):
            cyt.append(v); path.append(v[3:]); tm.append("24h")
        else:
            cyt.append("DROP"); path.append("DROP"); tm.append("DROP")
    adata.obs["cytokine"] = cyt
    adata.obs["pathogen"] = path
    adata.obs["time"] = tm
    adata.obs["donor"] = adata.obs["chem"].astype(str) + "_" + adata.obs["assignment"].astype(str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--base_path", default=BASE_PATH)
    ap.add_argument("--n_per_cell_type", type=int, default=N_PER_CELL_TYPE)
    ap.add_argument("--n_pseudo_tubes", type=int, default=N_PSEUDO_TUBES)
    ap.add_argument("--n_hvgs", type=int, default=N_HVGS)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--inspect_only", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw)
    base = Path(args.base_path)
    rng = np.random.default_rng(args.seed)

    _log("=== Oelen 1M-scBloodNL pseudo-tube build ===")
    assign = pd.read_csv(raw / "1M_assignments_conditions_expid.tsv", sep="\t",
                         index_col=0)
    ctypes = pd.read_csv(raw / "1M_cell_types.tsv", sep="\t", index_col=0)
    meta = assign.join(ctypes[["cell_type_lowerres"]], how="left")
    meta = meta[meta["timepoint"].notna() & (meta["timepoint"] != "NA")]
    meta = meta[~meta["cell_type_lowerres"].isin(DROP_CELLTYPES)]
    meta = meta[meta["cell_type_lowerres"].notna()]
    _log(f"metadata after filter: {len(meta)} cells")
    _log(f"timepoints: {sorted(meta['timepoint'].unique())}")
    _log(f"chems:      {sorted(meta['chem'].unique())}")
    _log(f"donors (chem_assignment): "
         f"{meta.assign(d=meta['chem'].astype(str)+'_'+meta['assignment'].astype(str))['d'].nunique()}")
    _log(f"cell types: {sorted(meta['cell_type_lowerres'].dropna().unique())}")
    if args.inspect_only:
        _log("--inspect_only; exiting before mtx load.")
        return

    parts = []
    for chem in sorted(meta["chem"].unique()):
        parts.append(_load_chem(raw, chem, meta[meta["chem"] == chem], _log))
    _log("Concatenating chemistries (inner-join genes)...")
    adata = ad.concat(parts, join="inner", merge="same")
    del parts
    _log(f"  combined: {adata.n_obs} cells x {adata.n_vars} genes")
    _parse_conditions(adata)
    adata = adata[adata.obs["cytokine"] != "DROP"].copy()
    _log(f"  conditions: {sorted(adata.obs['cytokine'].unique())}")
    _log(f"  donors:     {adata.obs['donor'].nunique()}")

    adata.layers["counts"] = adata.X.copy()
    _log("Selecting HVGs (seurat_v3, batch_key=chem)...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs, flavor="seurat_v3",
                                    batch_key="chem", layer="counts")
    except Exception as e:
        _log(f"  seurat_v3 batch_key failed ({e}); retry without batch_key")
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs, flavor="seurat_v3",
                                    layer="counts")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    hvg_mask = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask].tolist()
    adata_hvg = adata[:, hvg_mask].copy()
    _log(f"  HVGs: {len(hvg_genes)}")

    base.mkdir(parents=True, exist_ok=True)
    with open(base / "hvg_list.json", "w") as fh:
        json.dump(hvg_genes, fh)

    _log("Building pseudo-tubes per (cytokine, donor)...")
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
        "dataset": "Oelen2022_1M-scBloodNL",
        "source": "eQTLGen/molgenis (processed UMI; raw EGA EGAS00001005376)",
        "control_label": CONTROL_OUT,
        "n_tubes": len(manifest),
        "n_cytokines": len(cyts),
        "n_donors": len(donors),
        "n_hvgs": len(hvg_genes),
        "cytokines": cyts,
    }
    with open(base / "build_metadata.json", "w") as fh:
        json.dump(build_meta, fh, indent=2)
    _log("\n=== RUN SUMMARY ===")
    _log(f"  tubes:      {len(manifest)}")
    _log(f"  conditions: {len(cyts)} -> {cyts}")
    _log(f"  donors:     {len(donors)}")
    _log(f"  HVGs:       {len(hvg_genes)}")
    _log("=== DONE ===")


if __name__ == "__main__":
    main()
