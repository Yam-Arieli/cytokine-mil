#!/usr/bin/env python
"""Assemble the flat files from convert_vaccine_rds_to_h5ad.R into an AnnData.

cascadir needs an AnnData; the R step dumped the Seurat object to MatrixMarket +
CSVs. This reads them, builds `cells × genes` RNA X, stashes the ADT surface
proteins in `obsm["protein"]` (+ `uns["protein_names"]`), and writes a single
raw h5ad that prepare_vaccine_tcell.py consumes.

Usage:
  python scripts/assemble_vaccine_h5ad.py \
      --flat_dir /cs/.../SARSCoV2_Vaccine/raw/flat \
      --out      /cs/.../SARSCoV2_Vaccine/raw/vaccine_cite_raw.h5ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix

FLAT_DEFAULT = "/cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/raw/flat"
OUT_DEFAULT = "/cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/raw/vaccine_cite_raw.h5ad"


def _read_lines(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in path.read_text().splitlines() if ln.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat_dir", default=FLAT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    flat = Path(args.flat_dir)

    genes = _read_lines(flat / "genes.csv")
    barcodes = _read_lines(flat / "barcodes.csv")
    print(f"[load] {len(genes)} genes × {len(barcodes)} cells")

    # counts.mtx is genes × cells (R/Seurat convention) → transpose to cells × genes.
    counts = mmread(str(flat / "counts.mtx")).tocsr()
    if counts.shape == (len(genes), len(barcodes)):
        X = counts.T.tocsr()
    elif counts.shape == (len(barcodes), len(genes)):
        X = counts.tocsr()
    else:
        raise SystemExit(
            f"counts.mtx shape {counts.shape} matches neither (genes,cells)="
            f"{(len(genes), len(barcodes))} nor its transpose.")
    X = csr_matrix(X, dtype=np.float32)
    print(f"[rna] X = {X.shape} (cells × genes); min={X.min():.3g} max={X.max():.3g}")

    # meta.csv: first column 'barcode', already row-ordered to the barcode order.
    meta = pd.read_csv(flat / "meta.csv")
    if "barcode" not in meta.columns:
        raise SystemExit("meta.csv missing 'barcode' column (R step contract).")
    meta = meta.set_index("barcode")
    # Align meta to the matrix barcode order (defensive — R already ordered it).
    if list(meta.index) != barcodes:
        missing = [b for b in barcodes if b not in meta.index]
        if missing:
            raise SystemExit(f"{len(missing)} matrix barcodes absent from meta.csv "
                             f"(e.g. {missing[:3]}).")
        meta = meta.loc[barcodes]
    print(f"[meta] {meta.shape[0]} cells × {meta.shape[1]} columns: {list(meta.columns)}")

    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    adata = ad.AnnData(X=X, obs=meta, var=var)
    adata.obs_names = barcodes

    # ADT surface protein → obsm["protein"] (cells × proteins) + uns["protein_names"].
    adt_path = flat / "adt.mtx"
    if adt_path.exists():
        adt_names = _read_lines(flat / "adt_names.csv")
        adt = mmread(str(adt_path)).tocsr()
        if adt.shape == (len(adt_names), len(barcodes)):
            P = adt.T.toarray()
        elif adt.shape == (len(barcodes), len(adt_names)):
            P = adt.toarray()
        else:
            raise SystemExit(f"adt.mtx shape {adt.shape} matches neither orientation.")
        adata.obsm["protein"] = np.asarray(P, dtype=np.float32)
        adata.uns["protein_names"] = list(adt_names)
        print(f"[adt] obsm['protein'] = {adata.obsm['protein'].shape}; "
              f"{len(adt_names)} proteins (e.g. {adt_names[:6]})")
    else:
        print("[adt] no adt.mtx — proceeding RNA-only (state labeling will use RNA markers)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    print(f"[write] {args.out}  ({adata.n_obs} cells × {adata.n_vars} genes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
