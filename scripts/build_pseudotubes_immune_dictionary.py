"""
Build pseudo-tubes for the Immune Dictionary (Cui et al., Nature 2024; GSE202186 /
SCP2554).

DATA-SOURCE NOTE (revised 2026-06-01 after the GEO-demux dead end):
  The GEO RAW.tar provides raw 10x gene-expression matrices per *hashed lane*
  (cytokine-samplesNN), but NO per-cell cytokine/cell-type labels — the MULTI-seq
  sample tags were processed separately and there are zero HTO features in the
  deposited matrices. The per-cell labels live in the public SCP2554 study, which
  we pull via the SCP REST API (no auth; see scripts/fetch_scp_id_metadata.py)
  into data/immune_dictionary_scp_metadata.parquet.

  This adapter JOINS the two:
    - EXPRESSION  <- GEO MTX lanes (extracted/*-cytokine-samplesNN-*.{mtx,tsv}.gz)
    - LABELS      <- SCP metadata (barcode16 + channel -> cyt, celltype, rep)
  Join key: SCP "<barcode16>-<NN>"  <->  GEO lane samplesNN, barcode "<barcode16>-1".

Pseudo-tube scheme (mirrors Sheu/Oesinghaus):
  - donor   = rep (rep01..rep14); each benchmark cytokine has rep01/02/03.
  - cytokine= SCP `cyt` (machine names: IL1b, IFNg, TNFa, ..., PBS). PBS already PBS.
  - cell types = SCP `celltype` (expert annotations; "doublet" dropped).
  - 4000 HVG (seurat_v3, batch_key=rep); normalize_total -> log1p.
  - N_PER_CELL_TYPE=30, N_PSEUDO_TUBES=10, MIN_CELLS_THRESHOLD=10.
  - val mouse = rep03 (each benchmark cytokine spans rep01/02/03 -> 2 train + 1 val).

Run on the cluster:
    python scripts/build_pseudotubes_immune_dictionary.py
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

# ---------------------------------------------------------------------------
# Defaults (cluster paths; override via CLI)
# ---------------------------------------------------------------------------
RAW_EXTRACTED = "/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary/raw/extracted"
SCP_METADATA  = str(REPO_ROOT / "reports" / "immune_dictionary" / "scp_metadata.parquet")
BASE_PATH     = "/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes"

N_PER_CELL_TYPE     = 30
MIN_CELLS_THRESHOLD = 10
N_PSEUDO_TUBES      = 10
N_HVGS              = 4000
RANDOM_SEED         = 42
VAL_REP             = "rep03"          # held-out donor (each benchmark cyt has rep01/02/03)
DROP_CELLTYPES      = {"doublet"}       # artifacts; keep all real (incl. stromal) types
MIN_MATCH_FRAC      = 0.30              # per-channel barcode-match floor before warning


# ===========================================================================
# GEO 10x lane loading
# ===========================================================================

def _find_lane_files(extracted: Path, channel: str):
    """Return (matrix, barcodes, features) paths for GEO lane cytokine-samples{channel}.

    GEO names files like 'GSM6102836_cytokine-samples01-barcodes.tsv.gz' — note the
    separator before 'cytokine' is '_', so the glob must NOT require a leading '-'.
    """
    infix = f"cytokine-samples{channel}-"
    mtx = list(extracted.glob(f"*{infix}matrix.mtx.gz"))
    bc  = list(extracted.glob(f"*{infix}barcodes.tsv.gz"))
    ft  = list(extracted.glob(f"*{infix}features.tsv.gz"))
    if not (len(mtx) == len(bc) == len(ft) == 1):
        return None
    return mtx[0], bc[0], ft[0]


def _strip_bc_suffix(bc: str) -> str:
    """'AAAC...-1' -> 'AAAC...'  (drop any trailing -N CellRanger suffix)."""
    return bc.rsplit("-", 1)[0] if "-" in bc else bc


def _load_lane_matched(extracted: Path, channel: str, meta_channel: pd.DataFrame, log):
    """
    Load one GEO lane, subset to the barcodes present in `meta_channel`
    (the SCP cells for this channel), attach cyt/celltype/rep.  Returns an
    AnnData (cells x genes, raw counts) or None if the lane is missing / no match.
    """
    found = _find_lane_files(extracted, channel)
    if found is None:
        log(f"  channel {channel}: GEO lane files not found / ambiguous — SKIP")
        return None
    mtx_path, bc_path, ft_path = found

    with gzip.open(ft_path, "rt") as fh:
        feats = [ln.rstrip("\n").split("\t") for ln in fh]
    gene_symbols = [f[1] if len(f) > 1 else f[0] for f in feats]
    with gzip.open(bc_path, "rt") as fh:
        geo_barcodes = [ln.strip() for ln in fh]

    # SCP barcodes we want from this channel
    want = {bc16: i for i, bc16 in enumerate(meta_channel["barcode16"].tolist())}
    geo_core = [_strip_bc_suffix(b) for b in geo_barcodes]
    geo_pos = {bc: j for j, bc in enumerate(geo_core)}

    matched_bc16 = [bc for bc in want if bc in geo_pos]
    match_frac = len(matched_bc16) / max(1, len(want))
    log(f"  channel {channel}: GEO barcodes={len(geo_barcodes)}, "
        f"SCP cells={len(want)}, matched={len(matched_bc16)} "
        f"({100*match_frac:.1f}%)")
    if not matched_bc16:
        log(f"  channel {channel}: ZERO barcode matches — SKIP (check lane mapping)")
        return None
    if match_frac < MIN_MATCH_FRAC:
        log(f"  channel {channel}: WARNING match_frac {match_frac:.2f} < "
            f"{MIN_MATCH_FRAC} — keeping matched cells but flag")

    # Load matrix (genes x barcodes), subset to matched columns
    mat = scipy.io.mmread(str(mtx_path)).tocsc()   # genes x barcodes
    geo_cols = [geo_pos[bc] for bc in matched_bc16]
    sub = mat[:, geo_cols].T.tocsr()               # cells x genes
    sub = sub.astype(np.float32)

    meta_idx = meta_channel.set_index("barcode16")
    obs = meta_idx.loc[matched_bc16, ["cyt", "celltype", "rep"]].copy()
    obs.index = [f"{bc}-{channel}" for bc in matched_bc16]

    adata = ad.AnnData(X=sub, obs=obs)
    adata.var_names = gene_symbols
    adata.var_names_make_unique()
    return adata


# ===========================================================================
# Pseudo-tube construction
# ===========================================================================

def _sample_one_tube(adata_sub, rng, n_per_cell_type, min_cells,
                     celltype_col="cell_type"):
    """Stratified sample n_per_cell_type cells per cell type within one
    (cytokine, donor) AnnData. Returns (indices, cell_types_included)."""
    idx = []
    types_included = []
    cts = adata_sub.obs[celltype_col].values
    for ct in sorted(set(cts)):
        ct_pos = np.where(cts == ct)[0]
        if len(ct_pos) < min_cells:
            continue
        take = min(n_per_cell_type, len(ct_pos))
        chosen = rng.choice(ct_pos, size=take, replace=False)
        idx.extend(chosen.tolist())
        types_included.append(ct)
    return np.array(sorted(idx), dtype=int), types_included


def build_pseudo_tubes_id(
    adata,
    base_path,
    n_per_cell_type=N_PER_CELL_TYPE,
    min_cells_threshold=MIN_CELLS_THRESHOLD,
    n_pseudo_tubes=N_PSEUDO_TUBES,
    rng=None,
    donor_col="mouse_id",
    cytokine_col="cytokine",
    celltype_col="cell_type",
):
    """
    Build pseudo-tubes from a (already-preprocessed) AnnData and write the
    manifest. Reusable across the real build (main()) and the demo fixture
    (tests/test_demo_id.py). The input adata's obs must carry the donor,
    cytokine and cell_type columns named by the *_col args. If a raw-count
    layer "counts" is present, a pseudotube_N_raw.h5ad is written alongside.

    Returns the manifest (list of dicts: path, donor, cytokine, n_cells,
    cell_types_included, tube_idx).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)
    has_raw = "counts" in getattr(adata, "layers", {})

    manifest = []
    pairs = adata.obs[[cytokine_col, donor_col]].drop_duplicates().values.tolist()
    for cyt, donor in sorted(pairs, key=lambda x: (str(x[0]), str(x[1]))):
        mask = ((adata.obs[cytokine_col].values == cyt)
                & (adata.obs[donor_col].values == donor))
        sub = adata[mask]
        if sub.n_obs < min_cells_threshold:
            continue
        cyt_dir = base / str(donor) / str(cyt).replace("/", "_")
        cyt_dir.mkdir(parents=True, exist_ok=True)
        for t in range(n_pseudo_tubes):
            idx, types_inc = _sample_one_tube(
                sub, rng, n_per_cell_type, min_cells_threshold, celltype_col,
            )
            if len(idx) < min_cells_threshold:
                continue
            tube = sub[idx].copy()
            out_path = cyt_dir / f"pseudotube_{t}.h5ad"
            tube.write_h5ad(str(out_path), compression="gzip")
            if has_raw:
                tube_raw = tube.copy()
                tube_raw.X = tube.layers["counts"].copy()
                tube_raw.write_h5ad(
                    str(cyt_dir / f"pseudotube_{t}_raw.h5ad"), compression="gzip",
                )
            manifest.append({
                "path": str(out_path),
                "donor": str(donor),
                "cytokine": str(cyt),
                "n_cells": int(tube.n_obs),
                "cell_types_included": types_inc,
                "tube_idx": t,
            })
    with open(base / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Build ID pseudo-tubes (GEO expr + SCP labels).")
    ap.add_argument("--raw_extracted", default=RAW_EXTRACTED)
    ap.add_argument("--scp_metadata", default=SCP_METADATA)
    ap.add_argument("--base_path", default=BASE_PATH)
    ap.add_argument("--n_per_cell_type", type=int, default=N_PER_CELL_TYPE)
    ap.add_argument("--n_pseudo_tubes", type=int, default=N_PSEUDO_TUBES)
    ap.add_argument("--n_hvgs", type=int, default=N_HVGS)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--val_rep", default=VAL_REP)
    ap.add_argument("--limit_channels", default=None,
                    help="Comma-separated channels (e.g. 01,02,03) for a smoke build.")
    ap.add_argument("--limit_cytokines", default=None,
                    help="Comma-separated cyt names to keep (smoke build).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    extracted = Path(args.raw_extracted)
    base = Path(args.base_path)
    base.mkdir(parents=True, exist_ok=True)

    def log(msg=""):
        print(msg, flush=True)

    log("=== Immune Dictionary pseudo-tube build (GEO expr + SCP labels) ===")
    log(f"raw_extracted: {extracted}")
    log(f"scp_metadata:  {args.scp_metadata}")
    log(f"base_path:     {base}")

    # ------------------------------------------------------------------
    # SCP per-cell metadata (the demux key)
    # ------------------------------------------------------------------
    if not Path(args.scp_metadata).exists():
        log(f"FATAL: SCP metadata not found at {args.scp_metadata}. "
            f"Run scripts/fetch_scp_id_metadata.py first.")
        sys.exit(2)
    meta = pd.read_parquet(args.scp_metadata)
    meta = meta[~meta["celltype"].isin(DROP_CELLTYPES)].copy()
    if args.limit_cytokines:
        keep = set(args.limit_cytokines.split(","))
        meta = meta[meta["cyt"].isin(keep)].copy()
    channels = sorted(meta["channel"].unique())
    if args.limit_channels:
        sel = set(args.limit_channels.split(","))
        channels = [c for c in channels if c in sel]
    log(f"SCP metadata: {len(meta)} cells, {meta['cyt'].nunique()} cytokines, "
        f"{meta['celltype'].nunique()} cell types, channels={len(channels)}")

    # ------------------------------------------------------------------
    # Load + join each channel
    # ------------------------------------------------------------------
    parts = []
    for ch in channels:
        meta_ch = meta[meta["channel"] == ch]
        a = _load_lane_matched(extracted, ch, meta_ch, log)
        if a is not None:
            parts.append(a)
    if not parts:
        log("FATAL: no channels produced matched cells. Check GEO lane mapping.")
        sys.exit(3)

    log(f"\nConcatenating {len(parts)} channels...")
    # inner join on genes — lanes share the same 31053-gene panel.
    adata = ad.concat(parts, join="inner", merge="same")
    del parts
    log(f"  combined: {adata.n_obs} cells x {adata.n_vars} genes")
    log(f"  cytokines: {adata.obs['cyt'].nunique()}  "
        f"(PBS present: {'PBS' in set(adata.obs['cyt'])})")
    log(f"  reps: {sorted(adata.obs['rep'].unique())}")

    # ------------------------------------------------------------------
    # Preprocess: HVG (seurat_v3 on raw counts) -> normalize -> log1p
    # ------------------------------------------------------------------
    adata.layers["counts"] = adata.X.copy()
    log("\nSelecting HVGs (seurat_v3, batch_key=rep)...")
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=args.n_hvgs, flavor="seurat_v3",
            batch_key="rep", layer="counts",
        )
    except Exception as e:
        log(f"  seurat_v3 with batch_key failed ({e}); retrying without batch_key")
        sc.pp.highly_variable_genes(
            adata, n_top_genes=args.n_hvgs, flavor="seurat_v3", layer="counts",
        )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    hvg_mask = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask].tolist()
    adata_hvg = adata[:, hvg_mask].copy()   # carries layers["counts"] subset to HVGs
    log(f"  HVGs selected: {len(hvg_genes)}")

    with open(base / "hvg_list.json", "w") as fh:
        json.dump(hvg_genes, fh)
    log("  Saved hvg_list.json")

    # ------------------------------------------------------------------
    # Build pseudo-tubes (reusable builder; standardise obs column names to
    # the manifest / PseudoTubeDataset contract: donor / cytokine / cell_type).
    # ------------------------------------------------------------------
    adata_hvg.obs = adata_hvg.obs.rename(
        columns={"cyt": "cytokine", "celltype": "cell_type", "rep": "mouse_id"}
    )
    log("\nBuilding pseudo-tubes per (cytokine, rep)...")
    manifest = build_pseudo_tubes_id(
        adata_hvg, base,
        n_per_cell_type=args.n_per_cell_type,
        min_cells_threshold=MIN_CELLS_THRESHOLD,
        n_pseudo_tubes=args.n_pseudo_tubes,
        rng=rng,
        donor_col="mouse_id", cytokine_col="cytokine", celltype_col="cell_type",
    )
    log(f"  built {len(manifest)} pseudo-tubes; saved manifest.json")

    # ------------------------------------------------------------------
    # Build metadata (val mouse, coverage)
    # ------------------------------------------------------------------
    cyts_in_manifest = sorted({m["cytokine"] for m in manifest})
    reps_in_manifest = sorted({m["donor"] for m in manifest})
    build_meta = {
        "dataset": "ImmuneDictionary_Cui2024",
        "source": "GEO GSE202186 expression + SCP2554 per-cell labels (public API)",
        "val_mouse": args.val_rep,
        "n_tubes": len(manifest),
        "n_cytokines": len(cyts_in_manifest),
        "reps": reps_in_manifest,
        "n_hvgs": len(hvg_genes),
        "cytokines": cyts_in_manifest,
    }
    with open(base / "build_metadata.json", "w") as fh:
        json.dump(build_meta, fh, indent=2)
    log(f"  Saved build_metadata.json  val_mouse={args.val_rep}")

    # Coverage summary
    log("\n=== RUN SUMMARY ===")
    log(f"  tubes:      {len(manifest)}")
    log(f"  cytokines:  {len(cyts_in_manifest)} (PBS in: {'PBS' in cyts_in_manifest})")
    log(f"  reps:       {reps_in_manifest}")
    log(f"  HVGs:       {len(hvg_genes)}")
    log(f"  val mouse:  {args.val_rep}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
