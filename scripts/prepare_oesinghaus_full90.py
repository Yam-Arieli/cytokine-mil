#!/usr/bin/env python
"""Stage 0 of the Oesinghaus full-90 DAG — materialise tubes + Stage-1 cells.

Writes, once, the artifacts every later stage reads:

  tubes/<donor>__<condition>.npy + tubes/meta.json
      The committed Oesinghaus pseudo-tubes, loaded into `cascadir`'s PseudoTube
      contract and sharded by (donor, condition). Building the tubes ONCE (rather than
      per array task) is what makes the 90 signatures comparable: `cross_asym` compares
      signatures across cytokines, and `cascadir.build_pseudotubes` advances a single RNG
      over the sorted (condition, donor) pairs, so re-building from a condition subset
      would yield different tubes for the same pair (the CLAUDE.md §27.6 failure mode).

  stage1_cells.h5ad
      UNIQUE cells for Stage-1 encoder pre-training — one tube per cytokine via
      `experiment_setup.build_stage1_manifest`, deduplicated by barcode.
      `cascadir.train_encoder` explicitly forbids being fed concatenated pseudo-tube
      contents (train.py:96-99) because that re-introduces cross-tube duplication.

  prepare_meta.json
      Actual counts (tubes, cells/tube, cell types, donors), so the later stages' memory
      asks can be checked against reality rather than an estimate.

Donors D2/D3 are excluded here and therefore from every downstream stage (CLAUDE.md §16).

This script performs no statistics: it reads h5ad files and serialises cascadir dataclasses.

Usage (cluster, CPU):
  python scripts/prepare_oesinghaus_full90.py --output_dir results/oes_full90
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _full90_config as C  # noqa: E402
from cytokine_mil.analysis.full90_tube_io import ShardWriter  # noqa: E402
from cytokine_mil.experiment_setup import build_stage1_manifest  # noqa: E402

from cascadir.types import PseudoTube  # noqa: E402


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _read_tube(path: str, gene_names: list) -> tuple:
    """Return (X float32 aligned to gene_names, cell_type labels, barcodes)."""
    import scanpy as sc
    import scipy.sparse

    tube = sc.read_h5ad(path)
    if list(tube.var_names) != gene_names:
        missing = [g for g in gene_names if g not in set(tube.var_names)]
        if missing:
            raise ValueError(
                f"{path}: {len(missing)} HVGs absent from the tube (e.g. {missing[:5]}). "
                "The tube was not built on this HVG list."
            )
        tube = tube[:, gene_names]
    X = tube.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if "cell_type" not in tube.obs.columns:
        raise ValueError(f"{path}: obs has no 'cell_type' column ({list(tube.obs.columns)})")
    return X, tube.obs["cell_type"].astype(str).to_numpy(), np.asarray(tube.obs_names, dtype=object)


def build_shards(manifest, gene_names, out_dir: Path) -> dict:
    """Stream the manifest into (donor, condition) shards, one group at a time."""
    groups = defaultdict(list)
    for entry in manifest:
        groups[(entry["donor"], entry["cytokine"])].append(entry)

    writer = ShardWriter(out_dir / "tubes")
    t0 = time.time()
    for i, (key, entries) in enumerate(sorted(groups.items()), 1):
        donor, condition = key
        tubes = []
        for entry in sorted(entries, key=lambda e: e.get("tube_idx", 0)):
            X, cell_types, _ = _read_tube(entry["path"], gene_names)
            included = entry.get("cell_types_included") or sorted(set(cell_types.tolist()))
            tubes.append(
                PseudoTube(
                    X=X,
                    condition=str(condition),
                    donor=str(donor),
                    cell_types=tuple(cell_types.tolist()),
                    cell_types_included=tuple(map(str, included)),
                    tube_idx=int(entry.get("tube_idx", 0)),
                )
            )
        writer.add_group(donor, condition, tubes)
        if i % 50 == 0 or i == len(groups):
            _log(f"  [{i}/{len(groups)}] groups written  ({time.time()-t0:.0f}s)")
    return writer.finalize(gene_names, C.CONTROL)


def build_stage1_cells(manifest, gene_names, out_path: Path) -> dict:
    """One tube per cytokine, pooled and deduplicated by barcode -> AnnData of unique cells."""
    import anndata as ad
    import pandas as pd

    stage1 = build_stage1_manifest(manifest)
    _log(f"  stage-1 manifest: {len(stage1)} tubes")

    X_parts, obs_rows, barcodes = [], [], []
    for entry in stage1:
        X, cell_types, bcs = _read_tube(entry["path"], gene_names)
        X_parts.append(X)
        barcodes.extend(bcs.tolist())
        for ct in cell_types:
            obs_rows.append((entry["cytokine"], entry["donor"], ct))

    X = np.concatenate(X_parts, axis=0)
    obs = pd.DataFrame(
        obs_rows, columns=[C.CONDITION_COL, C.DONOR_COL, C.CELLTYPE_COL]
    )
    obs.index = pd.Index([str(b) for b in barcodes], name="cell")

    n_before = len(obs)
    keep = ~obs.index.duplicated(keep="first")
    X, obs = X[keep], obs[keep]
    n_dup = n_before - len(obs)
    if n_dup:
        _log(f"  dropped {n_dup} duplicate barcodes ({n_dup/n_before:.2%})")
    if obs.index.duplicated().any():
        raise AssertionError("stage-1 cells are not unique after dedup")

    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = gene_names
    adata.write_h5ad(out_path)
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_duplicate_barcodes_dropped": int(n_dup),
        "cell_types": sorted(obs[C.CELLTYPE_COL].unique().tolist()),
        "donors": sorted(obs[C.DONOR_COL].unique().tolist()),
        "n_conditions": int(obs[C.CONDITION_COL].nunique()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest_path", default=C.MANIFEST_PATH)
    ap.add_argument("--hvg_path", default=C.HVG_PATH)
    ap.add_argument("--output_dir", default="results/oes_full90")
    ap.add_argument("--exclude_donors", nargs="*", default=C.VAL_DONORS)
    ap.add_argument("--limit_cytokines", type=int, default=None,
                    help="Debug only: cap the number of stimulus conditions.")
    ap.add_argument("--skip_tubes", action="store_true", help="Only rebuild stage1_cells.h5ad.")
    ap.add_argument("--sizing_only", action="store_true",
                    help="Preflight: report tube counts and the memory each later stage "
                         "needs, from the manifest alone. Writes nothing.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.hvg_path) as fh:
        gene_names = [str(g) for g in json.load(fh)]
    with open(args.manifest_path) as fh:
        manifest = json.load(fh)
    _log(f"[load] manifest={len(manifest)} entries  hvgs={len(gene_names)}")

    excl = set(args.exclude_donors or [])
    manifest = [e for e in manifest if e["donor"] not in excl]
    _log(f"[filter] excluded donors {sorted(excl)} -> {len(manifest)} entries")

    stimuli = sorted({e["cytokine"] for e in manifest} - {C.CONTROL})
    if args.limit_cytokines:
        stimuli = stimuli[: args.limit_cytokines]
        keep = set(stimuli) | {C.CONTROL}
        manifest = [e for e in manifest if e["cytokine"] in keep]
        _log(f"[debug] limited to {len(stimuli)} stimuli -> {len(manifest)} entries")
    if C.CONTROL not in {e["cytokine"] for e in manifest}:
        raise SystemExit(f"FATAL: control {C.CONTROL!r} absent from the filtered manifest")
    _log(f"[plan] {len(stimuli)} stimuli + {C.CONTROL}; "
         f"donors {sorted({e['donor'] for e in manifest})}")

    if args.sizing_only:
        cells = [int(e["n_cells"]) for e in manifest]
        total = int(np.sum(cells))
        copy_gb = total * len(gene_names) * 4 / 1e9
        groups = len({(e["donor"], e["cytokine"]) for e in manifest})
        _log("")
        _log("=== SIZING PREFLIGHT (nothing written) ===")
        _log(f"  tubes            : {len(manifest)}  in {groups} (donor, condition) shards")
        _log(f"  cells            : {total}  (mean {np.mean(cells):.0f}/tube, "
             f"min {np.min(cells)}, max {np.max(cells)})")
        _log(f"  genes            : {len(gene_names)}")
        _log(f"  one full copy    : {copy_gb:.1f} GB  (float32, dense)")
        _log(f"  shards on disk   : {copy_gb:.1f} GB")
        _log("")
        _log(f"  stage 2 per task : ~{copy_gb * (10 + 1) / max(len(stimuli), 1):.1f} GB "
             "(its 10 conditions + PBS)")
        _log(f"  stage 5 direction: ~{2*copy_gb:.0f} GB  (tube_set + cells_by_pair)")
        _log(f"  stage 4 coupling : ~{3*copy_gb:.0f} GB  (+ the per-donor cells_by_pair dicts)")
        _log("")
        _log("Compare against slurm/oes90/{train,direction,coupling}.slurm --mem.")
        return 0

    meta = {"stimuli": stimuli, "n_manifest_entries": len(manifest),
            "excluded_donors": sorted(excl), "hvg_path": args.hvg_path,
            "manifest_path": args.manifest_path}

    if not args.skip_tubes:
        _log("\n[tubes] writing shards ...")
        tube_meta = build_shards(manifest, gene_names, out)
        cells = [t["n_cells"] for s in tube_meta["shards"] for t in s["tubes"]]
        meta["tubes"] = {
            "n_tubes": tube_meta["n_tubes"],
            "n_shards": len(tube_meta["shards"]),
            "shards_sha256": tube_meta["shards_sha256"],
            "cells_per_tube_mean": float(np.mean(cells)),
            "cells_per_tube_min": int(np.min(cells)),
            "cells_per_tube_max": int(np.max(cells)),
            "total_cells": int(np.sum(cells)),
            "cell_types": tube_meta["cell_types"],
            "est_full_set_bytes": int(np.sum(cells)) * len(gene_names) * 4,
        }
        gb = meta["tubes"]["est_full_set_bytes"] / 1e9
        _log(f"[tubes] {tube_meta['n_tubes']} tubes, {np.sum(cells)} cells, "
             f"{len(tube_meta['cell_types'])} cell types -> one full copy = {gb:.1f} GB")
        _log(f"[tubes] a stage holding tube_set + cells_by_pair needs ~{2*gb:.0f} GB; "
             f"coupling (adds per-donor dicts) ~{3*gb:.0f} GB")

    _log("\n[stage1] building unique-cell AnnData for the encoder ...")
    meta["stage1_cells"] = build_stage1_cells(manifest, gene_names, out / "stage1_cells.h5ad")
    _log(f"[stage1] {meta['stage1_cells']['n_cells']} unique cells x "
         f"{meta['stage1_cells']['n_genes']} genes")

    C.write_json(out / "prepare_meta.json", meta)
    _log(f"\n[done] artifacts in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
