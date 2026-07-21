#!/usr/bin/env python
"""Assemble a single-timepoint Sheu 2024 AnnData for `cascadir.CascadeDirection`.

`cascadir.CascadeDirection.fit()` needs one `adata` (cells x genes) with
`condition_col` / `donor_col` / `celltype_col` obs columns (see cascadir/MANUAL.md
§1). The Sheu pseudo-tube manifest instead stores one preprocessed `.h5ad` per
(pseudo-donor, stimulus) pair, with per-tube `time_point` and `cell_type` obs
columns. This script pools cells across all manifest entries into one AnnData,
applying the SAME single-frame time filter as the legacy pipeline
(`cytokine_mil.analysis.eda_pair_benchmark.load_phase1_cells`): for stimulated
tubes, keep only cells whose `time_point == time_filter`; PBS tubes (0hr Unstim,
pooled per the Sheu adapter convention) are kept in full regardless of
`time_filter`, matching `load_phase1_cells`'s docstring exactly so the resulting
population matches what the existing 5hr `binary_ig.parquet` / `per_axis_summary.csv`
were computed over (this script does not reuse that legacy pipeline's code, only
its cell-selection semantics, so a fresh cascadir fit is comparable to it).

Usage (cluster):
    python scripts/prepare_sheu_cascadir.py \\
        --manifest_path /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_5hr_pseudotubes/manifest.json \\
        --time_filter 5hr \\
        --out /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_5hr_pseudotubes/prepared/sheu_5hr_prepared.h5ad

Note: the general `Sheu2024_pseudotubes/manifest.json` (CLAUDE.md §2.5) only covers
{0hr, 3hr} -- it has no 5hr cells at all. The 5hr-specific pseudotubes/manifest live
under a separate `Sheu2024_5hr_pseudotubes/` tree (built by
`slurm/build_pseudotubes_sheu_5hr.slurm` with `--time_points 0hr 5hr`), which is what
the existing 5hr `binary_ig.parquet` / `per_axis_summary.csv` were actually computed
from. Pointing this script at the wrong manifest silently drops every stimulated tube
(time filter matches zero cells) while PBS survives in full -- verify the assembled
AnnData's obs['cytokine'] contains all expected stimuli, not just PBS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse

MANIFEST_DEFAULT = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_5hr_pseudotubes/manifest.json"


def assemble(manifest_path: str, time_filter: str | None) -> ad.AnnData:
    with open(manifest_path) as f:
        entries = json.load(f)

    X_parts = []
    obs_cytokine, obs_donor, obs_celltype = [], [], []
    resolved_genes = None
    n_dropped_tubes = 0

    for entry in entries:
        cytokine = entry["cytokine"]
        donor = entry["donor"]
        tube = sc.read_h5ad(entry["path"])

        if time_filter is not None and cytokine != "PBS" and "time_point" in tube.obs.columns:
            mask = tube.obs["time_point"].astype(str).values == time_filter
            if not mask.any():
                n_dropped_tubes += 1
                continue
            tube = tube[mask].copy()

        if resolved_genes is None:
            resolved_genes = list(tube.var_names)
        else:
            tube = tube[:, resolved_genes]

        X = tube.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        cell_types = (
            tube.obs["cell_type"].astype(str).values
            if "cell_type" in tube.obs.columns
            else np.full(len(X), "unknown")
        )

        X_parts.append(X)
        obs_cytokine.extend([cytokine] * len(X))
        obs_donor.extend([donor] * len(X))
        obs_celltype.extend(list(cell_types))

    if not X_parts:
        raise RuntimeError(f"No cells assembled from {manifest_path} (time_filter={time_filter!r})")

    X_full = np.concatenate(X_parts, axis=0)
    obs = {
        "cytokine": np.array(obs_cytokine),
        "donor": np.array(obs_donor),
        "cell_type": np.array(obs_celltype),
    }
    adata = ad.AnnData(X=X_full, obs=obs, var={"gene": resolved_genes})
    adata.var_names = resolved_genes
    adata.uns["time_filter"] = time_filter or "none"
    adata.uns["n_dropped_tubes_no_matching_timepoint"] = n_dropped_tubes
    return adata


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest_path", default=MANIFEST_DEFAULT)
    ap.add_argument("--time_filter", default="5hr")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    adata = assemble(args.manifest_path, args.time_filter)
    print(f"[assemble] {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"[assemble] cytokines: {sorted(set(adata.obs['cytokine']))}")
    print(f"[assemble] donors:    {sorted(set(adata.obs['donor']))}")
    print(f"[assemble] cell_types: {sorted(set(adata.obs['cell_type']))}")
    print(f"[assemble] dropped tubes (no cells at time_filter): "
          f"{adata.uns['n_dropped_tubes_no_matching_timepoint']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"[write] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
