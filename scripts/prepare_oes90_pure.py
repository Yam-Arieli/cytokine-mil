"""Stage 0 of the Oesinghaus 90-cytokine PURE run — tube split + Stage-1 cell set.

Does NOT re-materialise the pseudo-tubes: the §36 prepare stage already wrote them as
sha-verified `(donor, condition)` shards, built once from the committed pseudo-tubes with
D2/D3 excluded. Rebuilding them per run would be the CLAUDE.md §27.6 failure in a new
costume (`cascadir.build_pseudotubes` advances a single RNG over the sorted
(condition, donor) pairs, so any subset yields different tubes for the same pair). This
stage verifies that shard set and records which tubes each downstream stage may use.

Writes
------
  tube_split.json          MAIN / RESERVE tube_idx lists + the shard digest they refer to
  stage1_cells.h5ad        unique cells for the encoder, equal weight per cytokine
  stage1_composition.csv   per (cytokine, donor, cell_type) cell counts actually used
  prepare_meta.json        counts, digests, and the memory arithmetic for later stages

The Stage-1 set is capped at CELLS_PER_CYTOKINE unique cells per cytokine, drawn evenly
across donors. Without the cap, cytokines whose tubes happen to draw from larger pools
would carry more weight in the encoder than others — exactly the non-neutrality this run
exists to remove.

Pure I/O and bookkeeping; no method math (CLAUDE.md §37).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_pure_config as C  # noqa: E402

from cytokine_mil.analysis.full90_tube_io import read_meta  # noqa: E402


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
    return (
        X,
        tube.obs["cell_type"].astype(str).to_numpy(),
        np.asarray(tube.obs_names, dtype=object),
    )


def verify_shards(shard_dir: str, exclude_donors=None) -> dict:
    """Read the reused shard meta and report what it contains."""
    exclude_donors = list(C.VAL_DONORS if exclude_donors is None else exclude_donors)
    meta = read_meta(shard_dir)
    idx = sorted({int(t["tube_idx"]) for s in meta["shards"] for t in s["tubes"]})
    donors = sorted({s["donor"] for s in meta["shards"]})
    conds = sorted({s["condition"] for s in meta["shards"]})
    C.log(f"[shards] {meta['n_tubes']} tubes  {len(conds)} conditions  {len(donors)} donors")
    C.log(f"[shards] tube_idx values present: {idx}")
    C.log(f"[shards] shards_sha256={meta['shards_sha256'][:16]}...")

    bad = sorted(set(donors) & set(exclude_donors))
    if bad:
        raise AssertionError(
            f"held-out donors {bad} are present in the shard set — CLAUDE.md §16 requires "
            "them excluded everywhere, Stage-1 included."
        )
    need = set(C.MAIN_TUBE_INDICES) | set(C.RESERVE_TUBE_INDICES)
    if not need.issubset(set(idx)):
        raise AssertionError(
            f"shards only carry tube_idx {idx}; the split needs {sorted(need)}."
        )
    return {
        "shards_sha256": meta["shards_sha256"],
        "n_tubes": int(meta["n_tubes"]),
        "conditions": conds,
        "donors": donors,
        "tube_indices_present": idx,
        "gene_names_n": len(meta["gene_names"]),
        "control_label": meta["control_label"],
    }


def build_stage1_cells(manifest, gene_names, out_path: Path, comp_path: Path) -> dict:
    """Unique cells for the encoder: equal weight per cytokine, balanced across donors.

    Streams one (cytokine, donor) group at a time so peak memory stays at a few tubes,
    not the whole 25 GB main tube set.
    """
    import anndata as ad

    groups = defaultdict(list)
    for entry in manifest:
        if int(entry.get("tube_idx", 0)) in set(C.MAIN_TUBE_INDICES):
            groups[(entry["cytokine"], entry["donor"])].append(entry)

    cytokines = sorted({k[0] for k in groups})
    donors = sorted({k[1] for k in groups})
    per_donor = max(1, C.CELLS_PER_CYTOKINE // max(len(donors), 1))
    C.log(
        f"  {len(cytokines)} conditions x {len(donors)} donors; target {per_donor} unique "
        f"cells per (condition, donor) -> <= {C.CELLS_PER_CYTOKINE} per condition"
    )

    rng = np.random.default_rng(C.STAGE1_CELL_SEED)
    X_parts, obs_rows, comp_rows = [], [], []
    t0 = time.time()
    for i, (cyt, donor) in enumerate(sorted(groups), 1):
        Xs, cts, bcs = [], [], []
        for entry in sorted(groups[(cyt, donor)], key=lambda e: e.get("tube_idx", 0)):
            X, ct, bc = _read_tube(entry["path"], gene_names)
            Xs.append(X)
            cts.append(ct)
            bcs.append(bc)
        X = np.concatenate(Xs, axis=0)
        ct = np.concatenate(cts)
        bc = np.concatenate(bcs)

        # a cell can be drawn into several tubes of the same (donor, cytokine)
        _, keep = np.unique(bc, return_index=True)
        keep = np.sort(keep)
        if len(keep) > per_donor:
            keep = np.sort(rng.choice(keep, size=per_donor, replace=False))

        X_parts.append(X[keep])
        for j in keep:
            obs_rows.append({
                C.CONDITION_COL: str(cyt),
                C.DONOR_COL: str(donor),
                C.CELLTYPE_COL: str(ct[j]),
            })
        for cell_type, n in zip(*np.unique(ct[keep], return_counts=True)):
            comp_rows.append({
                "cytokine": str(cyt), "donor": str(donor),
                "cell_type": str(cell_type), "n_cells": int(n),
            })
        if i % 100 == 0 or i == len(groups):
            C.log(f"  [{i}/{len(groups)}] groups  ({time.time()-t0:.0f}s)")

    X_all = np.concatenate(X_parts, axis=0)
    obs = pd.DataFrame(obs_rows)
    obs.index = pd.Index([str(i) for i in range(len(obs))], name="cell")
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
    adata = ad.AnnData(X=X_all, obs=obs, var=var)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(comp_path, index=False)

    per_cyt = obs.groupby(C.CONDITION_COL).size()
    return {
        "path": str(out_path),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "cells_per_cytokine_target": C.CELLS_PER_CYTOKINE,
        "cells_per_cytokine_min": int(per_cyt.min()),
        "cells_per_cytokine_median": float(per_cyt.median()),
        "cells_per_cytokine_max": int(per_cyt.max()),
        "n_conditions": int(obs[C.CONDITION_COL].nunique()),
        "donors": sorted(obs[C.DONOR_COL].unique().tolist()),
        "cell_types": sorted(obs[C.CELLTYPE_COL].unique().tolist()),
        "composition_csv": str(comp_path),
    }


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard_dir", default=C.SHARD_DIR)
    ap.add_argument("--manifest_path", default=C.MANIFEST_PATH)
    ap.add_argument("--hvg_path", default=C.HVG_PATH)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--exclude_donors", nargs="*", default=None,
                    help="Override the held-out donors (default: CLAUDE.md §16's D2/D3).")
    args = ap.parse_args()
    exclude = set(C.VAL_DONORS if args.exclude_donors is None else args.exclude_donors)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    C.log("[verify] reusing the §36 tube shards (read-only)")
    shard_info = verify_shards(args.shard_dir, exclude_donors=exclude)

    with open(args.hvg_path) as fh:
        gene_names = json.load(fh)
    with open(args.manifest_path) as fh:
        manifest = json.load(fh)
    C.log(f"[load] manifest={len(manifest)} entries  hvgs={len(gene_names)}")
    if len(gene_names) != shard_info["gene_names_n"]:
        raise AssertionError(
            f"HVG list has {len(gene_names)} genes but the shards were built on "
            f"{shard_info['gene_names_n']} — these are not the same tubes."
        )

    manifest = [e for e in manifest if e["donor"] not in exclude]
    C.log(f"[filter] excluded donors {sorted(exclude)} -> {len(manifest)} entries")

    C.write_json(out / "tube_split.json", {
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "reserve_tube_indices": C.RESERVE_TUBE_INDICES,
        "shards_sha256": shard_info["shards_sha256"],
        "shard_dir": args.shard_dir,
        "note": (
            "The same tube_idx values in every (donor, condition) group — the split "
            "encodes no per-cytokine choice. MAIN drives the whole method; RESERVE is "
            "used only to re-derive signatures for the memorisation check."
        ),
    })
    n_main = len(C.MAIN_TUBE_INDICES) * len(shard_info["donors"]) * len(shard_info["conditions"])
    C.log(f"[split] main = tube_idx {C.MAIN_TUBE_INDICES} -> ~{n_main} tubes")
    C.log(f"[split] reserve = tube_idx {C.RESERVE_TUBE_INDICES} (signature check only)")

    C.log("\n[stage1] building the equal-weight unique-cell AnnData for the encoder ...")
    stage1 = build_stage1_cells(
        manifest, gene_names, out / "stage1_cells.h5ad", out / "stage1_composition.csv"
    )
    C.log(
        f"[stage1] {stage1['n_cells']} cells x {stage1['n_genes']} genes; per-cytokine "
        f"min/median/max = {stage1['cells_per_cytokine_min']}/"
        f"{stage1['cells_per_cytokine_median']:.0f}/{stage1['cells_per_cytokine_max']}"
    )
    C.log(f"[stage1] {len(stage1['cell_types'])} cell types: {stage1['cell_types']}")

    meta = {
        "run": "oes90_pure",
        "shards": shard_info,
        "excluded_donors": sorted(exclude),
        "tube_split": {
            "main": C.MAIN_TUBE_INDICES,
            "reserve": C.RESERVE_TUBE_INDICES,
            "n_main_tubes_expected": n_main,
        },
        "stage1_cells": stage1,
        "hyperparameters": {
            "embed_dim": C.EMBED_DIM, "hidden_dims": list(C.HIDDEN_DIMS),
            "attention_hidden_dim": C.ATTENTION_HIDDEN_DIM,
            "stage1_epochs_cap": C.STAGE1_EPOCHS, "stage1_lr": C.STAGE1_LR,
            "stage1_val_fraction": C.STAGE1_VAL_FRACTION,
            "stage1_patience": C.STAGE1_PATIENCE,
            "stage1_extra_epochs_after_stop": C.STAGE1_EXTRA_EPOCHS,
            "stage2_epochs": C.STAGE2_EPOCHS, "stage2_lr": C.STAGE2_LR,
            "top_n": C.TOP_N, "n_ig_steps": C.N_IG_STEPS, "seed": C.SEED,
        },
    }
    C.write_json(out / "prepare_meta.json", meta)
    C.mark_done(out, "prepare")
    C.log(f"\n[done] artifacts in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
