"""Stage 0 of the encoder-breadth sweep — one cell bank, four arm-specific Stage-1 sets.

Builds a single bank of UNIQUE cells (deduplicated across the tubes of each
(condition, donor), since a cell can be drawn into several tubes of the same well), then
carves each arm's `stage1_cells.h5ad` out of it.

The controlled variable is condition breadth, so the total Stage-1 cell count is held
FIXED across arms and split evenly over the groups each arm uses (PBS counts as a group and
is present in every arm — it is the negative class of every binary model). With Stage-1
epochs also fixed at 20, every arm gets identical gradient exposure and the only thing that
differs is how many distinct perturbations the encoder is asked to be invariant to.

If the `pbs_only` arm cannot supply the target from PBS wells alone, the budget for ALL
arms drops to what PBS can supply, and the actual figure is recorded. A smaller controlled
sweep is worth more than a larger uncontrolled one.

This script chooses no formula: it reads tubes, deduplicates, subsamples, and writes files.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _encsweep_config as C  # noqa: E402
from prepare_oes90_pure import _read_tube, verify_shards  # noqa: E402


def build_bank(manifest, gene_names) -> tuple:
    """One streaming pass -> (X, obs, group_index). Unique cells only."""
    groups = defaultdict(list)
    for entry in manifest:
        if int(entry.get("tube_idx", 0)) in set(C.MAIN_TUBE_INDICES):
            groups[(entry["cytokine"], entry["donor"])].append(entry)

    conds = sorted({k[0] for k in groups})
    donors = sorted({k[1] for k in groups})
    C.log(f"  bank over {len(conds)} conditions x {len(donors)} donors "
          f"(caps: PBS {C.PBS_CAP_PER_DONOR}/donor, "
          f"stimuli {C.STIM_CAP_PER_DONOR}/donor)")

    rng = np.random.default_rng(C.STAGE1_CELL_SEED)
    X_parts, obs_rows, avail_rows = [], [], []
    group_index: dict = {}
    cursor = 0
    t0 = time.time()
    for i, (cond, donor) in enumerate(sorted(groups), 1):
        Xs, cts, bcs = [], [], []
        for entry in sorted(groups[(cond, donor)], key=lambda e: e.get("tube_idx", 0)):
            X, ct, bc = _read_tube(entry["path"], gene_names)
            Xs.append(X); cts.append(ct); bcs.append(bc)
        X = np.concatenate(Xs, axis=0)
        ct = np.concatenate(cts)
        bc = np.concatenate(bcs)

        _, keep = np.unique(bc, return_index=True)
        keep = np.sort(keep)
        n_available = len(keep)
        cap = C.PBS_CAP_PER_DONOR if str(cond) == C.CONTROL else C.STIM_CAP_PER_DONOR
        if n_available > cap:
            keep = np.sort(rng.choice(keep, size=cap, replace=False))

        X_parts.append(X[keep])
        for j in keep:
            obs_rows.append({
                C.CONDITION_COL: str(cond),
                C.DONOR_COL: str(donor),
                C.CELLTYPE_COL: str(ct[j]),
            })
        group_index[(str(cond), str(donor))] = (cursor, cursor + len(keep))
        cursor += len(keep)
        avail_rows.append({
            "cytokine": str(cond), "donor": str(donor),
            "n_unique_available": int(n_available), "n_banked": int(len(keep)),
        })
        if i % 100 == 0 or i == len(groups):
            C.log(f"  [{i}/{len(groups)}] groups banked ({time.time()-t0:.0f}s)")

    X_all = np.concatenate(X_parts, axis=0)
    obs = pd.DataFrame(obs_rows)
    C.log(f"  bank: {X_all.shape[0]} unique cells x {X_all.shape[1]} genes "
          f"({X_all.nbytes/1e9:.1f} GB)")
    return X_all, obs, group_index, pd.DataFrame(avail_rows), donors


def write_arm(arm, subset, X, obs, group_index, donors, budget, gene_names, out_dir,
              shard_info, shard_dir) -> dict:
    """Carve one arm's Stage-1 set out of the bank and write it."""
    import anndata as ad

    adir = C.arm_dir(out_dir, arm)
    adir.mkdir(parents=True, exist_ok=True)

    conds = [C.CONTROL] + list(subset)          # PBS is in every arm
    per_group = budget // len(conds)
    per_donor = max(1, per_group // len(donors))

    rows, comp_rows = [], []
    for cond in conds:
        for donor in donors:
            span = group_index.get((cond, donor))
            if span is None:
                continue
            lo, hi = span
            take = min(per_donor, hi - lo)
            rows.extend(range(lo, lo + take))     # bank order is already a random draw
    rows = np.asarray(sorted(rows), dtype=np.int64)

    sub_obs = obs.iloc[rows].reset_index(drop=True)
    sub_obs.index = pd.Index([str(i) for i in range(len(sub_obs))], name="cell")
    adata = ad.AnnData(
        X=np.ascontiguousarray(X[rows]),
        obs=sub_obs,
        var=pd.DataFrame(index=pd.Index(gene_names, name="gene")),
    )
    adata.write_h5ad(adir / "stage1_cells.h5ad")

    for (cond, donor, ctype), n in sub_obs.groupby(
        [C.CONDITION_COL, C.DONOR_COL, C.CELLTYPE_COL]
    ).size().items():
        comp_rows.append({"cytokine": cond, "donor": donor,
                          "cell_type": ctype, "n_cells": int(n)})
    pd.DataFrame(comp_rows).to_csv(adir / "stage1_composition.csv", index=False)

    # Each arm needs its own split file: _oes90_pure_estimator.load_tubes reads it from the
    # directory it is handed, and the sweep's tube budget (k=10) differs from §37's.
    C.write_json(adir / "tube_split.json", {
        "shard_dir": str(shard_dir),
        "shards_sha256": shard_info["shards_sha256"],
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "reserve_tube_indices": C.RESERVE_TUBE_INDICES,
    })

    per_cond = sub_obs.groupby(C.CONDITION_COL).size()
    info = {
        "arm": arm,
        "n_encoder_conditions": len(conds),
        "n_stimulus_conditions": len(subset),
        "n_cells": int(adata.n_obs),
        "budget": int(budget),
        "cells_per_condition_target": int(per_group),
        "cells_per_donor_target": int(per_donor),
        "cells_per_condition_min": int(per_cond.min()),
        "cells_per_condition_max": int(per_cond.max()),
        "n_cell_types": int(sub_obs[C.CELLTYPE_COL].nunique()),
        "donors": sorted(sub_obs[C.DONOR_COL].unique().tolist()),
    }
    C.log(f"  [{arm:9s}] {info['n_cells']:6d} cells over {len(conds):3d} groups "
          f"({per_group}/group, {per_donor}/donor), "
          f"{info['n_cell_types']} cell types")
    return info


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--shard_dir", default=C.SHARD_DIR)
    ap.add_argument("--manifest", default=C.MANIFEST_PATH)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shard_info = verify_shards(args.shard_dir, exclude_donors=C.VAL_DONORS,
                               need_indices=C.MAIN_TUBE_INDICES)
    all_conds = [c for c in shard_info["conditions"] if c != C.CONTROL]
    C.log(f"[conditions] {len(all_conds)} stimuli + {C.CONTROL}")

    panel = C.draw_panel(all_conds)
    C.log(f"[panel] seeded-random {len(panel)} of {len(all_conds)}: {panel}")
    subsets = {a: C.draw_encoder_subset(all_conds, a) for a in C.ARMS}
    for a in C.ARMS:
        C.log(f"[arm:{a:9s}] encoder sees {len(subsets[a])} stimuli + {C.CONTROL}")

    manifest = C.read_json(args.manifest)
    manifest = [e for e in manifest if e["donor"] not in set(C.VAL_DONORS)]
    gene_names = list(C.read_json(
        str(Path(args.manifest).parent / "hvg_list.json")
    ))
    C.log(f"[manifest] {len(manifest)} tubes after excluding {C.VAL_DONORS}; "
          f"{len(gene_names)} HVGs")

    C.log("\n[bank] building the unique-cell bank (one pass)")
    X, obs, group_index, avail, donors = build_bank(manifest, gene_names)
    avail.to_csv(out / "cell_availability.csv", index=False)

    pbs_banked = int(avail.loc[avail.cytokine == C.CONTROL, "n_banked"].sum())
    pbs_avail = int(avail.loc[avail.cytokine == C.CONTROL, "n_unique_available"].sum())
    budget = min(C.STAGE1_TOTAL_CELLS, pbs_banked)
    C.log(f"\n[budget] PBS unique cells: {pbs_avail} available, {pbs_banked} banked")
    if budget < C.STAGE1_TOTAL_CELLS:
        C.log(f"[budget] LOWERED to {budget} for ALL arms (pbs_only cannot reach "
              f"{C.STAGE1_TOTAL_CELLS}); the sweep stays controlled at the smaller size.")
    else:
        C.log(f"[budget] {budget} cells per arm, as targeted.")

    C.log("\n[arms] writing per-arm Stage-1 sets")
    arm_info = {
        a: write_arm(a, subsets[a], X, obs, group_index, donors, budget,
                     gene_names, out, shard_info, args.shard_dir)
        for a in C.ARMS
    }

    C.write_json(out / "encsweep_meta.json", {
        "panel": panel,
        "panel_seed": C.PANEL_SEED,
        "encoder_subsets": subsets,
        "encoder_subset_seed": C.ENCODER_SUBSET_SEED,
        "stage1_budget": int(budget),
        "stage1_budget_target": C.STAGE1_TOTAL_CELLS,
        "pbs_unique_available": pbs_avail,
        "arms": arm_info,
        "shards_sha256": shard_info["shards_sha256"],
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "donors": donors,
        "n_genes": len(gene_names),
        "pinned": {
            "embed_dim": C.EMBED_DIM, "hidden_dims": list(C.HIDDEN_DIMS),
            "attention_hidden_dim": C.ATTENTION_HIDDEN_DIM,
            "stage1_epochs": C.STAGE1_EPOCHS, "stage1_lr": C.STAGE1_LR,
            "stage2_epochs": C.STAGE2_EPOCHS, "stage2_lr": C.STAGE2_LR,
            "top_n": C.TOP_N, "n_ig_steps": C.N_IG_STEPS,
            "early_stopping": False,
        },
    })
    C.mark_done(out, "prepare")
    C.log("\n[done] four arm Stage-1 sets + encsweep_meta.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
