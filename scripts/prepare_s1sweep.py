"""Stage 0 of the Stage-1 CONSTRUCTION sweep (CLAUDE.md §38.4).

The breadth sweep (§38.1/§38.3) held Stage-1 cell count fixed at 36K and varied how many
conditions the encoder saw. The ladder came out flat, so breadth is not the lever. What it
did NOT vary is the thing the published anchor actually does differently: its Stage-1 set is
built by `build_stage1_manifest`, which takes **one tube per condition** at `tube_idx == 0`
rotating donors — so each condition is contributed by exactly ONE donor, and the total is
~17 tubes (~7.5K cells) rather than 36K balanced over ten donors.

Four arms, each pair differing in exactly one variable:

    pub_replica        one tube per condition, rotating donors, D2/D3 INCLUDED
    pub_replica_clean  one tube per condition, rotating donors, D2/D3 excluded
    vol_small          donor-balanced, ~7.5K cells (the published Stage-1 magnitude)
    vol_large          donor-balanced, 36K cells   (= the breadth sweep's all90 regime)

    pub_replica     vs pub_replica_clean  ->  D2/D3 leakage, nothing else
    pub_replica_cln vs vol_large          ->  donor STRUCTURE, at matched-ish volume
    vol_small       vs vol_large          ->  VOLUME, at matched structure

`pub_replica` deliberately violates CLAUDE.md §16 by putting the held-out donors into
Stage 1. That is the point — it measures how large the published anchor's leakage is. Its
outputs are diagnostic only and must never seed a production fit.

Everything else is pinned at the published values via `_encsweep_config`, and the readout
panel is the SAME seeded-random 24, so these arms are directly comparable to §38.3's.

This script chooses no formula: it reads tubes, subsamples, and writes files.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _encsweep_config as C  # noqa: E402
from prepare_encsweep import build_bank, write_arm  # noqa: E402
from prepare_oes90_pure import _read_tube, verify_shards  # noqa: E402

OUT_DIR = C.REPO_ROOT / "results" / "s1sweep"

# The published Stage-1 magnitude: 16 cytokines + PBS = 17 tubes at ~440 cells each.
# A SCALE, not a cytokine choice — nothing here consults which cytokines those were.
VOL_SMALL_CELLS = 7500
VOL_LARGE_CELLS = C.STAGE1_TOTAL_CELLS  # 36000, the breadth sweep's budget

ARMS = ("pub_replica", "pub_replica_clean", "vol_small", "vol_large")


def replica_entries(manifest, conditions) -> list:
    """`build_stage1_manifest`'s rule: one tube_idx==0 entry per condition, rotating donors.

    Mirrors `cytokine_mil/experiment_setup.py:build_stage1_manifest` exactly — conditions
    sorted, donors sorted, condition i takes donor `i % n_donors` — so the condition/donor
    entanglement of the published encoder is reproduced rather than described.
    """
    by_cond = defaultdict(list)
    for e in manifest:
        if int(e.get("tube_idx", 0)) == 0 and e["cytokine"] in conditions:
            by_cond[e["cytokine"]].append(e)
    picked = []
    for i, cond in enumerate(sorted(by_cond)):
        entries = sorted(by_cond[cond], key=lambda e: e["donor"])
        picked.append(entries[i % len(entries)])
    return picked


def write_replica_arm(arm, entries, gene_names, out_dir, shard_info, shard_dir) -> dict:
    """Whole tubes, exactly as `CellDataset(preload=True)` consumed them."""
    import anndata as ad

    adir = C.arm_dir(out_dir, arm)
    adir.mkdir(parents=True, exist_ok=True)

    Xs, rows = [], []
    for e in sorted(entries, key=lambda e: (e["cytokine"], e["donor"])):
        X, ct, _ = _read_tube(e["path"], gene_names)
        Xs.append(X)
        rows.extend({C.CONDITION_COL: str(e["cytokine"]),
                     C.DONOR_COL: str(e["donor"]),
                     C.CELLTYPE_COL: str(c)} for c in ct)
    X_all = np.concatenate(Xs, axis=0)
    obs = pd.DataFrame(rows)
    obs.index = pd.Index([str(i) for i in range(len(obs))], name="cell")

    ad.AnnData(
        X=np.ascontiguousarray(X_all),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(gene_names, name="gene")),
    ).write_h5ad(adir / "stage1_cells.h5ad")

    obs.groupby([C.CONDITION_COL, C.DONOR_COL, C.CELLTYPE_COL]).size().rename(
        "n_cells").reset_index().to_csv(adir / "stage1_composition.csv", index=False)

    # Stage-2 tubes ALWAYS exclude D2/D3, in every arm — only Stage 1 differs.
    C.write_json(adir / "tube_split.json", {
        "shard_dir": str(shard_dir),
        "shards_sha256": shard_info["shards_sha256"],
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "reserve_tube_indices": C.RESERVE_TUBE_INDICES,
    })

    donors = sorted(obs[C.DONOR_COL].unique().tolist())
    info = {
        "arm": arm,
        "construction": "one tube per condition, rotating donors (build_stage1_manifest)",
        "n_encoder_conditions": int(obs[C.CONDITION_COL].nunique()),
        "n_stimulus_conditions": int(obs[C.CONDITION_COL].nunique()) - 1,
        "n_cells": int(len(obs)),
        "n_tubes": len(entries),
        "mean_tube_cells": float(len(obs) / max(len(entries), 1)),
        "n_cell_types": int(obs[C.CELLTYPE_COL].nunique()),
        "donors": donors,
        "n_donors": len(donors),
        "donors_per_condition": 1,
        "includes_val_donors": bool(set(donors) & set(C.VAL_DONORS)),
    }
    C.log(f"  [{arm:18s}] {info['n_cells']:6d} cells over {info['n_tubes']:3d} tubes "
          f"({info['mean_tube_cells']:.0f}/tube), {info['n_donors']} donors, "
          f"{info['n_cell_types']} cell types"
          + ("  << D2/D3 IN (leaky, diagnostic only)" if info["includes_val_donors"] else ""))
    return info


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--shard_dir", default=C.SHARD_DIR)
    ap.add_argument("--manifest", default=C.MANIFEST_PATH)
    # Exposed so the local demo can run at toy scale; the DAG uses the defaults.
    ap.add_argument("--vol_small", type=int, default=VOL_SMALL_CELLS)
    ap.add_argument("--vol_large", type=int, default=VOL_LARGE_CELLS)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shard_info = verify_shards(args.shard_dir, exclude_donors=C.VAL_DONORS,
                               need_indices=C.MAIN_TUBE_INDICES)
    all_conds = [c for c in shard_info["conditions"] if c != C.CONTROL]
    C.log(f"[conditions] {len(all_conds)} stimuli + {C.CONTROL}")

    panel = C.draw_panel(all_conds)
    C.log(f"[panel] seeded-random {len(panel)} of {len(all_conds)} (same as §38.3): {panel}")

    manifest_all = C.read_json(args.manifest)
    manifest_clean = [e for e in manifest_all if e["donor"] not in set(C.VAL_DONORS)]
    gene_names = list(C.read_json(str(Path(args.manifest).parent / "hvg_list.json")))
    C.log(f"[manifest] {len(manifest_all)} tubes total, "
          f"{len(manifest_clean)} after excluding {C.VAL_DONORS}; {len(gene_names)} HVGs")

    every_cond = set(all_conds) | {C.CONTROL}
    arm_info = {}

    C.log("\n[replica arms] one tube per condition, rotating donors")
    arm_info["pub_replica"] = write_replica_arm(
        "pub_replica", replica_entries(manifest_all, every_cond),
        gene_names, out, shard_info, args.shard_dir)
    arm_info["pub_replica_clean"] = write_replica_arm(
        "pub_replica_clean", replica_entries(manifest_clean, every_cond),
        gene_names, out, shard_info, args.shard_dir)

    C.log("\n[bank] building the unique-cell bank for the volume arms (one pass)")
    X, obs, group_index, avail, donors = build_bank(manifest_clean, gene_names)
    avail.to_csv(out / "cell_availability.csv", index=False)

    C.log("\n[volume arms] donor-balanced over every condition")
    for arm, budget in (("vol_small", args.vol_small), ("vol_large", args.vol_large)):
        info = write_arm(arm, sorted(all_conds), X, obs, group_index, donors, budget,
                         gene_names, out, shard_info, args.shard_dir)
        info["construction"] = "donor-balanced subsample of the unique-cell bank"
        info["donors_per_condition"] = len(donors)
        info["includes_val_donors"] = False
        arm_info[arm] = info

    # Every arm's encoder sees every condition (breadth is settled, §38.3), so `seen_contrast`
    # is uninformative here by design; the key is written so the shared analyzer still runs.
    subsets = {a: sorted(all_conds) for a in ARMS}

    C.write_json(out / "encsweep_meta.json", {
        "sweep": "stage1_construction",
        "panel": panel,
        "panel_seed": C.PANEL_SEED,
        "encoder_subsets": subsets,
        "arms": arm_info,
        "vol_small_cells": args.vol_small,
        "vol_large_cells": args.vol_large,
        "shards_sha256": shard_info["shards_sha256"],
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "donors": donors,
        "n_genes": len(gene_names),
        "contrasts": {
            "d2d3_leakage": ["pub_replica", "pub_replica_clean"],
            "donor_structure": ["pub_replica_clean", "vol_large"],
            "stage1_volume": ["vol_small", "vol_large"],
        },
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
    C.log("\n[done] four Stage-1 construction arms + encsweep_meta.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
