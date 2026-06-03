"""
Train binary AB-MIL models for the cytokines MISSING from the existing
binary_ig.parquet, so the full Path A -> Bridge -> Path B pipeline can be run
over ALL 121 Path A coupled axes (CLAUDE.md §27.1), not just the 53 currently
evaluable. The missing set is computed at runtime:

    missing = {cytokines in cytokine_axes.csv} - {cytokines in binary_ig.parquet}

(expected ~21-24 of the 45). This script is CHUNKED for a SLURM job array so
the ~21-24 models stay safely under the 12h `short` partition: each array task
trains a contiguous slice `missing[chunk_id]` (own shared Stage-1 encoder, own
binaries), writing per-cytokine `model_<cyt>.pt` to a SHARED output dir
(disjoint filenames across chunks -> no collision). The IG probe then loads the
union.

Wide HPs match the existing trained pool (embed=512, hidden=(512,512),
attn=128, Stage1 20@0.005, Stage2 250@3e-5) so the IG probe loads all models
uniformly via state-dict shape inference, and the new S_X are comparable to the
existing ones.

Usage (one array task):
  python scripts/train_oesinghaus_binary_groupu.py \
      --output_dir results/group_u/binary_missing \
      --n_chunks 8 --chunk_id $SLURM_ARRAY_TASK_ID
"""

import argparse
import copy
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset  # noqa: E402
from cytokine_mil.experiment_setup import (  # noqa: E402
    build_mil_model,
    build_stage1_manifest,
    filter_manifest,
    make_binary_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.models.instance_encoder import InstanceEncoder  # noqa: E402
from cytokine_mil.training.train_encoder import train_encoder  # noqa: E402
from cytokine_mil.training.train_mil import train_mil  # noqa: E402


MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
AXES_CSV = REPO_ROOT / "reports/cascade_pairs/cytokine_axes.csv"
# Canonical existing coverage = the merged 24-cytokine parquet the §26/full19
# pipeline ran on (8 original + 16 missing). NOT the 8-cytokine binary_ig/.
EXISTING_BINARY_IG = REPO_ROOT / "results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet"
VAL_DONORS = ["Donor2", "Donor3"]
CONTROL = "PBS"

# Wide HPs — match the existing trained pool so IG loads uniformly
EMBED_DIM = 512
HIDDEN_DIMS = (512, 512)
ATTENTION_HIDDEN_DIM = 128
STAGE1_EPOCHS = 20
STAGE1_LR = 0.005
STAGE1_MOMENTUM = 0.9
STAGE2_EPOCHS = 250
STAGE2_LR = 0.00003
STAGE2_MOMENTUM = 0.90
LOG_EVERY = 5
SEED = 42


def _compute_missing(axes_csv: Path, existing_ig: Path, manifest_cytokines: set, log) -> list:
    import pandas as pd
    axes = pd.read_csv(axes_csv)
    axis_cyts = set(axes["axis_a"]) | set(axes["axis_b"])
    if existing_ig.exists():
        ig = pd.read_parquet(existing_ig)
        have = set(ig["cytokine"].unique())
    else:
        have = set()
        log(f"WARN: existing binary_ig not found at {existing_ig}; treating all "
            f"axis cytokines as missing.")
    missing = sorted(axis_cyts - have)
    # only those present in the manifest can be trained
    trainable = [c for c in missing if c in manifest_cytokines]
    dropped = [c for c in missing if c not in manifest_cytokines]
    if dropped:
        log(f"WARN: {len(dropped)} missing cytokines not in manifest (skipped): {dropped}")
    log(f"axis cytokines={len(axis_cyts)}  already have S_X={len(have)}  "
        f"missing(trainable)={len(trainable)}")
    return trainable


def _train_one_binary_model(target, control, manifest, gene_names, shared_encoder,
                            out_dir, device, seed, log, embed_dim, hidden_dims,
                            attention_hidden_dim):
    log(f"\n{'=' * 50}\nTraining binary model: {target} vs {control}\n{'=' * 50}")
    bin_manifest, label_enc = make_binary_manifest(manifest, target, control=control)
    train_m, val_m = split_manifest_by_donor(bin_manifest, VAL_DONORS)
    log(f"  {target}: {len(train_m)} train tubes, {len(val_m)} val tubes")

    safe_target = target.replace("/", "_")
    train_m_path = out_dir / f"manifest_train_{safe_target}.json"
    val_m_path = out_dir / f"manifest_val_{safe_target}.json"
    with open(train_m_path, "w") as fh:
        json.dump(train_m, fh)
    with open(val_m_path, "w") as fh:
        json.dump(val_m, fh)
    with open(out_dir / f"label_encoder_{safe_target}.json", "w") as fh:
        json.dump({"positive": label_enc.positive, "negative": label_enc.negative}, fh)

    train_dataset = PseudoTubeDataset(str(train_m_path), label_enc, gene_names=gene_names, preload=True)
    val_dataset = PseudoTubeDataset(str(val_m_path), label_enc, gene_names=gene_names, preload=True)

    encoder = copy.deepcopy(shared_encoder)
    model = build_mil_model(
        encoder, embed_dim=embed_dim, attention_hidden_dim=attention_hidden_dim,
        n_classes=label_enc.n_classes(), encoder_frozen=True,
    )
    dynamics = train_mil(
        model, train_dataset, n_epochs=STAGE2_EPOCHS, lr=STAGE2_LR,
        momentum=STAGE2_MOMENTUM, log_every_n_epochs=LOG_EVERY, device=device,
        seed=seed, verbose=True, val_dataset=val_dataset,
    )
    torch.save(model.state_dict(), out_dir / f"model_{safe_target}.pt")
    with open(out_dir / f"dynamics_{safe_target}.pkl", "wb") as fh:
        pickle.dump({
            "records": dynamics["records"], "val_records": dynamics["val_records"],
            "logged_epochs": dynamics["logged_epochs"], "condition": target,
            "control": control,
        }, fh)
    log(f"  Saved: model_{safe_target}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "results/group_u/binary_missing"))
    parser.add_argument("--axes_csv", type=str, default=str(AXES_CSV))
    parser.add_argument("--existing_binary_ig", type=str, default=str(EXISTING_BINARY_IG))
    parser.add_argument("--cytokines", nargs="+", default=None,
                        help="Override the computed missing list (full set).")
    parser.add_argument("--n_chunks", type=int, default=8)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=list(HIDDEN_DIMS))
    parser.add_argument("--attention_hidden_dim", type=int, default=ATTENTION_HIDDEN_DIM)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"run_log_chunk{args.chunk_id}.txt"

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    for required in ("numpy", "torch", "scanpy", "anndata", "pandas"):
        try:
            __import__(required)
        except ImportError as e:
            print(f"FATAL: missing dependency '{required}': {e}", flush=True)
            sys.exit(2)

    seed = args.seed
    embed_dim = args.embed_dim
    hidden_dims = tuple(args.hidden_dims)
    attention_hidden_dim = args.attention_hidden_dim

    log("=" * 60)
    log(f"Oesinghaus binary training — Group-U missing cytokines "
        f"(chunk {args.chunk_id}/{args.n_chunks})")
    log("=" * 60)

    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    with open(HVG_PATH) as fh:
        gene_names = json.load(fh)
    manifest_cytokines = {e["cytokine"] for e in manifest}

    if args.cytokines is not None:
        full_missing = sorted(args.cytokines)
        log(f"Using override cytokine list: {full_missing}")
    else:
        full_missing = _compute_missing(
            Path(args.axes_csv), Path(args.existing_binary_ig), manifest_cytokines, log,
        )
    log(f"Full missing set ({len(full_missing)}): {full_missing}")

    # contiguous chunk slice
    n = len(full_missing)
    chunk_size = (n + args.n_chunks - 1) // args.n_chunks
    lo = args.chunk_id * chunk_size
    hi = min(lo + chunk_size, n)
    my_cytokines = full_missing[lo:hi] if lo < n else []
    log(f"This chunk trains [{lo}:{hi}] -> {my_cytokines}")

    # record this chunk's cytokines for the downstream IG --targets union
    (out_dir / f"groupu_cytokines_chunk{args.chunk_id}.txt").write_text(
        " ".join(my_cytokines) + "\n"
    )
    if not my_cytokines:
        log("Empty chunk -> nothing to train. Exiting 0.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    filtered_manifest = filter_manifest(manifest, my_cytokines, include_pbs=True)
    log(f"Filtered manifest size: {len(filtered_manifest)} entries")

    # Shared Stage-1 encoder (cell-type classification) on this chunk + PBS
    stage1_path = out_dir / f"manifest_stage1_chunk{args.chunk_id}.json"
    stage1_manifest = build_stage1_manifest(filtered_manifest, save_path=str(stage1_path))
    cell_dataset = CellDataset(str(stage1_path), gene_names=gene_names, preload=True)
    cell_loader = DataLoader(cell_dataset, batch_size=256, shuffle=True, num_workers=0)
    log(f"Stage 1 cells: {len(cell_dataset)}  cell types: {cell_dataset.n_cell_types()}")

    encoder = InstanceEncoder(
        input_dim=len(gene_names), embed_dim=embed_dim,
        n_cell_types=cell_dataset.n_cell_types(), hidden_dims=hidden_dims,
    )
    train_encoder(encoder, cell_loader, n_epochs=STAGE1_EPOCHS, lr=STAGE1_LR,
                  momentum=STAGE1_MOMENTUM, device=device)
    torch.save(encoder.state_dict(), out_dir / f"encoder_chunk{args.chunk_id}.pt")
    log("Shared encoder trained.")

    n_done = 0
    for i, target in enumerate(my_cytokines, start=1):
        log(f"\n[{i}/{len(my_cytokines)}] {target} START")
        try:
            _train_one_binary_model(
                target=target, control=CONTROL, manifest=filtered_manifest,
                gene_names=gene_names, shared_encoder=encoder, out_dir=out_dir,
                device=device, seed=seed, log=log, embed_dim=embed_dim,
                hidden_dims=hidden_dims, attention_hidden_dim=attention_hidden_dim,
            )
            n_done += 1
            log(f"[{i}/{len(my_cytokines)}] {target} DONE")
        except Exception as e:
            import traceback
            log(f"[{i}/{len(my_cytokines)}] {target} ERROR: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            continue

    log(f"\nChunk {args.chunk_id} done: {n_done}/{len(my_cytokines)} trained.")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
