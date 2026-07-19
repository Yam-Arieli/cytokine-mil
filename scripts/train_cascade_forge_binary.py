"""
Per-label BINARY (label vs PBS) AB-MIL training on cascade_forge, sharing ONE frozen
Stage-1 encoder across all labels -- mirrors the project's binary-IG convention (CLAUDE.md
Section 26.2: "the binary models share one frozen Stage-1 encoder so the S_X are
comparable").

Motivation: the multiclass read (train_cascade_forge_full.py -> source_potency) has a
confound the user flagged -- plateau timing in a shared 21-class softmax is sensitive to
how similar a label's signature is to OTHER labels in the same softmax, not just to its
own cascade depth. Two labels that happen to be transcriptionally close to EACH OTHER can
take longer to separate regardless of true cascade role. A per-label binary model (label
vs PBS only) removes that cross-label confusability confound entirely: each model's only
task is "this condition's cells vs resting," so its plateau timing tracks how distinguishable
the condition's OWN signature is from PBS, independent of what any other label looks like.

Writes dynamics.pkl in the SAME schema as train_cascade_forge_full.py's output (records /
val_records / logged_epochs, PBS already excluded), so the existing
validate_source_potency_cascade_forge.py and plot_cascade_forge_trajectories.py scripts
work UNCHANGED against this output directory.

Usage:
    python scripts/train_cascade_forge_binary.py \
        --manifest results/cascade_forge_potency/pseudotubes/manifest.json \
        --gene_list results/cascade_forge_potency/pseudotubes/gene_list.json \
        --val_donors donor_8,donor_9 \
        --output_dir results/cascade_forge_potency_binary/seed_42 --seed 42 \
        --stage1_epochs 30 --stage1_lr 0.01 --stage2_epochs 4000 --lr 0.001
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset
from cytokine_mil.experiment_setup import (
    build_encoder, build_mil_model, build_stage1_manifest, make_binary_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder
from cytokine_mil.training.train_mil import train_mil

EMBED_DIM = 128
ATTENTION_HIDDEN_DIM = 64
STAGE1_MOMENTUM = 0.9
STAGE2_MOMENTUM = 0.9


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--gene_list", required=True)
    p.add_argument("--val_donors", required=True, help="comma-separated donor names")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage1_epochs", type=int, default=30)
    p.add_argument("--stage1_lr", type=float, default=0.01)
    p.add_argument("--stage2_epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    p.add_argument("--attention_hidden_dim", type=int, default=ATTENTION_HIDDEN_DIM)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--control", default="PBS")
    return p.parse_args()


def main():
    args = _parse_args()
    val_donors = [d.strip() for d in args.val_donors.split(",") if d.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train.log", "w")

    def log(msg=""):
        print(msg)
        print(msg, file=log_file, flush=True)

    log(f"Output directory: {out_dir}")
    log(f"Seed: {args.seed}")
    log(f"Val donors: {val_donors}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    with open(args.gene_list) as fh:
        gene_names = json.load(fh)
    log(f"  {len(manifest)} tubes, {len(gene_names)} genes")

    labels = sorted({m["cytokine"] for m in manifest if m["cytokine"] != args.control})
    log(f"  {len(labels)} labels: {labels}")

    # ------------------------------------------------------------------
    # Stage 1: ONE shared cell-type encoder, frozen for every binary model below.
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("STAGE 1: Encoder pre-training (shared across all binary models)")
    log("=" * 60)
    stage1_manifest_path = out_dir / "manifest_stage1.json"
    build_stage1_manifest(manifest, save_path=str(stage1_manifest_path))
    cell_dataset = CellDataset(str(stage1_manifest_path), gene_names=gene_names, preload=True)
    cell_loader = DataLoader(cell_dataset, batch_size=256, shuffle=True, num_workers=0)
    n_cell_types = len(cell_dataset.cell_type_to_idx)
    encoder = build_encoder(n_input_genes=len(gene_names), n_cell_types=n_cell_types,
                             embed_dim=args.embed_dim)
    log(f"  Encoder: input={len(gene_names)}, embed_dim={args.embed_dim}, "
        f"n_cell_types={n_cell_types}")
    train_encoder(encoder=encoder, dataloader=cell_loader, n_epochs=args.stage1_epochs,
                  lr=args.stage1_lr, momentum=STAGE1_MOMENTUM, device=device, verbose=True)
    torch.save(encoder.state_dict(), out_dir / "encoder_stage1.pt")
    log("  Saved: encoder_stage1.pt (shared)")

    # ------------------------------------------------------------------
    # Stage 2: one binary (label vs control) AB-MIL model per label, same frozen encoder.
    # ------------------------------------------------------------------
    combined_train_records = []
    combined_val_records = []
    logged_epochs_ref = None
    per_label_summary = {}

    for i, label in enumerate(labels):
        log("\n" + "=" * 60)
        log(f"STAGE 2 [{i + 1}/{len(labels)}]: binary AB-MIL, {label} vs {args.control}")
        log("=" * 60)
        bin_manifest, bin_label = make_binary_manifest(manifest, label, control=args.control)
        train_manifest, val_manifest = split_manifest_by_donor(bin_manifest, val_donors)
        train_m_path = out_dir / f"manifest_train_{label}.json"
        val_m_path = out_dir / f"manifest_val_{label}.json"
        with open(train_m_path, "w") as fh:
            json.dump(train_manifest, fh)
        with open(val_m_path, "w") as fh:
            json.dump(val_manifest, fh)

        train_dataset = PseudoTubeDataset(str(train_m_path), bin_label, gene_names=gene_names,
                                           preload=True)
        val_dataset = PseudoTubeDataset(str(val_m_path), bin_label, gene_names=gene_names,
                                         preload=True)
        model = build_mil_model(encoder, embed_dim=args.embed_dim,
                                 attention_hidden_dim=args.attention_hidden_dim,
                                 n_classes=bin_label.n_classes(), encoder_frozen=True)

        dynamics = train_mil(
            model, train_dataset, n_epochs=args.stage2_epochs, lr=args.lr,
            momentum=STAGE2_MOMENTUM, log_every_n_epochs=args.log_every, device=device,
            seed=args.seed, verbose=False, val_dataset=val_dataset,
        )
        logged_epochs_ref = dynamics["logged_epochs"]

        # Keep only this label's OWN trajectory -- the control-class (PBS) record in a
        # one-off binary run isn't a meaningful trajectory to pool across 20 separate models.
        own_train = [r for r in dynamics["records"] if r["cytokine"] == label]
        own_val = [r for r in dynamics["val_records"] if r["cytokine"] == label]
        combined_train_records.extend(own_train)
        combined_val_records.extend(own_val)

        train_traj = np.mean([r["p_correct_trajectory"] for r in own_train], axis=0)
        val_traj = (np.mean([r["p_correct_trajectory"] for r in own_val], axis=0)
                    if own_val else None)
        per_label_summary[label] = {
            "train_final": float(train_traj[-1]), "train_peak": float(train_traj.max()),
            "val_final": float(val_traj[-1]) if val_traj is not None else None,
            "val_peak": float(val_traj.max()) if val_traj is not None else None,
        }
        log(f"  {label}: train_final={train_traj[-1]:.4f} train_peak={train_traj.max():.4f}"
            + (f" val_final={val_traj[-1]:.4f}" if val_traj is not None else ""))

        train_m_path.unlink()
        val_m_path.unlink()

    dynamics_payload = {
        "records": combined_train_records,
        "val_records": combined_val_records,
        "logged_epochs": logged_epochs_ref,
        "labels": labels,
        "seed": args.seed,
        "val_donors": val_donors,
        "stage2_epochs": args.stage2_epochs,
        "stage2_lr": args.lr,
        "model_type": f"per_label_binary_vs_{args.control}",
    }
    with open(out_dir / "dynamics.pkl", "wb") as fh:
        pickle.dump(dynamics_payload, fh)
    log("\nSaved: dynamics.pkl (combined per-label binary trajectories, "
        f"{len(combined_train_records)} train records, {len(combined_val_records)} val records)")

    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump({
            "seed": args.seed, "timestamp": datetime.now().isoformat(),
            "labels": labels, "per_label": per_label_summary,
            "stage1_epochs": args.stage1_epochs, "stage2_epochs": args.stage2_epochs,
            "lr": args.lr, "val_donors": val_donors,
        }, fh, indent=2)
    log("Saved: run_summary.json\nDone.")
    log_file.close()


if __name__ == "__main__":
    sys.exit(main())
