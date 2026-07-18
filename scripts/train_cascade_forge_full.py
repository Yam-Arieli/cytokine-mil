"""
21-class (20 labels + PBS) AB-MIL training on the cascade_forge synthetic dataset, for
validating source_potency (training-dynamics plateau shape) against a KNOWN cascade graph.

Trimmed relative to train_oesinghaus_full.py: no centroid tracking, no checkpoints, no
Stage 3 -- this run's only purpose is Stage1 cell-type pretrain + Stage2 multiclass AB-MIL
with per-cytokine p_correct trajectories logged to convergence (a real train plateau), so
cytokine_mil.analysis.source_potency can be computed and validated against the exact
authored cascade_forge graph (scripts/build_pseudotubes_cascade_forge.py) instead of a
partial, hand-audited real graph.

Usage:
    python scripts/train_cascade_forge_full.py \
        --manifest results/cascade_forge_potency/pseudotubes/manifest.json \
        --gene_list results/cascade_forge_potency/pseudotubes/gene_list.json \
        --val_donors donor_8,donor_9 \
        --output_dir results/cascade_forge_potency/seed_42 --seed 42 \
        --stage1_epochs 30 --stage2_epochs 300 --lr 0.01
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import (
    build_encoder, build_mil_model, build_stage1_manifest, split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder
from cytokine_mil.training.train_mil import train_mil

EMBED_DIM = 128
HIDDEN_DIMS = (512, 256)
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
    p.add_argument("--stage2_epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    p.add_argument("--attention_hidden_dim", type=int, default=ATTENTION_HIDDEN_DIM)
    p.add_argument("--log_every", type=int, default=1)
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

    label_enc = CytokineLabel().fit(manifest)
    log(f"  {label_enc.n_classes()} classes: {list(label_enc.cytokines)}")
    with open(out_dir / "label_encoder.json", "w") as fh:
        json.dump({"cytokines": list(label_enc.cytokines)}, fh)

    stage1_manifest_path = out_dir / "manifest_stage1.json"
    stage1_manifest = build_stage1_manifest(manifest, save_path=str(stage1_manifest_path))
    log(f"  Stage1 manifest: {len(stage1_manifest)} tubes (one per label)")

    train_manifest, val_manifest = split_manifest_by_donor(manifest, val_donors)
    train_m_path = out_dir / "manifest_train.json"
    val_m_path = out_dir / "manifest_val.json"
    with open(train_m_path, "w") as fh:
        json.dump(train_manifest, fh)
    with open(val_m_path, "w") as fh:
        json.dump(val_manifest, fh)
    log(f"  Train: {len(train_manifest)} tubes | Val: {len(val_manifest)} tubes")

    # ------------------------------------------------------------------
    # Stage 1: cell-type encoder pretraining
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("STAGE 1: Encoder pre-training (cell-type classification)")
    log("=" * 60)
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
    log("  Saved: encoder_stage1.pt")

    # ------------------------------------------------------------------
    # Stage 2: multiclass AB-MIL (frozen encoder)
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log(f"STAGE 2: AB-MIL training (frozen encoder, {label_enc.n_classes()}-class)")
    log("=" * 60)
    train_dataset = PseudoTubeDataset(str(train_m_path), label_enc, gene_names=gene_names,
                                       preload=True)
    val_dataset = PseudoTubeDataset(str(val_m_path), label_enc, gene_names=gene_names,
                                     preload=True)
    model = build_mil_model(encoder, embed_dim=args.embed_dim,
                             attention_hidden_dim=args.attention_hidden_dim,
                             n_classes=label_enc.n_classes(), encoder_frozen=True)
    log(f"  Stage 2: {args.stage2_epochs} epochs, lr={args.lr}, log_every={args.log_every}")

    dynamics = train_mil(
        model, train_dataset, n_epochs=args.stage2_epochs, lr=args.lr,
        momentum=STAGE2_MOMENTUM, log_every_n_epochs=args.log_every, device=device,
        seed=args.seed, verbose=True, val_dataset=val_dataset,
    )
    torch.save(model.state_dict(), out_dir / "model_stage2.pt")
    log("  Saved: model_stage2.pt")

    dynamics_payload = {
        "records": dynamics["records"],
        "val_records": dynamics["val_records"],
        "logged_epochs": dynamics["logged_epochs"],
        "label_encoder_cytokines": list(label_enc.cytokines),
        "seed": args.seed,
        "val_donors": val_donors,
        "stage2_epochs": args.stage2_epochs,
        "stage2_lr": args.lr,
    }
    with open(out_dir / "dynamics.pkl", "wb") as fh:
        pickle.dump(dynamics_payload, fh)
    log("  Saved: dynamics.pkl")
    log(f"  Records: {len(dynamics['records'])} train, {len(dynamics['val_records'])} val")

    epochs = dynamics["logged_epochs"]
    train_agg = np.mean([r["p_correct_trajectory"] for r in dynamics["records"]], axis=0)
    val_agg = (np.mean([r["p_correct_trajectory"] for r in dynamics["val_records"]], axis=0)
               if dynamics["val_records"] else None)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_agg, "-o", ms=3, color="tab:blue", label="train (mean over tubes)")
    if val_agg is not None:
        ax.plot(epochs, val_agg, "-o", ms=3, color="tab:red", label="val (mean over tubes)")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean p_correct")
    ax.set_title(f"cascade_forge Stage2 ({label_enc.n_classes()}-class), seed={args.seed}")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "aggregate_curve.png", dpi=150)
    plt.close(fig)
    log(f"  train_final={train_agg[-1]:.4f}"
        + (f"  val_final={val_agg[-1]:.4f}" if val_agg is not None else ""))

    summary = {
        "seed": args.seed, "timestamp": datetime.now().isoformat(),
        "n_classes": label_enc.n_classes(),
        "train_final": float(train_agg[-1]), "train_peak": float(train_agg.max()),
        "train_plateaued": bool(train_agg[-1] >= 0.98 * train_agg.max()),
        "val_final": float(val_agg[-1]) if val_agg is not None else None,
        "val_peak": float(val_agg.max()) if val_agg is not None else None,
        "stage1_epochs": args.stage1_epochs, "stage2_epochs": args.stage2_epochs,
        "lr": args.lr, "val_donors": val_donors,
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"  Saved: run_summary.json (train_plateaued={summary['train_plateaued']})")
    log("\nDone.")
    log_file.close()


if __name__ == "__main__":
    sys.exit(main())
