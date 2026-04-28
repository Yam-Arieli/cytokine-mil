"""
Train Stage 1 + Stage 2 AB-MIL on the synthetic cascade dataset.

Mirrors train_oesinghaus_full.py but uses synthetic_cascades_v1 data.
Val donors: Donor5, Donor6 (last 2 of 6).

Results saved to:
    results/synthetic_cascades/run_{timestamp}_seed{SEED}/

Usage:
    python scripts/train_synthetic_cascades.py
    python scripts/train_synthetic_cascades.py --seed 123
"""

import argparse
import json
import os
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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import (
    build_encoder,
    build_mil_model,
    build_stage1_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder
from cytokine_mil.training.train_mil import train_mil
from cytokine_mil.analysis.dynamics import aggregate_to_donor_level


# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

SYNTHETIC_DIR = Path("/cs/labs/mornitzan/yam.arieli/datasets/synthetic_cascades_v2")
MANIFEST_PATH = str(SYNTHETIC_DIR / "manifest.json")
HVG_PATH      = str(SYNTHETIC_DIR / "hvg_list.json")
OUTPUT_BASE   = REPO_ROOT / "results" / "synthetic_cascades"
VAL_DONORS    = ["Donor5", "Donor6"]

EMBED_DIM            = 128
HIDDEN_DIMS          = (512, 256)
ATTENTION_HIDDEN_DIM = 64
STAGE1_EPOCHS        = 50
STAGE1_LR            = 0.01
STAGE1_MOMENTUM      = 0.9
STAGE2_EPOCHS        = 100
STAGE2_LR            = 0.01
STAGE2_MOMENTUM      = 0.9
LOG_EVERY            = 1
SEED                 = 42


def _parse_args():
    p = argparse.ArgumentParser(description="Synthetic cascade AB-MIL training.")
    p.add_argument("--seed",           type=int,   default=SEED)
    p.add_argument("--output_dir",     type=str,   default=None)
    p.add_argument("--stage1_epochs",  type=int,   default=STAGE1_EPOCHS)
    p.add_argument("--stage2_epochs",  type=int,   default=STAGE2_EPOCHS)
    p.add_argument("--lr",             type=float, default=STAGE2_LR)
    p.add_argument("--stage1_lr",      type=float, default=STAGE1_LR)
    p.add_argument("--embed_dim",      type=int,   default=EMBED_DIM)
    p.add_argument("--attention_hidden_dim", type=int, default=ATTENTION_HIDDEN_DIM)
    p.add_argument("--log_every",      type=int,   default=LOG_EVERY)
    return p.parse_args()


def main():
    args = _parse_args()

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_BASE / f"run_{ts}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(out_dir / "train.log", "w")
    def log(msg=""):
        print(msg)
        print(msg, file=log_file, flush=True)

    log(f"Output directory: {out_dir}")
    log(f"Seed: {args.seed}")
    log(f"Val donors: {VAL_DONORS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load manifest & gene names
    # ------------------------------------------------------------------
    log("\nLoading synthetic manifest...")
    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    with open(HVG_PATH) as fh:
        gene_names = json.load(fh)
    log(f"  {len(manifest)} tubes, {len(gene_names)} genes")

    # ------------------------------------------------------------------
    # Label encoder
    # ------------------------------------------------------------------
    label_enc = CytokineLabel().fit(manifest)
    all_cytokines = sorted({e["cytokine"] for e in manifest if e["cytokine"] != "PBS"})
    log(f"  {len(all_cytokines)} cytokines + PBS  (model outputs: {label_enc.n_classes()})")

    # Save label_encoder.json with padded list so run_experiment_geo.py can
    # reconstruct the full index mapping (indices 0..n_classes-1).
    # Unused indices get empty-string placeholders.
    n_cls = label_enc.n_classes()           # 91
    cytokines_ordered = [""] * n_cls
    for idx, name in label_enc._idx_to_label.items():
        cytokines_ordered[int(idx)] = name
    with open(out_dir / "label_encoder.json", "w") as fh:
        json.dump({"cytokines": cytokines_ordered}, fh)

    # ------------------------------------------------------------------
    # Stage 1 manifest (one tube per cytokine)
    # ------------------------------------------------------------------
    log("\nBuilding Stage 1 manifest...")
    stage1_manifest_path = out_dir / "manifest_stage1.json"
    stage1_manifest = build_stage1_manifest(manifest, save_path=str(stage1_manifest_path))
    log(f"  {len(stage1_manifest)} tubes (one per cytokine)")

    # ------------------------------------------------------------------
    # Train/val split
    # ------------------------------------------------------------------
    log("\nSplitting train/val by donor...")
    train_manifest, val_manifest = split_manifest_by_donor(manifest, VAL_DONORS)
    train_m_path = out_dir / "manifest_train.json"
    val_m_path   = out_dir / "manifest_val.json"
    with open(train_m_path, "w") as fh:
        json.dump(train_manifest, fh)
    with open(val_m_path, "w") as fh:
        json.dump(val_manifest, fh)
    log(f"  Train: {len(train_manifest)} tubes | Val: {len(val_manifest)} tubes")

    # ------------------------------------------------------------------
    # Stage 1: Encoder pre-training
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("STAGE 1: Encoder pre-training (cell-type classification)")
    log("=" * 60)

    cell_dataset = CellDataset(
        str(stage1_manifest_path), gene_names=gene_names, preload=True,
    )
    cell_loader = DataLoader(cell_dataset, batch_size=256, shuffle=True, num_workers=0)

    n_cell_types = len(cell_dataset.cell_type_to_idx)
    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=n_cell_types,
        embed_dim=args.embed_dim,
    )
    log(f"  Encoder: {len(gene_names)} → {args.embed_dim}d  |  {n_cell_types} cell types")

    # train_encoder returns the trained encoder in-place (not a dict).
    encoder = train_encoder(
        encoder,
        cell_loader,
        n_epochs=args.stage1_epochs,
        lr=args.stage1_lr,
        momentum=STAGE1_MOMENTUM,
        device=device,
    )
    torch.save(encoder.state_dict(), out_dir / "encoder_stage1.pt")
    log(f"  Saved encoder_stage1.pt")

    # ------------------------------------------------------------------
    # Stage 2: MIL training (frozen encoder)
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("STAGE 2: MIL training (frozen encoder)")
    log("=" * 60)

    train_dataset = PseudoTubeDataset(
        str(train_m_path), label_enc, gene_names=gene_names, preload=True,
    )
    val_dataset = PseudoTubeDataset(
        str(val_m_path), label_enc, gene_names=gene_names, preload=True,
    )

    model = build_mil_model(
        encoder,
        embed_dim=args.embed_dim,
        attention_hidden_dim=args.attention_hidden_dim,
        n_classes=label_enc.n_classes(),
        encoder_frozen=True,
    )

    dynamics = train_mil(
        model,
        train_dataset,
        n_epochs=args.stage2_epochs,
        lr=args.lr,
        momentum=STAGE2_MOMENTUM,
        log_every_n_epochs=args.log_every,
        device=device,
        val_dataset=val_dataset,
    )

    torch.save(model.state_dict(), out_dir / "model_stage2.pt")
    log(f"  Saved model_stage2.pt")

    with open(out_dir / "dynamics_stage2.pkl", "wb") as fh:
        pickle.dump(dynamics, fh)

    # ------------------------------------------------------------------
    # Quick learnability summary
    # ------------------------------------------------------------------
    log("\nLearnability summary (donor-aggregated train AUC):")
    donor_traj = aggregate_to_donor_level(dynamics["records"])
    ranking = sorted(
        [(cyt, info["auc"]) for cyt, info in donor_traj.items() if cyt != "PBS"],
        key=lambda x: x[1], reverse=True,
    )
    for rank, (cyt, auc) in enumerate(ranking, 1):
        log(f"  {rank:2d}. {cyt:<20}  AUC={auc:.4f}")

    # Train/val loss curves
    if "loss_components" in dynamics:
        train_losses = dynamics["loss_components"].get("total", [])
        if train_losses:
            fig, ax = plt.subplots()
            ax.plot(train_losses, label="train loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy")
            ax.set_title("Stage 2 MIL loss"); ax.legend()
            fig.savefig(out_dir / "stage2_loss.png", dpi=120); plt.close(fig)

    # Save run summary
    summary = {
        "seed": args.seed,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "n_train_tubes": len(train_manifest),
        "n_val_tubes": len(val_manifest),
        "n_cytokines": len(all_cytokines),
        "top5_learnability": [cyt for cyt, _ in ranking[:5]],
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    log(f"\nAll outputs saved to {out_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
