"""
Full 91-class Oesinghaus AB-MIL training for confusion dynamics analysis.

Trains a shared Stage 1 encoder (cell-type classification) across all 90
cytokines + PBS, then trains a single 91-class AB-MIL (Stage 2, frozen encoder).

Full softmax output per tube per epoch is stored in records['softmax_trajectory'],
enabling post-training confusion dynamics analysis via analysis/confusion_dynamics.py.

Pre-registered cascade predictions (CLAUDE.md Section 19.5):
  Exp 1 (positive control): C(IL-12, IFN-gamma, t) > C(IFN-gamma, IL-12, t) at late epochs.
  Exp 2 (negative control): C(IL-6, IL-10, t) ≈ C(IL-10, IL-6, t) (symmetric, early-onset).

Val donors: Donor2, Donor3 (held out — observer-only, no gradient updates).

Results saved to:
    results/oesinghaus_full/run_{timestamp}/

Usage:
    python scripts/train_oesinghaus_full.py
    python scripts/train_oesinghaus_full.py --seed 123
    python scripts/train_oesinghaus_full.py --seed 7 --output_dir /path/to/dir
    python scripts/train_oesinghaus_full.py --stage2_epochs 150 --lr 0.005
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

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.models.instance_encoder import InstanceEncoder
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

MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH      = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
OUTPUT_BASE   = Path(__file__).parent.parent / "results" / "oesinghaus_full"
VAL_DONORS    = ["Donor2", "Donor3"]

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


# ---------------------------------------------------------------------------
# Pre-registration statement
# ---------------------------------------------------------------------------

def _print_preregistration(log):
    log("=" * 70)
    log("PRE-REGISTRATION STATEMENT")
    log("Experiment: Oesinghaus full 91-class confusion dynamics")
    log("Hypothesis: cytokine cascade A→B is reflected by asymmetric confusion.")
    log("  C(A,B,t) > C(B,A,t) at late epochs → evidence for A→B cascade direction.")
    log("Positive control (must pass before biological analysis):")
    log("  IL-12 → IFN-gamma: C(IL-12,IFN-gamma,t) > C(IFN-gamma,IL-12,t), late-onset.")
    log("Negative control:")
    log("  IL-6 / IL-10: symmetric confusion, early-onset (shared STAT3 pathway).")
    log("Cascade graph: FDR-corrected (BH, alpha=0.05), seed-stable (rho>0.7).")
    log("Val donors (observer-only): Donor2, Donor3")
    log("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Oesinghaus full 91-class AB-MIL training.")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    _print_preregistration(log)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load manifest & gene names
    # ------------------------------------------------------------------
    log("\nLoading manifest...")
    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    with open(HVG_PATH) as fh:
        gene_names = json.load(fh)
    log(f"  {len(manifest)} tubes, {len(gene_names)} HVGs")

    # ------------------------------------------------------------------
    # Label encoder (full 91 classes)
    # ------------------------------------------------------------------
    all_cytokines = sorted({e["cytokine"] for e in manifest if e["cytokine"] != "PBS"})
    label_enc = CytokineLabel().fit(manifest)
    log(f"  {label_enc.n_classes()} classes ({len(all_cytokines)} cytokines + PBS)")

    with open(out_dir / "label_encoder.json", "w") as fh:
        json.dump({"cytokines": list(label_enc.cytokines)}, fh)

    # ------------------------------------------------------------------
    # Stage 1 manifest (one tube per cytokine for encoder pre-training)
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
    log(f"  Val donors: {VAL_DONORS}")

    # ------------------------------------------------------------------
    # Stage 1: Encoder pre-training
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("STAGE 1: Encoder pre-training (cell-type classification)")
    log("=" * 60)

    cell_dataset = CellDataset(
        str(stage1_manifest_path), gene_names=gene_names, preload=True,
    )
    cell_loader = DataLoader(
        cell_dataset, batch_size=256, shuffle=True, num_workers=0,
    )

    n_cell_types = len(cell_dataset.cell_type_to_idx)
    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=n_cell_types,
        embed_dim=args.embed_dim,
    )
    log(f"  Encoder: input={len(gene_names)}, embed_dim={args.embed_dim}, "
        f"n_cell_types={n_cell_types}")

    train_encoder(
        encoder=encoder,
        dataloader=cell_loader,
        n_epochs=args.stage1_epochs,
        lr=args.stage1_lr,
        momentum=STAGE1_MOMENTUM,
        device=device,
        verbose=True,
    )
    torch.save(encoder.state_dict(), out_dir / "encoder_stage1.pt")
    log("  Saved: encoder_stage1.pt")

    # ------------------------------------------------------------------
    # Stage 2: MIL training (frozen encoder)
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("STAGE 2: AB-MIL training (frozen encoder, 91-class)")
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
    log(f"  Model: embed_dim={args.embed_dim}, attn_hidden={args.attention_hidden_dim}, "
        f"n_classes={label_enc.n_classes()}")
    log(f"  Stage 2: {args.stage2_epochs} epochs, lr={args.lr}, log_every={args.log_every}")

    dynamics = train_mil(
        model,
        train_dataset,
        n_epochs=args.stage2_epochs,
        lr=args.lr,
        momentum=STAGE2_MOMENTUM,
        log_every_n_epochs=args.log_every,
        device=device,
        seed=args.seed,
        verbose=True,
        val_dataset=val_dataset,
    )

    torch.save(model.state_dict(), out_dir / "model_stage2.pt")
    log("  Saved: model_stage2.pt")

    # ------------------------------------------------------------------
    # Save dynamics
    # ------------------------------------------------------------------
    log("\nSaving dynamics...")
    dynamics_payload = {
        "records":                      dynamics["records"],
        "val_records":                  dynamics["val_records"],
        "logged_epochs":                dynamics["logged_epochs"],
        "confusion_entropy_trajectory": dynamics["confusion_entropy_trajectory"],
        "val_confusion_entropy_trajectory": dynamics["val_confusion_entropy_trajectory"],
        "loss_components":              dynamics["loss_components"],
        "label_encoder_cytokines":      list(label_enc.cytokines),
        "seed":                         args.seed,
        "val_donors":                   VAL_DONORS,
        "stage2_epochs":                args.stage2_epochs,
        "stage2_lr":                    args.lr,
    }
    with open(out_dir / "dynamics.pkl", "wb") as fh:
        pickle.dump(dynamics_payload, fh)
    log("  Saved: dynamics.pkl")
    log(f"  Records: {len(dynamics['records'])} train, {len(dynamics['val_records'])} val")
    if dynamics["records"]:
        sm = dynamics["records"][0].get("softmax_trajectory")
        if sm is not None:
            log(f"  softmax_trajectory shape per record: {sm.shape} (K x T)")

    # ------------------------------------------------------------------
    # Loss curve plot
    # ------------------------------------------------------------------
    _plot_loss_curve(dynamics["loss_components"], dynamics["logged_epochs"], out_dir, log)

    # ------------------------------------------------------------------
    # Learnability summary
    # ------------------------------------------------------------------
    _print_learnability_summary(dynamics, label_enc, log)

    # ------------------------------------------------------------------
    # Quick-access run summary (avoids loading full dynamics.pkl later)
    # ------------------------------------------------------------------
    _save_run_summary(dynamics, args, out_dir, log)

    log("\nDone.")
    log_file.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_loss_curve(loss_components, logged_epochs, out_dir, log):
    epochs = range(1, len(loss_components["total"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(epochs), loss_components["total"], label="total loss", color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 2 Training Loss — Oesinghaus Full 91-class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    log("  Saved: loss_curve.png")


def _save_run_summary(dynamics, args, out_dir, log):
    """Save run_summary.json — quick-access metrics without loading dynamics.pkl."""
    train_recs = dynamics["records"]
    val_recs   = dynamics.get("val_records", [])

    def _safe_mean(recs, key="p_correct_trajectory", agg=lambda x: x[-1]):
        if not recs:
            return None
        vals = [agg(r[key]) for r in recs if key in r]
        return float(np.mean(vals)) if vals else None

    final_train = _safe_mean(train_recs, agg=lambda x: x[-1])
    auc_train   = _safe_mean(train_recs, agg=np.mean)
    final_val   = _safe_mean(val_recs,   agg=lambda x: x[-1])
    auc_val     = _safe_mean(val_recs,   agg=np.mean)

    loss_total = dynamics.get("loss_components", {}).get("total", [])
    summary = {
        "seed":                    args.seed,
        "timestamp":               datetime.now().isoformat(),
        "final_train_p_correct":   final_train,
        "auc_train_p_correct":     auc_train,
        "final_val_p_correct":     final_val,
        "auc_val_p_correct":       auc_val,
        "final_loss":              float(loss_total[-1]) if loss_total else None,
        "n_train_records":         len(train_recs),
        "n_val_records":           len(val_recs),
        "stage1_epochs":           args.stage1_epochs,
        "stage2_epochs":           args.stage2_epochs,
        "stage2_lr":               args.lr,
        "embed_dim":               args.embed_dim,
        "attention_hidden_dim":    args.attention_hidden_dim,
        "val_donors":              VAL_DONORS,
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"  Saved: run_summary.json  "
        f"(train_final={final_train:.4f}, val_final={final_val:.4f})")


def _print_learnability_summary(dynamics, label_enc, log):
    from cytokine_mil.analysis.dynamics import rank_cytokines_by_learnability
    records = dynamics["records"]
    if not records:
        return
    try:
        ranking = rank_cytokines_by_learnability(records)
        log("\nTop-10 most learnable cytokines (train, donor-aggregated):")
        for i, entry in enumerate(ranking["ranking"][:10]):
            log(f"  {i+1:2d}. {entry['cytokine']:<20s}  AUC={entry['auc']:.4f}")
        log("\nBottom-10 cytokines:")
        for i, entry in enumerate(ranking["ranking"][-10:]):
            rank = len(ranking["ranking"]) - 9 + i
            log(f"  {rank:2d}. {entry['cytokine']:<20s}  AUC={entry['auc']:.4f}")
        log(f"\n{ranking['metric_description']}")
    except Exception as e:
        log(f"  (learnability summary skipped: {e})")


if __name__ == "__main__":
    main()
