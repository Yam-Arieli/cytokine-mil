"""
Sheu 2024 BMDM time-course — Stage 1 (encoder pretraining) + Stage 2 (AB-MIL).

Phase 1 of the Sheu axis-discovery experiment (CLAUDE.md §21).
Stage 3 deliberately omitted per the gated plan; only re-introduce if the
§21 axis gate is GREEN and direction-inference work continues.

Pre-registered TLR cascade axes (CLAUDE.md §21):
  MUST recover:
    1. LPS — TNF        (TLR4 → NF-κB → autocrine TNF)
    2. polyIC — IFNb    (TLR3/TRIF → IRF3 → type-I IFN)
  SHOULD recover:
    3. LPS — IFNb       (LPS engages TRIF arm too)
    4. P3CSK — CpG      (both MyD88-only, no TRIF)
    5. LPSlo — P3CSK    (both MyD88-biased)
  MUST NOT call:
    - P3CSK — IFNb      (TLR2 has no TRIF arm)
    - CpG — IFNb        (TLR9/IFN-α is pDC-restricted)
    - TNF — IFNb        (no cross-induction in macrophages)

Val pseudo-donors (held out, observer-only): M2_IL4_rep1, PM_B6.old_rep1.

Usage:
    python scripts/train_sheu2024_stage12.py --seed 42
    python scripts/train_sheu2024_stage12.py --seed 123 --output_dir ...
"""

from __future__ import annotations

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
from cytokine_mil.experiment_setup import (
    build_encoder,
    build_mil_model,
    build_stage1_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder
from cytokine_mil.training.train_mil import train_mil


# ---------------------------------------------------------------------------
# Paths & Constants (cluster defaults; override via CLI)
# ---------------------------------------------------------------------------

MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json"
HVG_PATH      = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json"
OUTPUT_BASE   = Path(__file__).parent.parent / "results" / "sheu2024_full"

# Pseudo-donor val split per CLAUDE.md §15. Single val pseudo-donor — only 4
# pseudo-donors are downloadable at 3hr (GEO batches 14-16 missing).
VAL_PSEUDO_DONORS = ["M2_IL4_rep1"]

# Phase 1 training hyperparameters
EMBED_DIM            = 128
ATTENTION_HIDDEN_DIM = 64
STAGE1_EPOCHS        = 50
STAGE1_LR            = 0.01
STAGE1_MOMENTUM      = 0.9
STAGE2_EPOCHS        = 100
STAGE2_LR            = 0.003
STAGE2_MOMENTUM      = 0.9
LR_WARMUP            = 5
LOG_EVERY            = 1
SEED                 = 42

# Pre-registered axis-discovery gate (informational; the gate itself is run
# post-training by scripts/run_latent_geometry.py + report_cytokine_axes.py).
PREREG_MUST_AXES = [
    ("LPS", "TNF"),
    ("polyIC", "IFNb"),
]
PREREG_SHOULD_AXES = [
    ("LPS", "IFNb"),
    ("P3CSK", "CpG"),
    ("LPSlo", "P3CSK"),
]
PREREG_MUST_NOT_AXES = [
    ("P3CSK", "IFNb"),
    ("CpG", "IFNb"),
    ("TNF", "IFNb"),
]


# ---------------------------------------------------------------------------
# Pre-registration statement
# ---------------------------------------------------------------------------

def _print_preregistration(log):
    log("=" * 70)
    log("PRE-REGISTRATION STATEMENT — Sheu 2024 phase 1 axis-discovery gate")
    log("Hypothesis: trained AB-MIL latent geometry on Sheu 3h BMDM data recovers")
    log("            textbook TLR cascade pairs as latent-space axes.")
    log("")
    log("MUST recover (failure ⇒ pipeline broken on this dataset OR signal absent):")
    for a, b in PREREG_MUST_AXES:
        log(f"  • {a} — {b}")
    log("SHOULD recover (textbook but secondary):")
    for a, b in PREREG_SHOULD_AXES:
        log(f"  • {a} — {b}")
    log("MUST NOT call (false-positive guards):")
    for a, b in PREREG_MUST_NOT_AXES:
        log(f"  • {a} — {b}")
    log("")
    log("Gate criterion (per axis): BH-FDR ≤ 0.05 on pseudo-donor-level Wilcoxon,")
    log("                          Spearman ρ ≥ 0.7 across 3 seeds.")
    log("Composite GREEN: 2/2 MUST + ≥2/3 SHOULD + 0/3 MUST-NOT + M0-only agreement.")
    log(f"Val pseudo-donors (observer-only): {VAL_PSEUDO_DONORS}")
    log("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Sheu 2024 BMDM Stage 1 + 2 training.")
    p.add_argument("--seed",           type=int,   default=SEED)
    p.add_argument("--output_dir",     type=str,   default=None)
    p.add_argument("--manifest_path",  type=str,   default=MANIFEST_PATH)
    p.add_argument("--hvg_path",       type=str,   default=HVG_PATH)
    p.add_argument("--stage1_epochs",  type=int,   default=STAGE1_EPOCHS)
    p.add_argument("--stage2_epochs",  type=int,   default=STAGE2_EPOCHS)
    p.add_argument("--lr",             type=float, default=STAGE2_LR,
                   help="Stage 2 learning rate.")
    p.add_argument("--stage1_lr",      type=float, default=STAGE1_LR)
    p.add_argument("--lr_warmup_epochs", type=int, default=LR_WARMUP)
    p.add_argument("--embed_dim",      type=int,   default=EMBED_DIM)
    p.add_argument("--attention_hidden_dim", type=int, default=ATTENTION_HIDDEN_DIM)
    p.add_argument("--log_every",      type=int,   default=LOG_EVERY)
    p.add_argument("--val_donors", nargs="*", default=VAL_PSEUDO_DONORS,
                   help="Pseudo-donor names to hold out for val (default: "
                        f"{VAL_PSEUDO_DONORS}).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_loss_curve(loss_components, out_dir, log):
    epochs = range(1, len(loss_components["total"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(epochs), loss_components["total"], label="total loss", color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 2 Training Loss — Sheu 2024 phase 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    log("  Saved: loss_curve.png")


def _print_learnability_summary(dynamics, log):
    from cytokine_mil.analysis.dynamics import rank_cytokines_by_learnability
    records = dynamics.get("records", [])
    if not records:
        return
    try:
        ranking = rank_cytokines_by_learnability(records)
        log("\nAll active classes (train, donor-aggregated p_correct AUC):")
        for entry in ranking["ranking"]:
            log(f"  • {entry['cytokine']:<15s}  AUC={entry['auc']:.4f}")
        log(f"\n{ranking['metric_description']}")
    except Exception as e:
        log(f"  (learnability summary skipped: {e})")


def _save_run_summary(dynamics, args, out_dir, log):
    train_recs = dynamics["records"]
    val_recs = dynamics.get("val_records", [])

    def _safe_agg(recs, agg):
        if not recs:
            return None
        vals = [agg(r["p_correct_trajectory"]) for r in recs if r.get("p_correct_trajectory")]
        return float(np.mean(vals)) if vals else None

    summary = {
        "seed":                  args.seed,
        "timestamp":             datetime.now().isoformat(),
        "final_train_p_correct": _safe_agg(train_recs, lambda x: x[-1]),
        "auc_train_p_correct":   _safe_agg(train_recs, np.mean),
        "final_val_p_correct":   _safe_agg(val_recs,   lambda x: x[-1]),
        "auc_val_p_correct":     _safe_agg(val_recs,   np.mean),
        "final_loss":            float(dynamics["loss_components"]["total"][-1])
                                 if dynamics.get("loss_components", {}).get("total") else None,
        "n_train_records":       len(train_recs),
        "n_val_records":         len(val_recs),
        "stage1_epochs":         args.stage1_epochs,
        "stage2_epochs":         args.stage2_epochs,
        "stage2_lr":             args.lr,
        "embed_dim":             args.embed_dim,
        "attention_hidden_dim":  args.attention_hidden_dim,
        "val_pseudo_donors":     args.val_donors,
        "active_classes":        sorted({e["cytokine"] for e in train_recs}
                                         | {e["cytokine"] for e in val_recs}),
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"  Saved: run_summary.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

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
    # Manifest + gene names
    # ------------------------------------------------------------------
    log("\nLoading manifest...")
    with open(args.manifest_path) as fh:
        manifest = json.load(fh)
    with open(args.hvg_path) as fh:
        gene_names = json.load(fh)
    log(f"  {len(manifest)} tubes, {len(gene_names)} genes (full Sheu panel — no HVG selection)")

    # ------------------------------------------------------------------
    # Label encoder (PBS pinned at 90; active classes occupy 0..N-1)
    # ------------------------------------------------------------------
    all_active = sorted({e["cytokine"] for e in manifest if e["cytokine"] != "PBS"})
    label_enc = CytokineLabel().fit(manifest)
    log(f"  {label_enc.n_classes()}-dim output (PBS at 90)")
    log(f"  Active stimuli: {all_active}")

    with open(out_dir / "label_encoder.json", "w") as fh:
        json.dump({"cytokines": list(label_enc.cytokines)}, fh)

    # ------------------------------------------------------------------
    # Stage 1 manifest (one tube per cytokine for encoder pre-training)
    # ------------------------------------------------------------------
    log("\nBuilding Stage 1 manifest...")
    stage1_manifest_path = out_dir / "manifest_stage1.json"
    stage1_manifest = build_stage1_manifest(
        manifest, save_path=str(stage1_manifest_path), donor_offset=0,
    )
    log(f"  {len(stage1_manifest)} stage-1 tubes (one per cytokine)")

    # ------------------------------------------------------------------
    # Train/val split (donor-level, using pseudo-donor names)
    # ------------------------------------------------------------------
    log("\nSplitting train/val by pseudo-donor...")
    train_manifest, val_manifest = split_manifest_by_donor(manifest, args.val_donors)
    train_m_path = out_dir / "manifest_train.json"
    val_m_path   = out_dir / "manifest_val.json"
    with open(train_m_path, "w") as fh:
        json.dump(train_manifest, fh)
    with open(val_m_path, "w") as fh:
        json.dump(val_manifest, fh)
    log(f"  Train: {len(train_manifest)} tubes | Val: {len(val_manifest)} tubes")
    log(f"  Val pseudo-donors: {args.val_donors}")
    log(f"  Train pseudo-donors: {sorted({e['donor'] for e in train_manifest})}")

    # ------------------------------------------------------------------
    # Stage 1: encoder pretraining (cell-type classification)
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
    log(f"  Encoder: input={len(gene_names)}, embed_dim={args.embed_dim}, "
        f"n_cell_types={n_cell_types}")
    log(f"  Stage 1: {args.stage1_epochs} epochs, lr={args.stage1_lr}")

    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=n_cell_types,
        embed_dim=args.embed_dim,
    )
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
    # Stage 2: AB-MIL training (frozen encoder)
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("STAGE 2: AB-MIL training (frozen encoder, "
        f"{label_enc.n_classes()}-dim output / {len(all_active) + 1} active classes)")
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
    log(f"  Stage 2: {args.stage2_epochs} epochs, lr={args.lr}, warmup={args.lr_warmup_epochs}")

    dynamics = train_mil(
        model,
        train_dataset,
        n_epochs=args.stage2_epochs,
        lr=args.lr,
        momentum=STAGE2_MOMENTUM,
        lr_warmup_epochs=args.lr_warmup_epochs,
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
        "records":                          dynamics["records"],
        "val_records":                      dynamics["val_records"],
        "logged_epochs":                    dynamics["logged_epochs"],
        "confusion_entropy_trajectory":     dynamics["confusion_entropy_trajectory"],
        "val_confusion_entropy_trajectory": dynamics["val_confusion_entropy_trajectory"],
        "loss_components":                  dynamics["loss_components"],
        "label_encoder_cytokines":          list(label_enc.cytokines),
        "seed":                             args.seed,
        "val_pseudo_donors":                args.val_donors,
        "active_classes":                   all_active + ["PBS"],
        "stage2_epochs":                    args.stage2_epochs,
        "stage2_lr":                        args.lr,
        "dataset":                          "Sheu2024",
    }
    with open(out_dir / "dynamics.pkl", "wb") as fh:
        pickle.dump(dynamics_payload, fh)
    log("  Saved: dynamics.pkl")
    log(f"  Records: {len(dynamics['records'])} train, {len(dynamics['val_records'])} val")

    _plot_loss_curve(dynamics["loss_components"], out_dir, log)
    _print_learnability_summary(dynamics, log)
    _save_run_summary(dynamics, args, out_dir, log)

    log("\nDone.")
    log_file.close()


if __name__ == "__main__":
    main()
