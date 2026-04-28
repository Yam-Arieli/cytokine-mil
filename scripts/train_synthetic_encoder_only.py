"""
Train Stage 1 encoder only (cell-type classification) on the synthetic cascade dataset.

This is the exact same procedure used for the real Oesinghaus robustness experiments:
  - Build Stage 1 manifest (one tube per cytokine, rotating donors)
  - Train InstanceEncoder (cell-type classification, 50 epochs)
  - Wrap encoder in CytokineABMIL and save as model_stage2.pt
    (the geo script loads model_stage2.pt; the MIL head is unused — only H embeddings matter)

Output directory: results/synthetic_cascades/seed{SEED}/
  encoder_stage1.pt      <- encoder weights
  model_stage2.pt        <- full CytokineABMIL wrapper (geo script expects this name)
  label_encoder.json     <- cytokine index mapping (padded to length 91)
  manifest_train.json    <- all tubes (used by geo script for embedding extraction)
  manifest_stage1.json   <- one tube per cytokine (used for Stage 1 training)
  train.log

Usage:
    python scripts/train_synthetic_encoder_only.py --seed 42
    python scripts/train_synthetic_encoder_only.py --seed 123 --stage1_epochs 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import CellDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import (
    build_encoder,
    build_mil_model,
    build_stage1_manifest,
)
from cytokine_mil.training.train_encoder import train_encoder


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SYNTHETIC_DIR = Path("/cs/labs/mornitzan/yam.arieli/datasets/synthetic_cascades_v2")
MANIFEST_PATH = str(SYNTHETIC_DIR / "manifest.json")
HVG_PATH      = str(SYNTHETIC_DIR / "hvg_list.json")
OUTPUT_BASE   = REPO_ROOT / "results" / "synthetic_cascades"

EMBED_DIM            = 128
STAGE1_EPOCHS        = 50
STAGE1_LR            = 0.01
STAGE1_MOMENTUM      = 0.9
SEED                 = 42


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train Stage 1 encoder on synthetic cascade data.")
    p.add_argument("--seed",           type=int,   default=SEED)
    p.add_argument("--output_dir",     type=str,   default=None,
                   help="Override output directory (default: results/synthetic_cascades/seed{N})")
    p.add_argument("--stage1_epochs",  type=int,   default=STAGE1_EPOCHS)
    p.add_argument("--lr",             type=float, default=STAGE1_LR)
    p.add_argument("--embed_dim",      type=int,   default=EMBED_DIM)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_BASE / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(out_dir / "train.log", "w")
    def log(msg=""):
        print(msg, flush=True)
        print(msg, file=log_file, flush=True)

    log("=" * 62)
    log("Synthetic cascade — Stage 1 encoder training")
    log(f"  seed          : {args.seed}")
    log(f"  stage1_epochs : {args.stage1_epochs}")
    log(f"  lr            : {args.lr}")
    log(f"  embed_dim     : {args.embed_dim}")
    log(f"  out_dir       : {out_dir}")
    log("=" * 62)

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
    log(f"  {len(manifest)} tubes  |  {len(gene_names)} genes")

    # ------------------------------------------------------------------
    # Label encoder  (padded to 91 so PBS = index 90 as in geo script)
    # ------------------------------------------------------------------
    label_enc = CytokineLabel().fit(manifest)
    n_cls = label_enc.n_classes()   # 91
    cytokines_ordered = [""] * n_cls
    for idx, name in label_enc._idx_to_label.items():
        cytokines_ordered[int(idx)] = name
    with open(out_dir / "label_encoder.json", "w") as fh:
        json.dump({"cytokines": cytokines_ordered}, fh)
    log(f"  {n_cls} classes (including PBS at index 90)  — label_encoder.json saved")

    # ------------------------------------------------------------------
    # Stage 1 manifest (one tube per cytokine, rotating donors)
    # ------------------------------------------------------------------
    log("\nBuilding Stage 1 manifest...")
    stage1_manifest_path = out_dir / "manifest_stage1.json"
    stage1_manifest = build_stage1_manifest(manifest, save_path=str(stage1_manifest_path))
    log(f"  {len(stage1_manifest)} tubes (one per cytokine)")

    # ------------------------------------------------------------------
    # Full manifest saved as manifest_train.json
    # (geo script loads this to embed all tubes for centroid computation)
    # ------------------------------------------------------------------
    train_manifest_path = out_dir / "manifest_train.json"
    with open(train_manifest_path, "w") as fh:
        json.dump(manifest, fh)
    log(f"  manifest_train.json saved ({len(manifest)} tubes — all donors)")

    # ------------------------------------------------------------------
    # Stage 1: encoder pre-training (cell-type classification)
    # ------------------------------------------------------------------
    log("\n" + "=" * 62)
    log("STAGE 1: Encoder pre-training (cell-type classification)")
    log("=" * 62)

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

    encoder = train_encoder(
        encoder,
        cell_loader,
        n_epochs=args.stage1_epochs,
        lr=args.lr,
        momentum=STAGE1_MOMENTUM,
        device=device,
    )
    torch.save(encoder.state_dict(), out_dir / "encoder_stage1.pt")
    log(f"  encoder_stage1.pt saved")

    # ------------------------------------------------------------------
    # Wrap encoder in CytokineABMIL and save as model_stage2.pt
    # The geo script (run_experiment_geo.py) loads model_stage2.pt and uses
    # only the encoder's H embeddings — the MIL attention/classifier head
    # is never used for PBS-RC geometry, so random weights are fine here.
    # ------------------------------------------------------------------
    model = build_mil_model(
        encoder,
        embed_dim=args.embed_dim,
        attention_hidden_dim=64,
        n_classes=n_cls,
        encoder_frozen=False,
    )
    torch.save(model.state_dict(), out_dir / "model_stage2.pt")
    log(f"  model_stage2.pt saved (encoder wrapped in CytokineABMIL for geo script)")

    log(f"\nAll outputs saved to {out_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
