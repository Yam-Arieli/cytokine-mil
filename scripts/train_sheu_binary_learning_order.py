"""
Train per-stimulus binary AB-MIL models on Sheu 3hr pseudo-tubes for the
gene learning-order experiment (LEARNING_ORDER_PREREGISTRATION.md).

One shared Stage 1 encoder (cell-type classification) is trained on the
Sheu pseudo-tubes, then a separate binary AB-MIL (stimulus vs PBS) is trained
for each stimulus in --stimuli (default: polyIC, LPS).

CRITICAL: every Stage-2 epoch is checkpointed so that
scripts/extract_gene_attribution_trajectory.py can loop through the full
training trajectory and compute per-gene attribution at each epoch.

Outputs (under --output_dir):
  encoder_shared_stage1.pt
  manifest_stage1_shared.json
  gene_names.json
  For each stimulus S:
    checkpoints_<S>/epoch_XXXX.pt
    manifest_train_<S>.json
    label_encoder_<S>.json

Usage:
  python scripts/train_sheu_binary_learning_order.py --seed 42
  python scripts/train_sheu_binary_learning_order.py \\
      --seed 123 --stimuli polyIC LPS --stage2_epochs 80
"""

from __future__ import annotations

import argparse
import copy
import json
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
    build_encoder,
    build_mil_model,
    build_stage1_manifest,
    make_binary_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder  # noqa: E402
from cytokine_mil.training.train_mil import train_mil  # noqa: E402


# ---------------------------------------------------------------------------
# Cluster defaults
# ---------------------------------------------------------------------------

MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json"
HVG_PATH      = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json"
OUTPUT_BASE   = REPO_ROOT / "results" / "gene_learning_order"

# Sheu val pseudo-donor per CLAUDE.md §2.5 / sheu2024.yaml
VAL_PSEUDO_DONORS = ["M2_IL4_rep1"]
CONTROL           = "PBS"

# Default stimuli for the pre-registered TRIF -> IFN cascade test
DEFAULT_STIMULI = ["PIC", "LPS"]   # Sheu manifest names poly(I:C) as "PIC"

# Hyperparameters matching sheu2024.yaml "narrowed" settings (small model,
# moderate LR so learning curves are smooth for trajectory analysis).
EMBED_DIM            = 32
ATTENTION_HIDDEN_DIM = 16
STAGE1_EPOCHS        = 30
STAGE1_LR            = 0.003
STAGE1_MOMENTUM      = 0.9
STAGE2_EPOCHS        = 60       # CLI default; override with --stage2_epochs
STAGE2_LR            = 0.0005
STAGE2_MOMENTUM      = 0.9
LR_WARMUP            = 5
LOG_EVERY            = 1
SEED                 = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def _train_one_binary(
    stimulus: str,
    manifest: list,
    gene_names: list,
    shared_encoder,
    out_dir: Path,
    device: torch.device,
    seed: int,
    stage2_epochs: int,
    log,
) -> None:
    """Train one binary AB-MIL (stimulus vs PBS) with per-epoch checkpoints."""
    log(f"\n{'=' * 54}")
    log(f"Binary model: {stimulus} vs {CONTROL}")
    log("=" * 54)

    safe = _safe_name(stimulus)

    # Build binary manifest + donor split
    bin_manifest, label_enc = make_binary_manifest(manifest, stimulus, control=CONTROL)
    train_m, _val_m = split_manifest_by_donor(bin_manifest, VAL_PSEUDO_DONORS)

    log(f"  Train tubes: {len(train_m)}   (val not used for attribution extraction)")

    # Write train manifest and label encoder
    train_m_path = out_dir / f"manifest_train_{safe}.json"
    with open(train_m_path, "w") as fh:
        json.dump(train_m, fh)

    with open(out_dir / f"label_encoder_{safe}.json", "w") as fh:
        json.dump({"positive": stimulus, "negative": CONTROL}, fh)

    train_dataset = PseudoTubeDataset(
        str(train_m_path), label_enc, gene_names=gene_names, preload=True,
    )
    log(f"  Train dataset: {len(train_dataset)} tubes loaded")

    # Deep-copy encoder so each stimulus model trains independently
    encoder = copy.deepcopy(shared_encoder)
    model = build_mil_model(
        encoder,
        embed_dim=EMBED_DIM,
        attention_hidden_dim=ATTENTION_HIDDEN_DIM,
        n_classes=label_enc.n_classes(),
        encoder_frozen=True,
    )
    log(f"  Model: embed_dim={EMBED_DIM}, attn_hidden={ATTENTION_HIDDEN_DIM}, "
        f"n_classes={label_enc.n_classes()}")

    ckpt_subdir = out_dir / f"checkpoints_{safe}"
    all_epochs = list(range(1, stage2_epochs + 1))

    log(f"  Stage 2: {stage2_epochs} epochs  |  checkpoints -> {ckpt_subdir.name}/")

    train_mil(
        model,
        train_dataset,
        n_epochs=stage2_epochs,
        lr=STAGE2_LR,
        momentum=STAGE2_MOMENTUM,
        lr_warmup_epochs=LR_WARMUP,
        log_every_n_epochs=LOG_EVERY,
        device=device,
        seed=seed,
        verbose=True,
        val_dataset=None,
        # CRITICAL: checkpoint every epoch so the attribution extractor can
        # loop epoch_XXXX.pt -> per-gene gradient at each training step.
        checkpoint_dir=str(ckpt_subdir),
        checkpoint_epochs=all_epochs,
    )

    log(f"  Done. Checkpoints saved: {len(all_epochs)} epochs under {ckpt_subdir.name}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train Sheu binary AB-MIL for gene learning-order experiment.",
    )
    p.add_argument("--seed", type=int, default=SEED,
                   help=f"Training seed (default: {SEED})")
    p.add_argument("--stage2_epochs", type=int, default=STAGE2_EPOCHS,
                   help=f"Stage 2 epochs (default: {STAGE2_EPOCHS})")
    p.add_argument("--stimuli", nargs="+", default=DEFAULT_STIMULI,
                   help=f"Stimuli to train binary models for "
                        f"(default: {DEFAULT_STIMULI})")
    p.add_argument("--manifest", type=str, default=MANIFEST_PATH,
                   help="Path to Sheu pseudo-tube manifest.json")
    p.add_argument("--hvg", type=str, default=HVG_PATH,
                   help="Path to Sheu hvg_list.json (500 genes)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: results/gene_learning_order/seed_<seed>)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    seed = args.seed

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = OUTPUT_BASE / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "train.log"
    log_file = open(log_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        print(msg, file=log_file, flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log("=" * 60)
    log("Sheu binary training — gene learning-order experiment")
    log("=" * 60)
    log(f"  Stimuli:         {args.stimuli}")
    log(f"  Seed:            {seed}")
    log(f"  Stage2 epochs:   {args.stage2_epochs}")
    log(f"  embed_dim:       {EMBED_DIM}")
    log(f"  attn_hidden:     {ATTENTION_HIDDEN_DIM}")
    log(f"  Stage1 epochs:   {STAGE1_EPOCHS}  lr={STAGE1_LR}")
    log(f"  Stage2 lr:       {STAGE2_LR}  warmup={LR_WARMUP}")
    log(f"  Val donors:      {VAL_PSEUDO_DONORS}")
    log(f"  Output dir:      {out_dir}")
    log(f"  Started:         {timestamp}")
    log("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Load manifest + gene names
    # ------------------------------------------------------------------
    log("\nLoading manifest and gene list...")
    with open(args.manifest) as fh:
        manifest = json.load(fh)
    with open(args.hvg) as fh:
        gene_names = json.load(fh)
    log(f"  Manifest entries: {len(manifest)}")
    log(f"  Genes (full Sheu panel): {len(gene_names)}")

    # Validate stimuli are present
    manifest_cyts = {e["cytokine"] for e in manifest}
    missing = [s for s in args.stimuli if s not in manifest_cyts]
    if missing:
        log(f"FATAL: stimuli not found in manifest: {missing}")
        log(f"  Available: {sorted(manifest_cyts)}")
        sys.exit(2)

    # Save gene names so the extractor can align to the same list
    with open(out_dir / "gene_names.json", "w") as fh:
        json.dump(gene_names, fh)

    # ------------------------------------------------------------------
    # Shared Stage 1 — encoder pretraining (cell-type classification)
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("SHARED STAGE 1 — encoder pretraining (cell-type classification)")
    log("=" * 60)

    # Filter manifest to the stimuli of interest + PBS for Stage 1 coverage
    stim_set = set(args.stimuli) | {"PBS"}
    filtered_manifest = [e for e in manifest if e["cytokine"] in stim_set]
    log(f"  Filtered manifest for Stage 1: {len(filtered_manifest)} entries")

    stage1_path = out_dir / "manifest_stage1_shared.json"
    stage1_manifest = build_stage1_manifest(filtered_manifest, save_path=str(stage1_path))
    log(f"  Stage 1 manifest: {len(stage1_manifest)} entries -> {stage1_path.name}")

    cell_dataset = CellDataset(str(stage1_path), gene_names=gene_names, preload=True)
    cell_loader = DataLoader(cell_dataset, batch_size=256, shuffle=True, num_workers=0)
    n_cell_types = len(cell_dataset.cell_type_to_idx)
    log(f"  Encoder: input={len(gene_names)}, embed_dim={EMBED_DIM}, "
        f"n_cell_types={n_cell_types}")

    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=n_cell_types,
        embed_dim=EMBED_DIM,
    )
    train_encoder(
        encoder,
        cell_loader,
        n_epochs=STAGE1_EPOCHS,
        lr=STAGE1_LR,
        momentum=STAGE1_MOMENTUM,
        device=device,
        verbose=True,
    )
    torch.save(encoder.state_dict(), out_dir / "encoder_shared_stage1.pt")
    log("  Saved: encoder_shared_stage1.pt")

    # ------------------------------------------------------------------
    # Stage 2 — one binary model per stimulus, all epochs checkpointed
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("STAGE 2 — per-stimulus binary AB-MIL (frozen encoder, all epochs checkpointed)")
    log("=" * 60)

    for i, stimulus in enumerate(args.stimuli, start=1):
        log(f"\n[{i}/{len(args.stimuli)}] {stimulus} START")
        try:
            _train_one_binary(
                stimulus=stimulus,
                manifest=manifest,
                gene_names=gene_names,
                shared_encoder=encoder,
                out_dir=out_dir,
                device=device,
                seed=seed,
                stage2_epochs=args.stage2_epochs,
                log=log,
            )
            log(f"[{i}/{len(args.stimuli)}] {stimulus} DONE")
        except Exception as exc:
            import traceback
            log(f"[{i}/{len(args.stimuli)}] {stimulus} ERROR: "
                f"{type(exc).__name__}: {exc}")
            log(traceback.format_exc())
            # Continue so the remaining stimulus is still trained.
            continue

    log("\n" + "=" * 60)
    log(f"Done. {len(args.stimuli)} stimulus models trained.")
    log(f"Results in: {out_dir}")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_file.close()


if __name__ == "__main__":
    main()
