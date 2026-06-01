"""
Immune Dictionary (Cui Nature 2024) — Stage 1 (encoder pretraining) + Stage 2 (AB-MIL).

Multiclass AB-MIL across ~87 ID classes (86 cytokines + PBS).  Path A latent
geometry (run_latent_geometry.py) consumes the trained encoder + dynamics to
discover cytokine coupling axes (informative secondary check; Path B cross_asym
is the primary direction metric per CLAUDE.md §26).

Clone of train_sheu2024_stage12.py with:
  - ID manifest/HVG paths.
  - Wider Oes-style HPs (4000 HVG / ~87 classes; production train_oesinghaus_full
    is the scale reference: embed=128, attn=64, Stage1 50@0.01, Stage2 20@0.001).
  - Slim label encoder (87 active classes, PBS at idx 0) to drop padded logits.
  - No Sheu axis pre-registration block (Path A on ID is exploratory secondary
    output; Path B is the primary).
  - Val mouse auto-detected from <manifest_dir>/build_metadata.json (the adapter
    picked the outlier-PBS-PCA mouse).  Overridable via --val_donors.

Usage:
    python scripts/train_immune_dictionary_stage12.py --seed 42
    python scripts/train_immune_dictionary_stage12.py --seed 123 --output_dir ...
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset  # noqa: E402
from cytokine_mil.data.label_encoder import CytokineLabel  # noqa: E402
from cytokine_mil.experiment_setup import (  # noqa: E402
    build_encoder,
    build_mil_model,
    build_stage1_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder  # noqa: E402
from cytokine_mil.training.train_mil import train_mil  # noqa: E402


# ---------------------------------------------------------------------------
# Paths & Constants (cluster defaults; override via CLI)
# ---------------------------------------------------------------------------

MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/manifest.json"
HVG_PATH      = "/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/hvg_list.json"
OUTPUT_BASE   = REPO_ROOT / "results" / "id_cascade"

# Default val mouse fallback (overridden by build_metadata.json if present).
DEFAULT_VAL_DONORS = ["mouse_3"]

# HPs matching the production Oesinghaus full multiclass (4000 HVG, ~91 classes).
# Wider than Sheu (32/16) but not the binary-wide bridge config (512/128).
EMBED_DIM            = 128
ATTENTION_HIDDEN_DIM = 64
STAGE1_EPOCHS        = 50
STAGE1_LR            = 0.01
STAGE1_MOMENTUM      = 0.9
STAGE2_EPOCHS        = 20
STAGE2_LR            = 0.001
STAGE2_MOMENTUM      = 0.9
LR_WARMUP            = 5
LOG_EVERY            = 1
SEED                 = 42
SLIM_LABEL_ENCODER   = True


# ---------------------------------------------------------------------------
# Slim label encoder (clone of Sheu's, dataset-agnostic)
# ---------------------------------------------------------------------------

class _SlimIDLabel:
    """N-class label encoder where N == # distinct cytokines in the manifest.

    PBS is mapped to index 0; other cytokines sorted alphabetically and assigned
    1..N-1.  Carries the same surface as CytokineLabel (.encode / .decode /
    .n_classes() / .cytokines) so PseudoTubeDataset, build_mil_model, etc. work
    unchanged.  Used to drop the 4 unused logits (91 - 87) in the default 91-d
    head.
    """

    def __init__(self, cytokines):
        non_pbs = sorted(c for c in cytokines if c != "PBS")
        names = ["PBS"] + non_pbs
        self._label_to_idx = {c: i for i, c in enumerate(names)}
        self._idx_to_label = {i: c for i, c in enumerate(names)}

    @classmethod
    def fit(cls, manifest):
        return cls({e["cytokine"] for e in manifest})

    def encode(self, c): return self._label_to_idx[c]
    def decode(self, i): return self._idx_to_label[i]
    def n_classes(self): return len(self._label_to_idx)

    @property
    def cytokines(self):
        return [self._idx_to_label[i] for i in sorted(self._idx_to_label)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_val_mouse_from_metadata(manifest_path: str):
    """If <manifest_dir>/build_metadata.json has a 'val_mouse' key, return it.

    The ID adapter (build_pseudotubes_immune_dictionary.py) picks the outlier-
    PBS-PCA mouse at build time and writes it here.  Returns None if the file is
    missing or doesn't contain the key.
    """
    meta_path = Path(manifest_path).parent / "build_metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
        return meta.get("val_mouse")
    except (json.JSONDecodeError, OSError):
        return None


def _plot_loss_curve(loss_components, out_dir, log):
    epochs = range(1, len(loss_components["total"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(epochs), loss_components["total"], label="total loss",
            color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 2 Training Loss — Immune Dictionary (Cui 2024)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    log("  Saved: loss_curve.png")


def _save_run_summary(dynamics, args, val_donors, out_dir, log):
    train_recs = dynamics["records"]
    val_recs = dynamics.get("val_records", [])

    def _safe_agg(recs, agg):
        if not recs:
            return None
        vals = [agg(r["p_correct_trajectory"]) for r in recs
                if r.get("p_correct_trajectory")]
        return float(np.mean(vals)) if vals else None

    summary = {
        "seed":                  args.seed,
        "timestamp":             datetime.now().isoformat(),
        "final_train_p_correct": _safe_agg(train_recs, lambda x: x[-1]),
        "auc_train_p_correct":   _safe_agg(train_recs, np.mean),
        "final_val_p_correct":   _safe_agg(val_recs,   lambda x: x[-1]),
        "auc_val_p_correct":     _safe_agg(val_recs,   np.mean),
        "final_loss":            float(dynamics["loss_components"]["total"][-1])
                                 if dynamics.get("loss_components", {}).get("total")
                                 else None,
        "n_train_records":       len(train_recs),
        "n_val_records":         len(val_recs),
        "stage1_epochs":         args.stage1_epochs,
        "stage2_epochs":         args.stage2_epochs,
        "stage2_lr":             args.lr,
        "embed_dim":             args.embed_dim,
        "attention_hidden_dim":  args.attention_hidden_dim,
        "val_donors":            val_donors,
        "active_classes":        sorted({e["cytokine"] for e in train_recs}
                                         | {e["cytokine"] for e in val_recs}),
        "dataset":               "ImmuneDictionary_Cui2024",
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log("  Saved: run_summary.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Immune Dictionary multiclass AB-MIL Stage 1 + 2 (Path A)."
    )
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
    p.add_argument("--val_donors", nargs="*", default=None,
                   help="Mouse names to hold out for val.  If unset, read from "
                        "<manifest_dir>/build_metadata.json:val_mouse and fall "
                        f"back to {DEFAULT_VAL_DONORS}.")
    p.add_argument("--slim_label_encoder", action="store_true",
                   default=SLIM_LABEL_ENCODER,
                   help="Use slim N-class encoder (PBS at 0).  Default True for ID.")
    p.add_argument("--no_slim_label_encoder", dest="slim_label_encoder",
                   action="store_false",
                   help="Force the legacy 91-d encoder (PBS at 90).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_BASE / "stage12" / f"run_{ts}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(out_dir / "train.log", "w")
    def log(msg=""):
        print(msg)
        print(msg, file=log_file, flush=True)

    log(f"Output directory: {out_dir}")
    log(f"Seed: {args.seed}")
    log("Dataset: Immune Dictionary (Cui Nature 2024) — multiclass Path A")
    log("Primary metric is Path B cross_asym (§26); Path A axes are secondary.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Resolve val mouse
    # ------------------------------------------------------------------
    if args.val_donors is None:
        from_meta = _read_val_mouse_from_metadata(args.manifest_path)
        if from_meta is not None:
            val_donors = [from_meta]
            log(f"Val mouse (from build_metadata.json): {val_donors}")
        else:
            val_donors = list(DEFAULT_VAL_DONORS)
            log(f"Val mouse (DEFAULT fallback — no build_metadata.json): "
                f"{val_donors}")
    else:
        val_donors = list(args.val_donors)
        log(f"Val mouse (from --val_donors): {val_donors}")

    # ------------------------------------------------------------------
    # Manifest + gene names
    # ------------------------------------------------------------------
    log("\nLoading manifest...")
    with open(args.manifest_path) as fh:
        manifest = json.load(fh)
    with open(args.hvg_path) as fh:
        gene_names = json.load(fh)
    log(f"  {len(manifest)} tubes, {len(gene_names)} genes (HVG-filtered)")

    # ------------------------------------------------------------------
    # Label encoder
    # ------------------------------------------------------------------
    all_active = sorted({e["cytokine"] for e in manifest if e["cytokine"] != "PBS"})
    if args.slim_label_encoder:
        label_enc = _SlimIDLabel.fit(manifest)
        log(f"  Slim label encoder: {label_enc.n_classes()}-dim output "
            f"(PBS at idx 0, alphabetical sort thereafter)")
    else:
        label_enc = CytokineLabel().fit(manifest)
        log(f"  Legacy label encoder: {label_enc.n_classes()}-dim output "
            f"(PBS at 90)")
    log(f"  {len(all_active)} active cytokines + PBS = "
        f"{len(all_active) + 1} total classes")

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
    # Train/val split (donor-level, by mouse_id)
    # ------------------------------------------------------------------
    log("\nSplitting train/val by mouse...")
    train_manifest, val_manifest = split_manifest_by_donor(manifest, val_donors)
    train_m_path = out_dir / "manifest_train.json"
    val_m_path   = out_dir / "manifest_val.json"
    with open(train_m_path, "w") as fh:
        json.dump(train_manifest, fh)
    with open(val_m_path, "w") as fh:
        json.dump(val_manifest, fh)
    log(f"  Train: {len(train_manifest)} tubes | Val: {len(val_manifest)} tubes")
    log(f"  Val mice:   {val_donors}")
    log(f"  Train mice: {sorted({e['donor'] for e in train_manifest})}")

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
    # Stage 2: AB-MIL (frozen encoder)
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
    log(f"  Model: embed_dim={args.embed_dim}, "
        f"attn_hidden={args.attention_hidden_dim}, "
        f"n_classes={label_enc.n_classes()}")
    log(f"  Stage 2: {args.stage2_epochs} epochs, lr={args.lr}, "
        f"warmup={args.lr_warmup_epochs}")

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
        "val_donors":                       val_donors,
        "active_classes":                   all_active + ["PBS"],
        "stage2_epochs":                    args.stage2_epochs,
        "stage2_lr":                        args.lr,
        "dataset":                          "ImmuneDictionary_Cui2024",
    }
    with open(out_dir / "dynamics.pkl", "wb") as fh:
        pickle.dump(dynamics_payload, fh)
    log("  Saved: dynamics.pkl")
    log(f"  Records: {len(dynamics['records'])} train, "
        f"{len(dynamics['val_records'])} val")

    _plot_loss_curve(dynamics["loss_components"], out_dir, log)
    _save_run_summary(dynamics, args, val_donors, out_dir, log)

    log("\nDone.")
    log_file.close()


if __name__ == "__main__":
    main()
