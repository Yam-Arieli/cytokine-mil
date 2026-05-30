"""
Train binary AB-MIL models for the 16 cytokines NOT covered by the existing
oesinghaus_binary runs — required to unlock the full 17 KNOWN_DIRECTIONAL +
2 PRE_REGISTERED ground-truth set for the Path A → Bridge → Path B pipeline.

Mirrors scripts/train_oesinghaus_binary.py: one shared Stage 1 encoder
(cell-type classification) across all 16 cytokines + PBS, then one binary
AB-MIL per cytokine (cytokine vs PBS, n_classes=2) with the shared frozen
encoder.

Differences vs train_oesinghaus_binary.py:
  * Cytokine pool is the 16 missing from the prior 17-cytokine SIMPLE +
    COMPLEX experiment (see reports/SESSION_SUMMARY_2026-05-30.md §Scope).
  * No SIMPLE/COMPLEX grouping, no pre-registered Mann-Whitney hypothesis
    — this run is purely for unlocking pipeline validation, not for the
    learnability hypothesis.
  * Default HPs match the "wide" variant used to produce the existing 8
    binary models that the pipeline currently uses (embed_dim=512,
    hidden_dims=(512,512), attention_hidden_dim=128), so all 24 binary
    models share one architecture and the IG probe loads them uniformly
    via state-dict shape inference.

Outputs (one run_<timestamp> dir per submission):
  encoder_shared_stage1.pt
  manifest_stage1_shared.json
  model_<cytokine>.pt          ← one per cytokine
  dynamics_<cytokine>.pkl      ← one per cytokine
  manifest_train_<cytokine>.json, manifest_val_<cytokine>.json
  label_encoder_<cytokine>.json
  learning_curves_binary_<seed>.png

Usage:
  python scripts/train_oesinghaus_binary_missing16.py
  python scripts/train_oesinghaus_binary_missing16.py --seed 123
  python scripts/train_oesinghaus_binary_missing16.py --output_dir /path
"""

import argparse
import copy
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.dynamics import aggregate_to_donor_level  # noqa: E402
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH      = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
OUTPUT_BASE   = Path(__file__).parent.parent / "results" / "oesinghaus_binary_missing16"
VAL_DONORS    = ["Donor2", "Donor3"]
CONTROL       = "PBS"

# 16 cytokines missing from the existing oesinghaus_binary trained pool —
# needed to evaluate the full 17 KNOWN_DIRECTIONAL + 2 PRE_REGISTERED axes.
# Identified via:
#   axes with literature_status in {KNOWN_DIRECTIONAL, PRE_REGISTERED}
#   minus cytokines already in {IFN-beta, IL-1-beta, TNF-alpha, IL-6, IL-2,
#   IL-10, IL-12, TGF-beta1}
MISSING_CYTOKINES = [
    "IFN-gamma",
    "IFN-omega",
    "IFN-lambda1",
    "IL-15",
    "IL-17A",
    "IL-36-alpha",
    "IL-9",
    "IL-13",
    "IL-27",
    "IL-16",
    "CD30L",
    "Decorin",
    "VEGF",
    "GM-CSF",
    "TL1A",
    "IL-35",
]

# Wide HPs — match the existing oesinghaus_binary run we built the pipeline on
EMBED_DIM            = 512
HIDDEN_DIMS          = (512, 512)
ATTENTION_HIDDEN_DIM = 128
STAGE1_EPOCHS        = 20
STAGE1_LR            = 0.005
STAGE1_MOMENTUM      = 0.9
STAGE2_EPOCHS        = 250
STAGE2_LR            = 0.00003   # same as the prior wide run
STAGE2_MOMENTUM      = 0.90
LOG_EVERY            = 1
SEED                 = 42


# ---------------------------------------------------------------------------
# Per-cytokine binary model training (mirrors train_oesinghaus_binary)
# ---------------------------------------------------------------------------

def _train_one_binary_model(
    target: str,
    control: str,
    manifest: list,
    gene_names: list,
    shared_encoder,
    out_dir: Path,
    device: torch.device,
    seed: int,
    log,
    embed_dim: int,
    hidden_dims: tuple,
    attention_hidden_dim: int,
) -> dict:
    """
    Train a single binary AB-MIL model (Stage 2 only, shared frozen encoder)
    for target vs control. Returns the dynamics dict with keys:
        records, val_records, logged_epochs, condition, control
    """
    log(f"\n{'=' * 50}")
    log(f"Training binary model: {target} vs {control}")
    log("=" * 50)

    # Binary manifest + donor split
    bin_manifest, label_enc = make_binary_manifest(manifest, target, control=control)
    train_m, val_m = split_manifest_by_donor(bin_manifest, VAL_DONORS)
    log(f"  {target}: {len(train_m)} train tubes, {len(val_m)} val tubes")

    safe_target = target.replace("/", "_")
    train_m_path = out_dir / f"manifest_train_{safe_target}.json"
    val_m_path   = out_dir / f"manifest_val_{safe_target}.json"

    with open(train_m_path, "w") as fh:
        json.dump(train_m, fh)
    with open(val_m_path, "w") as fh:
        json.dump(val_m, fh)

    with open(out_dir / f"label_encoder_{safe_target}.json", "w") as fh:
        json.dump(
            {"positive": label_enc.positive, "negative": label_enc.negative}, fh,
        )

    train_dataset = PseudoTubeDataset(
        str(train_m_path), label_enc, gene_names=gene_names, preload=True,
    )
    val_dataset = PseudoTubeDataset(
        str(val_m_path), label_enc, gene_names=gene_names, preload=True,
    )

    # Each binary model gets its own deep copy of the shared encoder
    encoder = copy.deepcopy(shared_encoder)
    log("  Using deep copy of shared Stage 1 encoder")

    model = build_mil_model(
        encoder,
        embed_dim=embed_dim,
        attention_hidden_dim=attention_hidden_dim,
        n_classes=label_enc.n_classes(),
        encoder_frozen=True,
    )

    dynamics = train_mil(
        model,
        train_dataset,
        n_epochs=STAGE2_EPOCHS,
        lr=STAGE2_LR,
        momentum=STAGE2_MOMENTUM,
        log_every_n_epochs=LOG_EVERY,
        device=device,
        seed=seed,
        verbose=True,
        val_dataset=val_dataset,
    )

    torch.save(model.state_dict(), out_dir / f"model_{safe_target}.pt")

    payload = {
        "records":       dynamics["records"],
        "val_records":   dynamics["val_records"],
        "logged_epochs": dynamics["logged_epochs"],
        "condition":     target,
        "control":       control,
    }
    pkl_path = out_dir / f"dynamics_{safe_target}.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh)

    log(f"  Saved: model_{safe_target}.pt, dynamics_{safe_target}.pkl")
    return payload


def _donor_mean_trajectory(records, condition):
    cond_records = [r for r in records if r["cytokine"] == condition]
    if not cond_records:
        return None
    donor_traj = aggregate_to_donor_level(cond_records, "p_correct_trajectory")
    if condition not in donor_traj or not donor_traj[condition]:
        return None
    return np.mean(list(donor_traj[condition].values()), axis=0)


def _plot_learning_curves(all_dynamics, out_dir, seed, log):
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(MISSING_CYTOKINES)))
    for color, target in zip(colors, MISSING_CYTOKINES):
        dyn = all_dynamics.get(target)
        if dyn is None:
            continue
        logged = dyn["logged_epochs"]
        train_mean = _donor_mean_trajectory(dyn["records"], target)
        val_mean = _donor_mean_trajectory(dyn["val_records"], target)
        if train_mean is not None:
            ax.plot(logged, train_mean, color=color, linestyle="-",
                    linewidth=1.8, label=target)
        if val_mean is not None:
            ax.plot(logged, val_mean, color=color, linestyle="--",
                    linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("P(Y_correct)")
    ax.set_title("Oesinghaus binary (missing-16 cytokines) — cytokine vs PBS")
    ax.set_ylim(0, 1.05)
    train_line = mlines.Line2D([], [], color="gray", linestyle="-",
                                linewidth=1.8, label="train (solid)")
    val_line = mlines.Line2D([], [], color="gray", linestyle="--",
                             linewidth=1.0, alpha=0.6, label="val (dashed)")
    legend1 = ax.legend(handles=[train_line, val_line], loc="lower right",
                        fontsize=9, title="Split")
    ax.add_artist(legend1)
    ax.legend(loc="upper right", fontsize=7, ncol=2, title="Cytokine")
    plt.tight_layout()

    fname = f"learning_curves_binary_missing16_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train binary AB-MIL for the 16 cytokines missing from "
                    "the prior oesinghaus_binary trained pool.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Training seed (default: {SEED})")
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--hidden_dims", type=int, nargs="+",
                        default=list(HIDDEN_DIMS))
    parser.add_argument("--attention_hidden_dim", type=int,
                        default=ATTENTION_HIDDEN_DIM)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else SEED
    embed_dim = args.embed_dim
    hidden_dims = tuple(args.hidden_dims)
    attention_hidden_dim = args.attention_hidden_dim

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = OUTPUT_BASE / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    # Dependency audit (fail fast)
    for required in ("numpy", "torch", "scanpy", "anndata"):
        try:
            __import__(required)
        except ImportError as e:
            print(f"FATAL: missing dependency '{required}': {e}", flush=True)
            sys.exit(2)

    log("=" * 60)
    log("Oesinghaus binary training — MISSING 16 cytokines")
    log("=" * 60)
    log(f"Cytokines ({len(MISSING_CYTOKINES)}): {MISSING_CYTOKINES}")
    log(f"Val donors (held out):                {VAL_DONORS}")
    log(f"Control:                              {CONTROL}")
    log(f"Output dir:                           {out_dir}")
    log(f"Started:                              {timestamp}")
    log(f"Training seed:                        {seed}")
    log("")
    log("Hyperparameters (wide variant, matches existing trained 8):")
    log(f"  embed_dim:            {embed_dim}")
    log(f"  hidden_dims:          {hidden_dims}")
    log(f"  attention_hidden_dim: {attention_hidden_dim}")
    log(f"  Stage 1 epochs:       {STAGE1_EPOCHS}  lr={STAGE1_LR}")
    log(f"  Stage 2 epochs:       {STAGE2_EPOCHS}  lr={STAGE2_LR}")
    log("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    log("\nLoading manifest and HVG list...")
    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    with open(HVG_PATH) as fh:
        gene_names = json.load(fh)
    log(f"Manifest entries: {len(manifest)}")
    log(f"HVGs:             {len(gene_names)}")

    manifest_cytokines = {e["cytokine"] for e in manifest}
    missing = [c for c in MISSING_CYTOKINES if c not in manifest_cytokines]
    if missing:
        for c in missing:
            log(f"FATAL: cytokine not found in manifest: {c}")
        raise ValueError(f"Missing cytokines: {missing}")

    filtered_manifest = filter_manifest(
        manifest, MISSING_CYTOKINES, include_pbs=True,
    )
    log(f"Filtered manifest size: {len(filtered_manifest)} entries "
        f"({len(MISSING_CYTOKINES)} cytokines + PBS)")

    # -----------------------------------------------------------------
    # Shared Stage 1 — encoder pre-training (cell-type classification)
    # -----------------------------------------------------------------
    log("\n" + "=" * 60)
    log("SHARED STAGE 1 — encoder pre-training (cell-type classification)")
    log("=" * 60)

    stage1_path = out_dir / "manifest_stage1_shared.json"
    stage1_manifest = build_stage1_manifest(
        filtered_manifest, save_path=str(stage1_path),
    )
    log(f"Stage 1 manifest: {len(stage1_manifest)} entries "
        f"saved to {stage1_path.name}")

    cell_dataset = CellDataset(
        str(stage1_path), gene_names=gene_names, preload=True,
    )
    cell_loader = DataLoader(
        cell_dataset, batch_size=256, shuffle=True, num_workers=0,
    )
    log(f"Stage 1 cells: {len(cell_dataset)}  |  "
        f"Cell types: {cell_dataset.n_cell_types()}")

    encoder = InstanceEncoder(
        input_dim=len(gene_names),
        embed_dim=embed_dim,
        n_cell_types=cell_dataset.n_cell_types(),
        hidden_dims=hidden_dims,
    )
    log(f"Stage 1: training shared encoder, "
        f"n_cell_types={cell_dataset.n_cell_types()}")

    train_encoder(
        encoder,
        cell_loader,
        n_epochs=STAGE1_EPOCHS,
        lr=STAGE1_LR,
        momentum=STAGE1_MOMENTUM,
        device=device,
    )

    torch.save(encoder.state_dict(), out_dir / "encoder_shared_stage1.pt")
    log("Shared encoder saved: encoder_shared_stage1.pt")

    # -----------------------------------------------------------------
    # Loop — one binary model per missing cytokine
    # -----------------------------------------------------------------
    all_dynamics = {}
    for i, target in enumerate(MISSING_CYTOKINES, start=1):
        log(f"\n[{i}/{len(MISSING_CYTOKINES)}] {target} START")
        try:
            all_dynamics[target] = _train_one_binary_model(
                target=target,
                control=CONTROL,
                manifest=filtered_manifest,
                gene_names=gene_names,
                shared_encoder=encoder,
                out_dir=out_dir,
                device=device,
                seed=seed,
                log=log,
                embed_dim=embed_dim,
                hidden_dims=hidden_dims,
                attention_hidden_dim=attention_hidden_dim,
            )
            log(f"[{i}/{len(MISSING_CYTOKINES)}] {target} DONE")
        except Exception as e:
            import traceback
            log(f"[{i}/{len(MISSING_CYTOKINES)}] {target} ERROR: "
                f"{type(e).__name__}: {e}")
            log(traceback.format_exc())
            # Defensive: don't kill the whole run; continue with remaining
            # cytokines so the bulk of the 12-hour GPU budget isn't wasted.
            continue

    # -----------------------------------------------------------------
    # Learning-curve plot
    # -----------------------------------------------------------------
    log("\n" + "=" * 60)
    log("Saving learning-curve plot...")
    log("=" * 60)
    _plot_learning_curves(all_dynamics, out_dir, seed, log)

    log("")
    log(f"Done. {len(all_dynamics)}/{len(MISSING_CYTOKINES)} cytokines trained.")
    log(f"Results in: {out_dir}")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
