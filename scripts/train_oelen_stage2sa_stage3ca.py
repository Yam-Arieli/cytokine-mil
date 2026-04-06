"""
Oelen 2022 pathogen experiment — cluster job script.

Runs the complete pipeline from scratch on the Oelen (1M-scBloodNL, v3)
pseudo-tubes:
  Stage 1 — Encoder pre-training (cell-type classification)
  Stage 2 — AB-MIL with single attention layer (SA)
  Stage 3 — CA-only: adds a cross-attention layer, trains only that

4 classes (all used — no bootstrap):
    UT     — unstimulated control
    24hCA  — C. albicans (intermediate cascade)
    24hMTB — M. tuberculosis (hardest: most cascade-dependent)
    24hPA  — P. aeruginosa (easiest: direct TLR4/TLR5 activation)

Expected learnability ordering (ground truth for validation):
    24hPA (easiest) > 24hCA (intermediate) > 24hMTB (hardest)
    UT should be readily distinguishable as unstimulated baseline.

Val split: donor_6.0 and donor_95.0 held out (observer-only, no gradient updates).
Train/val comparison guards against memorization of donor-specific patterns.

Results saved to:
    results/oelen_sa_ca/  (relative to repo root)

Usage:
    python scripts/train_oelen_stage2sa_stage3ca.py
    python scripts/train_oelen_stage2sa_stage3ca.py --seed 123
"""

import argparse
import json
import os
import pickle
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for cluster
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.data.dataset import PseudoTubeDataset, CellDataset
from cytokine_mil.models.stage3_ca_model import Stage3CAModel
from cytokine_mil.training.train_encoder import train_encoder
from cytokine_mil.training.train_mil import train_mil
from cytokine_mil.experiment_setup import (
    build_stage1_manifest,
    build_encoder,
    build_mil_model,
    split_manifest_by_donor,
)
from cytokine_mil.training.trainer import (
    build_cytokine_queues,
    generate_epoch_megabatches,
)
from cytokine_mil.analysis.dynamics import (
    aggregate_to_donor_level,
    rank_cytokines_by_learnability,
    compute_confusion_entropy_summary,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_PATH = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oelen_pseudotubes/manifest.json"
)
OUTPUT_DIR = str(Path(__file__).parent.parent / "results" / "oelen_sa_ca")

VAL_DONORS = ["donor_6.0", "donor_95.0"]

# Stage 3 epoch override (independent of config)
STAGE3_EPOCHS = 100

# Expected cascade depth ordering (SIMPLE = short cascade, COMPLEX = deep cascade)
# Note: this is signal DEPTH, not classification difficulty — orthogonal axes.
DIFFICULTY_MAP = {
    "24hPA":  "SIMPLE",    # short cascade: direct TLR4/TLR5 activation, primary responders dominate
    "24hCA":  "MED",       # intermediate cascade depth
    "24hMTB": "COMPLEX",   # deep cascade: sustained multi-tier signaling across cell types
    "UT":     "CONTROL",   # unstimulated baseline
}

# Display order for plots (simplest → most complex cascade)
ORDERED_CONDITIONS = ["24hPA", "24hCA", "24hMTB", "UT"]


# ---------------------------------------------------------------------------
# Local dynamics helpers (mirrors train_mil.py internals for Stage 3)
# ---------------------------------------------------------------------------

def _compute_entropy(a: torch.Tensor) -> float:
    """Shannon entropy of attention weights (nats). Clipped for stability."""
    a_safe = a.clamp(min=1e-10)
    return float(-(a_safe * a_safe.log()).sum())


def _init_tube_trajectories(entries):
    """Create an empty trajectory dict keyed by dataset index."""
    return {
        i: {
            "p_correct": [],
            "entropy": [],
            "entropy_ca": [],
            "instance_confidence_epochs": [],
            "instance_confidence_sa_epochs": [],
            "instance_confidence_ca_epochs": [],
            "softmax_epochs": [],
        }
        for i in range(len(entries))
    }


def _compute_confusion_entropy_snapshot(entries, tube_trajectories, label_encoder,
                                        cytokine_confusion_epochs):
    """
    Compute one confusion-entropy snapshot per condition from the latest
    softmax snapshot.

    Steps per condition C:
      1. Collect latest softmax ŷ_b in R^K for each tube b of C.
      2. Average across tubes: ȳ_C in R^K.
      3. Remove the true class k=C and renormalize.
      4. H_confusion(C) = -sum_{k!=C} q_k log(q_k).
    """
    cytokine_to_indices = defaultdict(list)
    for idx, entry in enumerate(entries):
        cytokine_to_indices[entry["cytokine"]].append(idx)

    for cytokine, indices in cytokine_to_indices.items():
        true_label = label_encoder.encode(cytokine)
        softmaxes = np.stack(
            [tube_trajectories[idx]["softmax_epochs"][-1] for idx in indices]
        )  # (n_tubes, K)
        mean_softmax = softmaxes.mean(axis=0)  # (K,)

        off_diag = np.concatenate(
            [mean_softmax[:true_label], mean_softmax[true_label + 1:]]
        )
        off_sum = float(off_diag.sum())
        if off_sum < 1e-10:
            cytokine_confusion_epochs[cytokine].append(0.0)
            continue
        q = off_diag / off_sum
        q_safe = np.clip(q, 1e-10, None)
        entropy = -float((q_safe * np.log(q_safe)).sum())
        cytokine_confusion_epochs[cytokine].append(entropy)


@torch.no_grad()
def _log_dynamics_stage3(model, dataset, entries, tube_trajectories,
                         cytokine_confusion_epochs, label_encoder, device):
    """
    Evaluate all tubes and append one snapshot to each tube's trajectory.

    Records p_correct, SA entropy, CA entropy, SA confidence, CA confidence,
    and softmax for confusion entropy computation. Runs in eval mode, no_grad.
    """
    model.eval()

    for idx, entry in enumerate(entries):
        X, label, _donor, _cyt_name = dataset[idx]
        X = X.to(device)

        y_hat, a_SA, a_CA, _H = model(X)
        probs = F.softmax(y_hat, dim=0)
        p_correct = probs[label].item()

        entropy_sa = _compute_entropy(a_SA)
        entropy_ca = _compute_entropy(a_CA)
        instance_conf_sa = (a_SA * p_correct).cpu().numpy()
        instance_conf_ca = (a_CA * p_correct).cpu().numpy()

        traj = tube_trajectories[idx]
        traj["p_correct"].append(p_correct)
        traj["entropy"].append(entropy_sa)
        traj["entropy_ca"].append(entropy_ca)
        traj["instance_confidence_epochs"].append(instance_conf_sa)
        traj["instance_confidence_sa_epochs"].append(instance_conf_sa)
        traj["instance_confidence_ca_epochs"].append(instance_conf_ca)
        traj["softmax_epochs"].append(probs.cpu().numpy())

    _compute_confusion_entropy_snapshot(
        entries, tube_trajectories, label_encoder, cytokine_confusion_epochs
    )


def _build_records(entries, tube_trajectories):
    """Combine manifest metadata with trajectory data into flat records."""
    records = []
    for idx, entry in enumerate(entries):
        traj = tube_trajectories[idx]
        ic_epochs = traj["instance_confidence_epochs"]
        ic_trajectory = np.stack(ic_epochs, axis=0).T if ic_epochs else None
        ic_sa_epochs = traj.get("instance_confidence_sa_epochs", [])
        ic_ca_epochs = traj.get("instance_confidence_ca_epochs", [])
        ic_sa = np.stack(ic_sa_epochs, axis=0).T if ic_sa_epochs else None
        ic_ca = np.stack(ic_ca_epochs, axis=0).T if ic_ca_epochs else None
        records.append({
            "cytokine": entry["cytokine"],
            "donor": entry["donor"],
            "tube_idx": entry["tube_idx"],
            "tube_path": entry["path"],
            "n_cells": entry["n_cells"],
            "p_correct_trajectory": traj["p_correct"],
            "entropy_trajectory": traj["entropy"],
            "entropy_trajectory_ca": traj["entropy_ca"] if traj["entropy_ca"] else None,
            "instance_confidence_trajectory": ic_trajectory,
            "confidence_trajectory_sa": ic_sa,
            "confidence_trajectory_ca": ic_ca,
        })
    return records


def _train_epoch_stage3(model, train_tube_dataset, queues, optimizer, criterion,
                        device, rng):
    """
    Run one epoch of mega-batch training, only updating CA parameters.

    Accumulates cross-entropy over the mega-batch, steps once.
    Returns mean loss over megabatches.
    """
    model.train()
    megabatches = generate_epoch_megabatches(queues, rng)
    total_loss = 0.0
    n_mb = max(len(megabatches), 1)

    for mb_indices in megabatches:
        optimizer.zero_grad()
        mb_loss = torch.tensor(0.0, device=device)
        n = len(mb_indices)

        for _cyt_idx, ds_idx in mb_indices.items():
            X, label, _donor, _cyt_name = train_tube_dataset[ds_idx]
            X = X.to(device)
            label_t = torch.tensor([label], dtype=torch.long, device=device)
            y_hat, _a_SA, _a_CA, _H = model(X)
            loss = criterion(y_hat.unsqueeze(0), label_t) / n
            mb_loss = mb_loss + loss

        mb_loss.backward()
        optimizer.step()
        total_loss += mb_loss.item()

    return total_loss / n_mb


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _extract_layer_entropy(records, cytokine):
    """Return SA and CA entropy curves (one array per tube) for a condition."""
    sa_curves, ca_curves = [], []
    for rec in records:
        if rec["cytokine"] != cytokine:
            continue
        sa_traj = rec.get("entropy_trajectory")
        ca_traj = rec.get("entropy_trajectory_ca")
        if sa_traj is None or ca_traj is None:
            continue
        sa_curves.append(np.array(sa_traj))
        ca_curves.append(np.array(ca_traj))
    return sa_curves, ca_curves


def _plot_learning_curves(train_records, logged_epochs, stage_label, out_dir, seed,
                          val_records=None):
    """
    Learning curves for all 4 pathogen conditions.

    Train = solid lines. Val = dashed lines (same colour). If val_records is
    None, only train is plotted.
    """
    train_donor_traj = aggregate_to_donor_level(train_records)
    val_donor_traj   = aggregate_to_donor_level(val_records) if val_records else {}
    tab10 = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 5))
    for ci, cond in enumerate(ORDERED_CONDITIONS):
        color = tab10[ci % len(tab10)]
        difficulty = DIFFICULTY_MAP.get(cond, "")
        label_str = f"{cond} ({difficulty})"

        if cond in train_donor_traj:
            train_mean = np.mean(list(train_donor_traj[cond].values()), axis=0)
            ax.plot(logged_epochs, train_mean, color=color, linestyle="-",
                    linewidth=2, label=f"{label_str} — train")

        if cond in val_donor_traj:
            val_mean = np.mean(list(val_donor_traj[cond].values()), axis=0)
            ax.plot(logged_epochs, val_mean, color=color, linestyle="--",
                    linewidth=1.5, alpha=0.8, label=f"{label_str} — val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("P(Y_correct | t)")
    ax.set_title(f"{stage_label} — Oelen 2022 pathogen conditions")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    val_note = (
        f"Val donors (held out, observer-only): {VAL_DONORS}"
        if val_records else "No validation split."
    )
    plt.suptitle(
        "Metric: mean p_correct_trajectory(t), aggregated to donor level\n"
        f"(median per donor, mean across donors). {val_note}",
        fontsize=8,
    )
    plt.tight_layout()

    suffix = stage_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"learning_curves_{suffix}_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_sa_vs_ca_entropy(records, logged_epochs, all_conditions, out_dir, seed):
    """1x4 grid: per condition, SA (blue) vs CA (orange) attention entropy."""
    n_cond = len(ORDERED_CONDITIONS)
    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 4), sharey=True)

    for ax, cond in zip(axes, ORDERED_CONDITIONS):
        difficulty = DIFFICULTY_MAP.get(cond, "")
        sa_curves, ca_curves = _extract_layer_entropy(records, cond)
        mean_sa = np.zeros(len(logged_epochs))
        if sa_curves:
            mean_sa = np.mean(sa_curves, axis=0)
            mean_ca = np.mean(ca_curves, axis=0)
            ax.plot(logged_epochs, mean_sa, color="steelblue", linewidth=2,
                    label="SA (frozen)")
            ax.plot(logged_epochs, mean_ca, color="darkorange", linewidth=2,
                    linestyle="--", label="CA (trainable)")
        sa_delta = float(np.ptp(mean_sa)) if sa_curves else 0.0
        ax.set_title(f"{cond}\n({difficulty})  SA Δ={sa_delta:.4f}", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("H(a) [nats]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Stage 3 CA-only — SA vs CA attention entropy  |  Oelen 2022 pathogen\n"
        "Metric: H(a) = -sum_i a_i log(a_i). SA is frozen (should be flat); "
        "CA changes only if it learned signal.\n"
        "Mean across pseudo-tubes per condition.",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fname = f"sa_vs_ca_entropy_stage3_ca_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_ca_weight_norm(ca_weight_norm_trajectory, out_dir, seed):
    """CA weight norm trajectory — key diagnostic for whether CA learned anything."""
    epochs_all = list(range(1, len(ca_weight_norm_trajectory) + 1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs_all, ca_weight_norm_trajectory, color="darkviolet", lw=2)
    ax.axhline(ca_weight_norm_trajectory[0], color="gray", ls="--", lw=0.8,
               label="Initial norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 norm of CA parameters")
    ax.set_title(
        f"CA weight norm trajectory — Stage 3 CA-only  |  Oelen 2022\n"
        "Null hypothesis: norm stays near initial value (CA contributes nothing)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.suptitle(
        "Metric: L2 norm of [V_ca.weight, V_ca.bias, w_ca.weight, U_ca.weight] concatenated",
        fontsize=8,
    )
    plt.tight_layout()
    fname = f"ca_weight_norm_stage3_ca_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Oelen 2022 pathogen experiment: "
            "Stage 1 encoder + Stage 2 SA + Stage 3 CA-only"
        )
    )
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parent.parent / "configs" / "default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR}/run_{{timestamp}})",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override training seed from config",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(OUTPUT_DIR) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    log("Oelen 2022 pathogen experiment — SA + CA pipeline")
    log(f"Output directory: {out_dir}")
    log(f"Started: {timestamp}")
    log()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAINING_SEED = args.seed if args.seed is not None else cfg["dynamics"]["random_seeds"][0]

    log(f"Device:        {DEVICE}")
    log(f"Training seed: {TRAINING_SEED}")
    log(f"Val donors:    {VAL_DONORS}")
    log()

    # ------------------------------------------------------------------
    # 1. Data — load manifest, split train/val, build datasets
    # ------------------------------------------------------------------
    log("=" * 60)
    log("Loading data")
    log("=" * 60)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    HVG_PATH = str(Path(MANIFEST_PATH).parent / "hvg_list.json")
    with open(HVG_PATH) as f:
        gene_names = json.load(f)

    log(f"Manifest entries: {len(manifest)}")
    log(f"HVGs:             {len(gene_names)}")

    conditions = sorted({e["cytokine"] for e in manifest})
    donors     = sorted({e["donor"] for e in manifest})
    log(f"Conditions: {conditions}")
    log(f"Donors:     {len(donors)} total")

    # Donor-level train/val split
    train_manifest, val_manifest = split_manifest_by_donor(manifest, VAL_DONORS)
    train_donors = sorted({e["donor"] for e in train_manifest})
    val_donors   = sorted({e["donor"] for e in val_manifest})
    log(f"  Train donors ({len(train_donors)}): {train_donors}")
    log(f"  Val   donors ({len(val_donors)}):   {val_donors}")
    log(f"  Train tubes: {len(train_manifest)}  |  Val tubes: {len(val_manifest)}")
    log()

    # Label encoder — 4 classes (no PBS equivalent; UT is a regular class)
    label_encoder = CytokineLabel().fit(manifest)
    label_encoder.save(str(out_dir / "label_encoder.json"))
    n_classes = label_encoder.n_classes()
    log(f"Classes ({n_classes}):")
    for cond in conditions:
        log(f"  {label_encoder.encode(cond):2d}  {cond}  [{DIFFICULTY_MAP.get(cond, '')}]")
    log()

    # Save manifests
    TRAIN_MANIFEST_PATH = str(out_dir / "manifest_train.json")
    VAL_MANIFEST_PATH   = str(out_dir / "manifest_val.json")
    with open(TRAIN_MANIFEST_PATH, "w") as f:
        json.dump(train_manifest, f)
    with open(VAL_MANIFEST_PATH, "w") as f:
        json.dump(val_manifest, f)

    log("Preloading tube datasets...")
    train_tube_dataset = PseudoTubeDataset(
        TRAIN_MANIFEST_PATH, label_encoder, gene_names=gene_names, preload=True
    )
    val_tube_dataset = PseudoTubeDataset(
        VAL_MANIFEST_PATH, label_encoder, gene_names=gene_names, preload=True
    )
    log(f"Train tubes loaded: {len(train_tube_dataset)}")
    log(f"Val   tubes loaded: {len(val_tube_dataset)}")

    # Stage 1 manifest (one tube per condition, rotating donors — uses train only)
    STAGE1_MANIFEST_PATH = str(out_dir / "manifest_stage1.json")
    build_stage1_manifest(train_manifest, save_path=STAGE1_MANIFEST_PATH)
    cell_dataset = CellDataset(STAGE1_MANIFEST_PATH, gene_names=gene_names, preload=True)
    log(f"Cells: {len(cell_dataset)}  |  Cell types: {cell_dataset.n_cell_types()}")
    cell_loader = DataLoader(cell_dataset, batch_size=256, shuffle=True, num_workers=0)

    embed_dim        = cfg["model"]["embedding_dim"]
    attention_hidden = cfg["model"]["attention_hidden_dim"]

    # ------------------------------------------------------------------
    # 2. Stage 1 — Encoder pre-training
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Stage 1 — Encoder pre-training (cell-type classification)")
    log("=" * 60)

    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=cell_dataset.n_cell_types(),
        embed_dim=embed_dim,
    )
    encoder = train_encoder(
        encoder,
        cell_loader,
        n_epochs=cfg["training"]["stage1_epochs"],
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        device=DEVICE,
        verbose=True,
    )
    enc_path = str(out_dir / f"encoder_stage1_{TRAINING_SEED}.pt")
    torch.save(encoder.state_dict(), enc_path)
    log(f"Encoder saved: {enc_path}")

    # ------------------------------------------------------------------
    # 3. Stage 2 — Single-layer MIL (this becomes the frozen SA)
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Stage 2 — Single-layer MIL training (encoder frozen)")
    log("=" * 60)

    stage2_lr       = 0.0002
    stage2_momentum = 0.95

    mil_model = build_mil_model(
        encoder,
        embed_dim=embed_dim,
        attention_hidden_dim=attention_hidden,
        n_classes=n_classes,
        encoder_frozen=True,
    )
    dynamics_stage2 = train_mil(
        mil_model,
        train_tube_dataset,
        n_epochs=cfg["training"]["stage2_epochs"],
        lr=stage2_lr,
        momentum=stage2_momentum,
        lr_scheduler=cfg["training"]["lr_scheduler"],
        lr_warmup_epochs=cfg["training"]["lr_warmup_epochs"],
        log_every_n_epochs=cfg["dynamics"]["log_every_n_epochs"],
        device=DEVICE,
        seed=TRAINING_SEED,
        verbose=True,
        val_dataset=val_tube_dataset,
    )
    mil_path = str(out_dir / f"mil_stage2_{TRAINING_SEED}.pt")
    torch.save(mil_model.state_dict(), mil_path)
    log(f"Stage 2 model saved: {mil_path}")
    log(f"Train records: {len(dynamics_stage2['records'])}")
    log(f"Val   records: {len(dynamics_stage2['val_records'])}")

    with open(out_dir / "dynamics_stage2.pkl", "wb") as fh:
        pickle.dump({
            "records":       dynamics_stage2["records"],
            "val_records":   dynamics_stage2["val_records"],
            "logged_epochs": dynamics_stage2["logged_epochs"],
        }, fh)
    log("Stage 2 dynamics saved (dynamics_stage2.pkl).")

    # ------------------------------------------------------------------
    # 4. Stage 3 — CA-only training
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Stage 3 — CA-only training")
    log("=" * 60)

    stage3_model = Stage3CAModel(
        mil_model,
        embed_dim=embed_dim,
        attention_hidden_dim=attention_hidden,
    )
    stage3_model = stage3_model.to(DEVICE)

    trainable_params = sum(
        p.numel() for p in stage3_model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in stage3_model.parameters())
    log(f"Trainable parameters (CA only): {trainable_params}")
    log(f"Total parameters:               {total_params}")
    log(f"Initial CA weight norm:         {stage3_model.ca_weight_norm():.6f}")
    log()

    n_epochs  = STAGE3_EPOCHS
    log_every = cfg["dynamics"]["log_every_n_epochs"]
    lr        = stage2_lr
    momentum  = stage2_momentum

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, stage3_model.parameters()),
        lr=lr,
        momentum=momentum,
    )
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(TRAINING_SEED)

    log(f"n_epochs:      {n_epochs}  (fixed override, independent of config)")
    log(f"lr:            {lr}")
    log(f"momentum:      {momentum}")
    log(f"log_every:     {log_every}")
    log(f"training_seed: {TRAINING_SEED}")
    log()

    train_entries = train_tube_dataset.get_entries()
    val_entries   = val_tube_dataset.get_entries()

    train_tube_traj    = _init_tube_trajectories(train_entries)
    val_tube_traj      = _init_tube_trajectories(val_entries)
    train_confusion    = defaultdict(list)
    val_confusion      = defaultdict(list)

    queues = build_cytokine_queues(train_entries, label_encoder)

    logged_epochs             = []
    ca_weight_norm_trajectory = []
    loss_trajectory           = []

    for epoch in range(1, n_epochs + 1):
        epoch_loss = _train_epoch_stage3(
            stage3_model, train_tube_dataset, queues, optimizer, criterion, DEVICE, rng
        )
        loss_trajectory.append(epoch_loss)
        ca_weight_norm_trajectory.append(stage3_model.ca_weight_norm())

        if epoch % log_every == 0 or epoch == n_epochs:
            logged_epochs.append(epoch)
            _log_dynamics_stage3(
                stage3_model, train_tube_dataset, train_entries,
                train_tube_traj, train_confusion, label_encoder, DEVICE,
            )
            _log_dynamics_stage3(
                stage3_model, val_tube_dataset, val_entries,
                val_tube_traj, val_confusion, label_encoder, DEVICE,
            )
            stage3_model.train()

        print(
            f"[Stage 3 CA] Epoch {epoch:3d}/{n_epochs} | "
            f"loss={epoch_loss:.4f} | "
            f"CA norm={ca_weight_norm_trajectory[-1]:.6f}",
            flush=True,
        )

    # Sanity check: SA entropy variance should be near zero (frozen)
    if train_entries:
        first_traj = train_tube_traj[0]["entropy"]
        if len(first_traj) > 1:
            sa_variance = float(np.var(first_traj))
            if sa_variance > 1e-6:
                log(f"WARNING: SA entropy variance = {sa_variance:.2e} "
                    f"(expected ~0 for frozen SA). Freezing may be broken.")
            else:
                log(f"SA entropy variance check passed: var={sa_variance:.2e} < 1e-6")

    # ------------------------------------------------------------------
    # Build records and save artifacts
    # ------------------------------------------------------------------
    train_records = _build_records(train_entries, train_tube_traj)
    val_records   = _build_records(val_entries,   val_tube_traj)
    log(f"Train records: {len(train_records)}")
    log(f"Val   records: {len(val_records)}")

    model_path = str(out_dir / f"stage3_ca_model_{TRAINING_SEED}.pt")
    torch.save(stage3_model.state_dict(), model_path)
    log(f"Model saved: {model_path}")

    with open(out_dir / "dynamics_stage3.pkl", "wb") as fh:
        pickle.dump({
            "records":                          train_records,
            "val_records":                      val_records,
            "logged_epochs":                    logged_epochs,
            "ca_weight_norm_trajectory":        ca_weight_norm_trajectory,
            "loss_trajectory":                  loss_trajectory,
            "confusion_entropy_trajectory":     {
                k: np.array(v) for k, v in train_confusion.items()
            },
            "val_confusion_entropy_trajectory": {
                k: np.array(v) for k, v in val_confusion.items()
            },
        }, fh)
    log("Dynamics saved: dynamics_stage3.pkl")

    # ------------------------------------------------------------------
    # 5. Post-training analysis — Stage 2
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Dynamics analysis — Stage 2 (single-layer SA)")
    log("=" * 60)

    s2_train_donor = aggregate_to_donor_level(dynamics_stage2["records"])
    s2_val_donor   = aggregate_to_donor_level(dynamics_stage2["val_records"])
    s2_learn_train = rank_cytokines_by_learnability(s2_train_donor, exclude=[])
    s2_learn_val   = rank_cytokines_by_learnability(s2_val_donor,   exclude=[])
    s2_ranking     = s2_learn_train["ranking"]

    log()
    log("Condition learnability ranking — Stage 2")
    log(f"Metric: {s2_learn_train['metric_description']}")
    log()
    hdr = f"{'Rank':>4}  {'Condition':<12}  {'Train AUC':>9}  {'Val AUC':>8}  Difficulty"
    log(hdr)
    log("-" * 55)
    s2_val_auc = {c: a for c, a in s2_learn_val["ranking"]}
    for i, (cond, auc) in enumerate(s2_ranking, 1):
        difficulty = DIFFICULTY_MAP.get(cond, "")
        vauc = s2_val_auc.get(cond, float("nan"))
        log(f"  {i:2d}.  {cond:<12}  {auc:>9.3f}  {vauc:>8.3f}  {difficulty}")

    # ------------------------------------------------------------------
    # 6. Post-training analysis — Stage 3
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Dynamics analysis — Stage 3 CA-only")
    log("=" * 60)

    s3_train_donor = aggregate_to_donor_level(train_records)
    s3_val_donor   = aggregate_to_donor_level(val_records)
    s3_learn_train = rank_cytokines_by_learnability(s3_train_donor, exclude=[])
    s3_learn_val   = rank_cytokines_by_learnability(s3_val_donor,   exclude=[])
    ranking        = s3_learn_train["ranking"]

    log()
    log("Condition learnability ranking — Stage 3 CA-only")
    log(f"Metric: {s3_learn_train['metric_description']}")
    log()
    log(hdr)
    log("-" * 55)
    s3_val_auc = {c: a for c, a in s3_learn_val["ranking"]}
    for i, (cond, auc) in enumerate(ranking, 1):
        difficulty = DIFFICULTY_MAP.get(cond, "")
        vauc = s3_val_auc.get(cond, float("nan"))
        log(f"  {i:2d}.  {cond:<12}  {auc:>9.3f}  {vauc:>8.3f}  {difficulty}")

    # Stage 2 vs Stage 3 rank correlation
    s2_order = [c for c, _ in s2_ranking]
    s3_order = [c for c, _ in ranking]
    if set(s2_order) == set(s3_order) and len(s2_order) >= 2:
        s3_rank_by_cyt  = {c: i for i, c in enumerate(s3_order)}
        s3_aligned      = [s3_rank_by_cyt.get(c, len(s3_order)) for c in s2_order]
        rho_stage, pval_stage = spearmanr(range(len(s2_order)), s3_aligned)
        log()
        log(f"Stage 2 vs Stage 3 rank correlation: Spearman rho = {rho_stage:.3f}  "
            f"(p={pval_stage:.3f})")
        log(f"Stable (rho > 0.7): {rho_stage > 0.7}")

    # Confusion entropy
    log()
    log("-" * 60)
    confusion_result = compute_confusion_entropy_summary(
        {k: np.array(v) for k, v in train_confusion.items()}, exclude=[]
    )
    log("Condition confusion entropy ranking — Stage 3 CA-only")
    log(f"Metric: {confusion_result['metric_description']}")
    log()
    log(f"{'Condition':<14}  {'Train AUC(H_c)':>14}  Difficulty")
    log("-" * 40)
    for cond, auc in confusion_result["ranking"]:
        difficulty = DIFFICULTY_MAP.get(cond, "")
        log(f"  {cond:<14}  {auc:>14.3f}  {difficulty}")

    # Biological validation check: PA > CA > MTB ordering
    log()
    log("=" * 60)
    log("Biological validation — expected ordering: PA > CA > MTB")
    log("=" * 60)
    auc_map = {cond: auc for cond, auc in ranking}
    pa_auc  = auc_map.get("24hPA",  float("nan"))
    ca_auc  = auc_map.get("24hCA",  float("nan"))
    mtb_auc = auc_map.get("24hMTB", float("nan"))
    log(f"  24hPA  (EASY):  {pa_auc:.3f}")
    log(f"  24hCA  (MED):   {ca_auc:.3f}")
    log(f"  24hMTB (HARD):  {mtb_auc:.3f}")
    pa_gt_ca  = pa_auc  > ca_auc
    ca_gt_mtb = ca_auc  > mtb_auc
    pa_gt_mtb = pa_auc  > mtb_auc
    log(f"  PA > CA:  {pa_gt_ca}   (expected: True)")
    log(f"  CA > MTB: {ca_gt_mtb}  (expected: True)")
    log(f"  PA > MTB: {pa_gt_mtb}  (expected: True)")
    n_correct = sum([pa_gt_ca, ca_gt_mtb, pa_gt_mtb])
    log(f"  Ordering checks passed: {n_correct}/3")

    # ------------------------------------------------------------------
    # 7. Figures
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Saving figures...")
    log("=" * 60)

    fname_s2 = _plot_learning_curves(
        dynamics_stage2["records"], dynamics_stage2["logged_epochs"],
        "Stage 2 (SA)", out_dir, TRAINING_SEED,
        val_records=dynamics_stage2["val_records"],
    )
    log(f"  Saved: {fname_s2}")

    fname_s3 = _plot_learning_curves(
        train_records, logged_epochs,
        "Stage 3 CA-only", out_dir, TRAINING_SEED,
        val_records=val_records,
    )
    log(f"  Saved: {fname_s3}")

    fname_ent = _plot_sa_vs_ca_entropy(
        train_records, logged_epochs, conditions, out_dir, TRAINING_SEED,
    )
    log(f"  Saved: {fname_ent}")

    fname_norm = _plot_ca_weight_norm(ca_weight_norm_trajectory, out_dir, TRAINING_SEED)
    log(f"  Saved: {fname_norm}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    log()
    log(f"All results saved to: {out_dir}")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
