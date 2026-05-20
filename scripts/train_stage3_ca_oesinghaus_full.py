"""
Stage 3 CA-only training on the full 91-class Oesinghaus dataset.

Reuses an existing Stage 2 model from
``results/oesinghaus_full_v2/seed_{SEED}/model_stage2.pt``
as the frozen SA base. Trains only the CA head (~16K parameters) for
``--stage3_epochs`` epochs and saves SA + CA attention trajectories per
tube for downstream geo / ablation analysis on the SA and CA centroids.

This is the architectural-fix path for directional cascade inference
(see ``reports/cascade_pairs/literature_review.md`` §8 and CLAUDE.md §5.5):
SA → direct A-responders, CA → cascade relays. The asymmetry encodes
direction.

Usage
-----
    python scripts/train_stage3_ca_oesinghaus_full.py \\
        --seed 42 \\
        --stage2_dir results/oesinghaus_full_v2/seed_42 \\
        --output_dir results/oesinghaus_stage3_ca/seed_42 \\
        --stage3_epochs 100

Required inputs at --stage2_dir:
    model_stage2.pt        - trained CytokineABMIL (v1) Stage 2 model
    encoder_stage1.pt      - trained InstanceEncoder
    label_encoder.json     - CytokineLabel encoder
    manifest_train.json    - training manifest
    manifest_val.json      - validation manifest
    manifest_stage1.json   - Stage 1 cell-type manifest (for encoder shape)

Outputs at --output_dir:
    stage3_ca_model_{seed}.pt          - trained CA wrapper
    dynamics_stage3.pkl                - per-tube SA + CA attention trajectories
    ca_weight_norm_stage3_ca_{seed}.png
    sa_vs_ca_entropy_stage3_ca_{seed}.png (top 10 cytokines)
    run_log.txt
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model
from cytokine_mil.models.stage3_ca_model import Stage3CAModel
from cytokine_mil.training.trainer import (
    build_cytokine_queues,
    generate_epoch_megabatches,
)


# ---------------------------------------------------------------------------
# Dynamics helpers (mirror train_stage3_ca.py internals; kept self-contained)
# ---------------------------------------------------------------------------

def _compute_entropy(a: torch.Tensor) -> float:
    """Shannon entropy of attention weights (nats). Clipped for stability."""
    a_safe = a.clamp(min=1e-10)
    return float(-(a_safe * a_safe.log()).sum())


def _init_tube_trajectories(entries):
    return {
        i: {
            "p_correct": [],
            "entropy_sa": [],
            "entropy_ca": [],
            "instance_confidence_sa": [],
            "instance_confidence_ca": [],
            "softmax": [],
        }
        for i in range(len(entries))
    }


@torch.no_grad()
def _log_dynamics(model, dataset, entries, tube_traj,
                  cytokine_confusion_epochs, label_encoder, device):
    model.eval()
    for idx, _entry in enumerate(entries):
        X, label, _donor, _cyt_name = dataset[idx]
        X = X.to(device)
        y_hat, a_SA, a_CA, _H = model(X)
        probs = F.softmax(y_hat, dim=0)
        p_correct = float(probs[label].item())
        tube_traj[idx]["p_correct"].append(p_correct)
        tube_traj[idx]["entropy_sa"].append(_compute_entropy(a_SA))
        tube_traj[idx]["entropy_ca"].append(_compute_entropy(a_CA))
        tube_traj[idx]["instance_confidence_sa"].append(
            (a_SA * p_correct).cpu().numpy()
        )
        tube_traj[idx]["instance_confidence_ca"].append(
            (a_CA * p_correct).cpu().numpy()
        )
        tube_traj[idx]["softmax"].append(probs.cpu().numpy())

    # Compute per-cytokine confusion entropy (off-diagonal renormalized).
    cyt_indices = defaultdict(list)
    for idx, entry in enumerate(entries):
        cyt_indices[entry["cytokine"]].append(idx)
    for cyt, indices in cyt_indices.items():
        true_label = label_encoder.encode(cyt)
        softmaxes = np.stack([tube_traj[i]["softmax"][-1] for i in indices])
        mean_softmax = softmaxes.mean(axis=0)
        off_diag = np.concatenate([mean_softmax[:true_label],
                                   mean_softmax[true_label + 1:]])
        off_sum = float(off_diag.sum())
        if off_sum < 1e-10:
            cytokine_confusion_epochs[cyt].append(0.0)
            continue
        q = np.clip(off_diag / off_sum, 1e-10, None)
        cytokine_confusion_epochs[cyt].append(
            -float((q * np.log(q)).sum())
        )


def _train_epoch_stage3(model, train_dataset, queues, optimizer, criterion,
                        device, rng):
    model.train()
    megabatches = generate_epoch_megabatches(queues, rng)
    total_loss = 0.0
    n_mb = max(len(megabatches), 1)
    for mb_indices in megabatches:
        optimizer.zero_grad()
        mb_loss = torch.tensor(0.0, device=device)
        n = len(mb_indices)
        for _cyt_idx, ds_idx in mb_indices.items():
            X, label, _donor, _cyt_name = train_dataset[ds_idx]
            X = X.to(device)
            label_t = torch.tensor([label], dtype=torch.long, device=device)
            y_hat, _a_SA, _a_CA, _H = model(X)
            loss = criterion(y_hat.unsqueeze(0), label_t) / n
            mb_loss = mb_loss + loss
        mb_loss.backward()
        optimizer.step()
        total_loss += mb_loss.item()
    return total_loss / n_mb


def _build_records(entries, tube_traj):
    records = []
    for idx, entry in enumerate(entries):
        traj = tube_traj[idx]
        ic_sa = (np.stack(traj["instance_confidence_sa"], axis=0).T
                 if traj["instance_confidence_sa"] else None)
        ic_ca = (np.stack(traj["instance_confidence_ca"], axis=0).T
                 if traj["instance_confidence_ca"] else None)
        records.append({
            "cytokine": entry["cytokine"],
            "donor": entry["donor"],
            "tube_idx": entry["tube_idx"],
            "tube_path": entry["path"],
            "n_cells": entry["n_cells"],
            "p_correct_trajectory": traj["p_correct"],
            "entropy_trajectory_sa": traj["entropy_sa"],
            "entropy_trajectory_ca": traj["entropy_ca"],
            "confidence_trajectory_sa": ic_sa,
            "confidence_trajectory_ca": ic_ca,
        })
    return records


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_sa_vs_ca_entropy_top(train_records, logged_epochs, seed, out_dir):
    """Plot SA vs CA entropy for the 10 most-learnable cytokines by p_correct AUC."""
    cyt_p_correct = defaultdict(list)
    for rec in train_records:
        if rec["p_correct_trajectory"]:
            cyt_p_correct[rec["cytokine"]].append(
                float(np.trapz(rec["p_correct_trajectory"]) / len(rec["p_correct_trajectory"]))
            )
    cyt_means = {c: float(np.mean(v)) for c, v in cyt_p_correct.items() if v}
    top_cyts = sorted(cyt_means, key=cyt_means.get, reverse=True)[:10]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()
    for ax, cyt in zip(axes, top_cyts):
        sa_curves, ca_curves = [], []
        for rec in train_records:
            if rec["cytokine"] != cyt:
                continue
            sa_curves.append(np.array(rec["entropy_trajectory_sa"]))
            ca_curves.append(np.array(rec["entropy_trajectory_ca"]))
        if sa_curves:
            ax.plot(logged_epochs, np.mean(sa_curves, axis=0), color="steelblue",
                    lw=2, label="SA (frozen)")
            ax.plot(logged_epochs, np.mean(ca_curves, axis=0), color="darkorange",
                    lw=2, ls="--", label="CA (trainable)")
        ax.set_title(cyt, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("H(a) [nats]")
        ax.legend(fontsize=7)
    fig.suptitle(
        f"Stage 3 CA-only — SA vs CA attention entropy  |  Oesinghaus seed={seed}\n"
        "Top 10 cytokines by training p_correct AUC. "
        "SA frozen (flat); CA changes only if it learned signal.",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / f"sa_vs_ca_entropy_stage3_ca_{seed}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ca_weight_norm(ca_norm_traj, seed, out_dir):
    epochs = list(range(1, len(ca_norm_traj) + 1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, ca_norm_traj, color="darkviolet", lw=2)
    ax.axhline(ca_norm_traj[0], color="gray", ls="--", lw=0.8, label="Initial norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 norm of CA parameters")
    ax.set_title(
        f"CA weight norm trajectory — Oesinghaus Stage 3 CA-only  |  seed={seed}\n"
        "Null hypothesis: norm stays at initial value (CA contributes nothing)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"ca_weight_norm_stage3_ca_{seed}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True,
                   help="Seed (controls model init + ordering).")
    p.add_argument("--stage2_dir", type=str, required=True,
                   help="Directory containing model_stage2.pt + encoder_stage1.pt "
                        "+ label_encoder.json + manifest_train.json + manifest_val.json "
                        "+ manifest_stage1.json.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for Stage 3 CA artifacts.")
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "configs" / "default.yaml"))
    p.add_argument("--stage3_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--momentum", type=float, default=0.95)
    return p.parse_args()


def main():
    args = parse_args()

    stage2_dir = Path(args.stage2_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    log(f"Stage 3 CA-only — full Oesinghaus 91-class  |  seed={args.seed}")
    log(f"Loading Stage 2 base from: {stage2_dir}")
    log(f"Output directory:          {out_dir}")
    log(f"Started: {timestamp}")
    log()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Stage 3 epochs: {args.stage3_epochs}")
    log(f"LR: {args.lr}  momentum: {args.momentum}")

    # ------------------------------------------------------------------
    # Load existing artifacts from Stage 2 run
    # ------------------------------------------------------------------
    for needed in ["model_stage2.pt", "encoder_stage1.pt", "label_encoder.json",
                   "manifest_train.json", "manifest_val.json", "manifest_stage1.json"]:
        if not (stage2_dir / needed).exists():
            log(f"ERROR: missing required file in stage2_dir: {needed}")
            sys.exit(1)

    label_encoder = CytokineLabel.load(str(stage2_dir / "label_encoder.json"))
    log(f"Classes: {label_encoder.n_classes()}")

    HVG_PATH = str(Path(cfg["data"]["manifest_path"]).parent / "hvg_list.json")
    with open(HVG_PATH) as f:
        gene_names = json.load(f)
    log(f"HVGs: {len(gene_names)}")

    train_manifest_path = str(stage2_dir / "manifest_train.json")
    val_manifest_path = str(stage2_dir / "manifest_val.json")
    stage1_manifest_path = str(stage2_dir / "manifest_stage1.json")

    log("Preloading datasets...")
    train_tube_dataset = PseudoTubeDataset(
        train_manifest_path, label_encoder, gene_names=gene_names, preload=True
    )
    val_tube_dataset = PseudoTubeDataset(
        val_manifest_path, label_encoder, gene_names=gene_names, preload=True
    )
    log(f"Train tubes: {len(train_tube_dataset)}")
    log(f"Val tubes:   {len(val_tube_dataset)}")

    cell_dataset = CellDataset(stage1_manifest_path, gene_names=gene_names, preload=True)
    n_cell_types = cell_dataset.n_cell_types()
    log(f"Cells: {len(cell_dataset)}  Cell types: {n_cell_types}")

    embed_dim = cfg["model"]["embedding_dim"]
    attention_hidden = cfg["model"]["attention_hidden_dim"]
    n_classes = label_encoder.n_classes()

    # ------------------------------------------------------------------
    # Rebuild encoder + Stage 2 SA model + load weights
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Loading Stage 2 model")
    log("=" * 60)
    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=n_cell_types,
        embed_dim=embed_dim,
    )
    mil_model = build_mil_model(
        encoder,
        embed_dim=embed_dim,
        attention_hidden_dim=attention_hidden,
        n_classes=n_classes,
        encoder_frozen=True,
    )
    sa_state_dict = torch.load(stage2_dir / "model_stage2.pt", map_location="cpu")
    mil_model.load_state_dict(sa_state_dict)
    mil_model.to(device)
    log(f"Loaded {stage2_dir / 'model_stage2.pt'}")

    # ------------------------------------------------------------------
    # Wrap in Stage 3 CA model
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Stage 3 — CA-only training")
    log("=" * 60)
    stage3_model = Stage3CAModel(
        mil_model,
        embed_dim=embed_dim,
        attention_hidden_dim=attention_hidden,
    ).to(device)
    trainable = sum(p.numel() for p in stage3_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in stage3_model.parameters())
    log(f"Trainable parameters (CA only): {trainable}")
    log(f"Total parameters:               {total}")
    log(f"Initial CA weight norm:         {stage3_model.ca_weight_norm():.6f}")

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, stage3_model.parameters()),
        lr=args.lr, momentum=args.momentum,
    )
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_entries = train_tube_dataset.get_entries()
    val_entries = val_tube_dataset.get_entries()
    train_tube_traj = _init_tube_trajectories(train_entries)
    val_tube_traj = _init_tube_trajectories(val_entries)
    train_conf_epochs = defaultdict(list)
    val_conf_epochs = defaultdict(list)

    queues = build_cytokine_queues(train_entries, label_encoder)
    log_every = cfg["dynamics"]["log_every_n_epochs"]
    logged_epochs = []
    ca_norm_traj = []
    loss_traj = []

    for epoch in range(1, args.stage3_epochs + 1):
        epoch_loss = _train_epoch_stage3(
            stage3_model, train_tube_dataset, queues, optimizer, criterion, device, rng,
        )
        loss_traj.append(epoch_loss)
        ca_norm_traj.append(stage3_model.ca_weight_norm())
        if epoch % log_every == 0 or epoch == args.stage3_epochs:
            logged_epochs.append(epoch)
            _log_dynamics(stage3_model, train_tube_dataset, train_entries,
                          train_tube_traj, train_conf_epochs, label_encoder, device)
            _log_dynamics(stage3_model, val_tube_dataset, val_entries,
                          val_tube_traj, val_conf_epochs, label_encoder, device)
            stage3_model.train()
        if epoch % 10 == 0 or epoch <= 5 or epoch == args.stage3_epochs:
            log(f"[Stage 3 CA] Epoch {epoch:3d}/{args.stage3_epochs} | "
                f"loss={epoch_loss:.4f} | CA_norm={ca_norm_traj[-1]:.6f}")

    # SA entropy variance check (should be ~0)
    if train_entries:
        first_traj = train_tube_traj[0]["entropy_sa"]
        if len(first_traj) > 1:
            sa_var = float(np.var(first_traj))
            log(f"SA entropy variance check: var={sa_var:.2e} "
                f"({'PASS' if sa_var < 1e-6 else 'FAIL — freeze may be broken'})")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    train_records = _build_records(train_entries, train_tube_traj)
    val_records = _build_records(val_entries, val_tube_traj)
    log(f"Train records: {len(train_records)}")
    log(f"Val records:   {len(val_records)}")

    torch.save(stage3_model.state_dict(),
               str(out_dir / f"stage3_ca_model_{args.seed}.pt"))
    log(f"Model saved: stage3_ca_model_{args.seed}.pt")

    with open(out_dir / "dynamics_stage3.pkl", "wb") as fh:
        pickle.dump({
            "records": train_records,
            "val_records": val_records,
            "logged_epochs": logged_epochs,
            "ca_weight_norm_trajectory": ca_norm_traj,
            "loss_trajectory": loss_traj,
            "confusion_entropy_trajectory": {
                k: np.array(v) for k, v in train_conf_epochs.items()
            },
            "val_confusion_entropy_trajectory": {
                k: np.array(v) for k, v in val_conf_epochs.items()
            },
        }, fh)
    log("dynamics_stage3.pkl saved")

    log()
    log("Saving figures...")
    _plot_sa_vs_ca_entropy_top(train_records, logged_epochs, args.seed, out_dir)
    log(f"  Saved: sa_vs_ca_entropy_stage3_ca_{args.seed}.png")
    _plot_ca_weight_norm(ca_norm_traj, args.seed, out_dir)
    log(f"  Saved: ca_weight_norm_stage3_ca_{args.seed}.png")

    log()
    log(f"All results saved to: {out_dir}")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
