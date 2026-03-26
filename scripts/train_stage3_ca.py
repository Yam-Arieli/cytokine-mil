"""
Stage 3 CA-only training experiment — cluster job script.

Loads a fully trained Stage 2 CytokineABMIL (v1) checkpoint, freezes encoder,
SA attention, and classifier, then trains only the new CA layer. Tests whether
CA adds genuine signal beyond what SA + classifier already solved.

Derives all paths from the Stage 2 checkpoint directory:
  - cytokine_groups.json
  - label_encoder.json
  - manifest_train.json
  - manifest_val.json

Results are saved to:
    results/stage3_ca_seed{BOOTSTRAP_SEED}_{timestamp}/

Usage:
    python scripts/train_stage3_ca.py --stage2_checkpoint <path_to_v1_checkpoint.pt>
    python scripts/train_stage3_ca.py --stage2_checkpoint <path> --bootstrap_seed 42
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
from scipy.stats import mannwhitneyu, spearmanr

from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.models.stage3_ca_model import Stage3CAModel
from cytokine_mil.experiment_setup import (
    build_encoder,
    build_mil_model,
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
# Local dynamics helpers (mirrors train_mil.py internals)
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
    Compute one confusion-entropy snapshot per cytokine from the latest softmax snapshot.

    Steps per cytokine C:
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
    """Return SA and CA entropy curves (one array per tube) for a cytokine."""
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


def _plot_learning_curves(records, val_records, logged_epochs,
                          simple_cyts, complex_cyts, bootstrap_seed, out_dir):
    """Learning curves: train solid, val dotted, tab10 colors."""
    donor_traj = aggregate_to_donor_level(records)
    val_donor_traj = aggregate_to_donor_level(val_records)
    tab10 = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, group_cyts, group_label in zip(
        axes,
        [simple_cyts, complex_cyts],
        ["SIMPLE group (direct, PBMC-specific responses)",
         "COMPLEX group (indirect / non-PBMC-primary)"],
    ):
        for ci, cyt in enumerate(group_cyts):
            color = tab10[ci % len(tab10)]
            if cyt in donor_traj:
                train_mean = np.mean(list(donor_traj[cyt].values()), axis=0)
                ax.plot(logged_epochs, train_mean, color=color, linestyle="-",
                        alpha=0.9, label=f"{cyt} (train)")
            if cyt in val_donor_traj:
                val_mean = np.mean(list(val_donor_traj[cyt].values()), axis=0)
                ax.plot(logged_epochs, val_mean, color=color, linestyle=":",
                        alpha=0.55, label=f"{cyt} (val)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("P(Y_correct | t)")
        ax.set_title(group_label)
        ax.legend(fontsize=7, ncol=2)

    axes[1].annotate(
        "Solid = train donors (10)\nDotted = val donors (D2, D3)",
        xy=(0.02, 0.05), xycoords="axes fraction", fontsize=7, color="gray",
    )
    plt.suptitle(
        f"Stage 3 CA-only learning curves  |  Bootstrap seed: {bootstrap_seed}\n"
        "Metric: mean p_correct_trajectory(t), aggregated to donor level "
        "(median per donor, mean across donors)",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"learning_curves_stage3_ca_{bootstrap_seed}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sa_vs_ca_entropy(records, logged_epochs, simple_cyts, complex_cyts,
                           bootstrap_seed, out_dir):
    """2x5 grid: per cytokine, SA (blue) vs CA (orange) attention entropy over training."""
    all_cyts = simple_cyts + complex_cyts
    group_map = {c: "SIMPLE" for c in simple_cyts}
    group_map.update({c: "COMPLEX" for c in complex_cyts})

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    for ax, cyt in zip(axes, all_cyts):
        group = group_map.get(cyt, "")
        sa_curves, ca_curves = _extract_layer_entropy(records, cyt)
        if sa_curves:
            mean_sa = np.mean(sa_curves, axis=0)
            mean_ca = np.mean(ca_curves, axis=0)
            ax.plot(logged_epochs, mean_sa, color="steelblue", linewidth=2, label="SA (frozen)")
            ax.plot(logged_epochs, mean_ca, color="darkorange", linewidth=2,
                    linestyle="--", label="CA (trainable)")
        ax.set_title(f"{cyt}\n({group})", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("H(a) [nats]")
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Stage 3 CA-only — SA vs CA attention entropy  |  Bootstrap seed: {bootstrap_seed}\n"
        "Metric: H(a) = -sum_i a_i log(a_i). SA is frozen (should be flat); "
        "CA changes only if it learned signal.\n"
        "Mean across pseudo-tubes per cytokine.",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"sa_vs_ca_entropy_stage3_ca_{bootstrap_seed}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ca_weight_norm(ca_weight_norm_trajectory, bootstrap_seed, out_dir):
    """CA weight norm trajectory — key diagnostic for whether CA learned anything."""
    epochs_all = list(range(1, len(ca_weight_norm_trajectory) + 1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs_all, ca_weight_norm_trajectory, color="darkviolet", lw=2)
    ax.axhline(ca_weight_norm_trajectory[0], color="gray", ls="--", lw=0.8,
               label="Initial norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 norm of CA parameters")
    ax.set_title(
        f"CA weight norm trajectory — Stage 3 CA-only  |  Bootstrap seed: {bootstrap_seed}\n"
        "Null hypothesis: norm stays near initial value (CA contributes nothing)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.suptitle(
        "Metric: L2 norm of [V_ca.weight, V_ca.bias, w_ca.weight, U_ca.weight] concatenated",
        fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"ca_weight_norm_stage3_ca_{bootstrap_seed}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_latest_stage2_checkpoint(script_dir: Path):
    """
    Auto-discover the most recently modified Stage 2 v1 checkpoint.

    Searches results/ relative to the script file (same convention as
    run_2al_bootstrap.py), then one and two levels up as fallback.
    """
    search_roots = [
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
    ]
    candidates = []
    for root in search_roots:
        candidates.extend(root.glob("results/*/mil_stage2_bootstrap_*.pt"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 CA-only training experiment"
    )
    parser.add_argument("--config", type=str,
                        default="../../configs/default.yaml",
                        help="Path to YAML config (default: ../../configs/default.yaml)")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Path to Stage 2 v1 model checkpoint (.pt). "
                             "If omitted, auto-discovers the most recent one under results/.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/stage3_ca_seed{seed}_{timestamp})")
    parser.add_argument("--bootstrap_seed", type=int, default=None,
                        help="Bootstrap seed (default: inferred from checkpoint filename, else 42)")
    args = parser.parse_args()

    # Auto-discover checkpoint if not provided
    if args.stage2_checkpoint is None:
        checkpoint_path = _find_latest_stage2_checkpoint(Path(__file__).parent)
        if checkpoint_path is None:
            print("ERROR: No Stage 2 checkpoint found under results/. "
                  "Pass --stage2_checkpoint explicitly.")
            sys.exit(1)
        print(f"Auto-discovered Stage 2 checkpoint: {checkpoint_path}", flush=True)
    else:
        checkpoint_path = Path(args.stage2_checkpoint)

    # Infer bootstrap seed from checkpoint filename if not provided
    if args.bootstrap_seed is not None:
        BOOTSTRAP_SEED = args.bootstrap_seed
    else:
        import re
        m = re.search(r"mil_stage2_bootstrap_(\d+)\.pt", checkpoint_path.name)
        BOOTSTRAP_SEED = int(m.group(1)) if m else 42
        print(f"Bootstrap seed inferred from checkpoint: {BOOTSTRAP_SEED}", flush=True)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent / "results" / f"stage3_ca_seed{BOOTSTRAP_SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    log(f"Stage 3 CA-only training experiment — seed={BOOTSTRAP_SEED}")
    log(f"Stage 2 checkpoint: {checkpoint_path}")
    log(f"Output directory:   {out_dir}")
    log(f"Started: {timestamp}")
    log()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {DEVICE}")
    log()

    # ------------------------------------------------------------------
    # Derive paths from Stage 2 checkpoint directory
    # ------------------------------------------------------------------
    stage2_dir = checkpoint_path.parent

    with open(stage2_dir / "cytokine_groups.json") as f:
        groups = json.load(f)
    SIMPLE_CYTOKINES = groups["simple_cytokines"]
    COMPLEX_CYTOKINES = groups["complex_cytokines"]
    SUBSET_CYTOKINES = SIMPLE_CYTOKINES + COMPLEX_CYTOKINES

    log(f"Simple cytokines  ({len(SIMPLE_CYTOKINES)}): {SIMPLE_CYTOKINES}")
    log(f"Complex cytokines ({len(COMPLEX_CYTOKINES)}): {COMPLEX_CYTOKINES}")
    log()

    label_encoder = CytokineLabel()
    label_encoder.load(str(stage2_dir / "label_encoder.json"))
    log(f"Classes: {label_encoder.n_classes()}  "
        f"(PBS at index {label_encoder.encode('PBS')})")

    manifest_path = cfg["data"]["manifest_path"]
    HVG_PATH = str(Path(manifest_path).parent / "hvg_list.json")
    with open(HVG_PATH) as f:
        gene_names = json.load(f)
    log(f"HVGs: {len(gene_names)}")

    TRAIN_MANIFEST_PATH = str(stage2_dir / "manifest_train.json")
    VAL_MANIFEST_PATH   = str(stage2_dir / "manifest_val.json")

    log("Preloading tube datasets...")
    train_tube_dataset = PseudoTubeDataset(
        TRAIN_MANIFEST_PATH, label_encoder, gene_names=gene_names, preload=True)
    val_tube_dataset = PseudoTubeDataset(
        VAL_MANIFEST_PATH, label_encoder, gene_names=gene_names, preload=True)
    log(f"Train tubes: {len(train_tube_dataset)}")
    log(f"Val tubes:   {len(val_tube_dataset)}")
    log()

    # Save provenance and artifacts
    with open(out_dir / "stage2_dir.txt", "w") as f:
        f.write(str(stage2_dir) + "\n")
    with open(out_dir / "cytokine_groups.json", "w") as f:
        json.dump(groups, f, indent=2)
    label_encoder.save(str(out_dir / "label_encoder.json"))

    # ------------------------------------------------------------------
    # Reconstruct Stage 2 v1 model
    # ------------------------------------------------------------------
    log("=" * 60)
    log("Reconstructing Stage 2 v1 model from checkpoint")
    log("=" * 60)

    embed_dim = cfg["model"]["embedding_dim"]
    attention_hidden = cfg["model"]["attention_hidden_dim"]
    n_classes = label_encoder.n_classes()

    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=None,
        embed_dim=embed_dim,
    )
    mil_model = build_mil_model(
        encoder,
        embed_dim=embed_dim,
        attention_hidden_dim=attention_hidden,
        n_classes=n_classes,
        encoder_frozen=True,
    )
    state = torch.load(str(checkpoint_path), map_location=DEVICE)
    missing, unexpected = mil_model.load_state_dict(state, strict=False)
    log(f"Checkpoint loaded. Missing: {missing}. Unexpected: {unexpected}.")
    log()

    # ------------------------------------------------------------------
    # Build Stage3CAModel
    # ------------------------------------------------------------------
    log("=" * 60)
    log("Building Stage3CAModel (CA only trainable)")
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

    # ------------------------------------------------------------------
    # Stage 3 training loop
    # ------------------------------------------------------------------
    log("=" * 60)
    log("Stage 3 — CA-only training")
    log("=" * 60)

    n_epochs = cfg["training"]["stage3_epochs"]
    log_every = cfg["dynamics"]["log_every_n_epochs"]
    lr = cfg["training"]["lr"]
    momentum = cfg["training"]["momentum"]
    TRAINING_SEED = cfg["dynamics"]["random_seeds"][0]

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, stage3_model.parameters()),
        lr=lr,
        momentum=momentum,
    )
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(TRAINING_SEED)

    log(f"n_epochs:      {n_epochs}")
    log(f"lr:            {lr}")
    log(f"momentum:      {momentum}")
    log(f"log_every:     {log_every}")
    log(f"training_seed: {TRAINING_SEED}")
    log()

    train_entries = train_tube_dataset.get_entries()
    val_entries = val_tube_dataset.get_entries()
    train_tube_traj = _init_tube_trajectories(train_entries)
    val_tube_traj = _init_tube_trajectories(val_entries)
    train_confusion_epochs = defaultdict(list)
    val_confusion_epochs = defaultdict(list)

    queues = build_cytokine_queues(train_entries, label_encoder)

    logged_epochs = []
    ca_weight_norm_trajectory = []
    loss_trajectory = []

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
                train_tube_traj, train_confusion_epochs, label_encoder, DEVICE,
            )
            _log_dynamics_stage3(
                stage3_model, val_tube_dataset, val_entries,
                val_tube_traj, val_confusion_epochs, label_encoder, DEVICE,
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
    # Build records
    # ------------------------------------------------------------------
    train_records = _build_records(train_entries, train_tube_traj)
    val_records = _build_records(val_entries, val_tube_traj)
    log(f"Train records: {len(train_records)}")
    log(f"Val records:   {len(val_records)}")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    torch.save(
        stage3_model.state_dict(),
        str(out_dir / f"stage3_ca_model_{BOOTSTRAP_SEED}.pt"),
    )
    log(f"Model saved: stage3_ca_model_{BOOTSTRAP_SEED}.pt")

    with open(out_dir / "dynamics_stage3.pkl", "wb") as fh:
        pickle.dump({
            "records": train_records,
            "val_records": val_records,
            "logged_epochs": logged_epochs,
            "ca_weight_norm_trajectory": ca_weight_norm_trajectory,
            "loss_trajectory": loss_trajectory,
            "confusion_entropy_trajectory": {
                k: np.array(v) for k, v in train_confusion_epochs.items()
            },
            "val_confusion_entropy_trajectory": {
                k: np.array(v) for k, v in val_confusion_epochs.items()
            },
        }, fh)
    log("Dynamics saved: dynamics_stage3.pkl")

    # ------------------------------------------------------------------
    # Post-training analysis
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Dynamics analysis — Stage 3 CA-only")
    log("=" * 60)

    donor_traj = aggregate_to_donor_level(train_records)
    val_donor_traj = aggregate_to_donor_level(val_records)

    learnability_result = rank_cytokines_by_learnability(donor_traj, exclude=["PBS"])
    val_learnability_result = rank_cytokines_by_learnability(val_donor_traj, exclude=["PBS"])
    ranking = learnability_result["ranking"]
    val_ranking = val_learnability_result["ranking"]
    val_auc_map = {cyt: auc for cyt, auc in val_ranking}

    log()
    log(f"Cytokine learnability ranking — Stage 3 CA-only  |  Bootstrap seed: {BOOTSTRAP_SEED}")
    log(f"Metric: {learnability_result['metric_description']}")
    log()
    hdr = (f"{'Rank':>4}  {'Cytokine':<14}  {'Train AUC':>9}  {'Val AUC':>8}  Group")
    log(hdr)
    log("-" * 55)
    for i, (cyt, auc) in enumerate(ranking, 1):
        group = "SIMPLE" if cyt in SIMPLE_CYTOKINES else "COMPLEX"
        val_auc = val_auc_map.get(cyt, float("nan"))
        log(f"  {i:2d}.  {cyt:<14}  {auc:>9.3f}  {val_auc:>8.3f}  {group}")

    simple_aucs = [auc for cyt, auc in ranking if cyt in SIMPLE_CYTOKINES]
    complex_aucs = [auc for cyt, auc in ranking if cyt in COMPLEX_CYTOKINES]
    log()
    log("Group summary (train AUC):")
    log(f"  SIMPLE  mean={np.mean(simple_aucs):.3f}  median={np.median(simple_aucs):.3f}"
        f"  values={[f'{x:.2f}' for x in sorted(simple_aucs, reverse=True)]}")
    log(f"  COMPLEX mean={np.mean(complex_aucs):.3f}  median={np.median(complex_aucs):.3f}"
        f"  values={[f'{x:.2f}' for x in sorted(complex_aucs, reverse=True)]}")

    # Confusion entropy summary
    log()
    log("-" * 60)
    confusion_result = compute_confusion_entropy_summary(
        {k: np.array(v) for k, v in train_confusion_epochs.items()}, exclude=["PBS"]
    )
    val_confusion_result = compute_confusion_entropy_summary(
        {k: np.array(v) for k, v in val_confusion_epochs.items()}, exclude=["PBS"]
    )
    val_conf_map = {cyt: auc for cyt, auc in val_confusion_result["ranking"]}
    log("Cytokine confusion entropy ranking — Stage 3 CA-only")
    log(f"Metric: {confusion_result['metric_description']}")
    log()
    log(f"{'Cytokine':<20}  {'Train AUC(H_c)':>14}  {'Val AUC(H_c)':>12}  Group")
    log("-" * 64)
    for cyt, auc in confusion_result["ranking"]:
        group = "SIMPLE" if cyt in SIMPLE_CYTOKINES else (
            "COMPLEX" if cyt in COMPLEX_CYTOKINES else "PBS"
        )
        val_auc = val_conf_map.get(cyt, float("nan"))
        log(f"  {cyt:<20}  {auc:>14.3f}  {val_auc:>12.3f}  {group}")

    # ------------------------------------------------------------------
    # Hypothesis test
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Hypothesis test — Stage 3 CA-only (train donors)")
    log("=" * 60)

    auc_map = {cyt: auc for cyt, auc in ranking}
    simple_auc_vals = [auc_map[c] for c in SIMPLE_CYTOKINES if c in auc_map]
    complex_auc_vals = [auc_map[c] for c in COMPLEX_CYTOKINES if c in auc_map]

    if simple_auc_vals and complex_auc_vals:
        u_stat, p_one_sided = mannwhitneyu(
            simple_auc_vals, complex_auc_vals, alternative="greater"
        )
        n1, n2 = len(simple_auc_vals), len(complex_auc_vals)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)

        log(f"  SIMPLE  AUCs: {[f'{x:.3f}' for x in sorted(simple_auc_vals, reverse=True)]}")
        log(f"  COMPLEX AUCs: {[f'{x:.3f}' for x in sorted(complex_auc_vals, reverse=True)]}")
        log()
        log(f"  Mann-Whitney U statistic: {u_stat:.1f}")
        log(f"  One-sided p-value (simple > complex): {p_one_sided:.4f}")
        log(f"  Rank-biserial correlation r = {r_rb:.3f}")
        log()
        alpha = 0.05
        if p_one_sided < alpha:
            log(f"  Result: p < {alpha} -> hypothesis SUPPORTED (simple > complex, train donors)")
        else:
            log(f"  Result: p >= {alpha} -> hypothesis NOT SUPPORTED at alpha={alpha}")
        log("  Note: n=5 per group -> low power. Report effect size alongside p-value.")

        # Validation generalization
        train_order = [c for c, _ in ranking]
        val_order = [c for c, _ in val_ranking]
        val_rank_by_cyt = {c: i for i, c in enumerate(val_order)}
        val_ranks_aligned = [val_rank_by_cyt.get(c, len(val_order))
                             for c in train_order]
        if len(train_order) >= 2:
            rho_gen, pval_gen = spearmanr(range(len(train_order)), val_ranks_aligned)
            log()
            log(f"Train/val rank correlation: Spearman rho = {rho_gen:.3f}  "
                f"(p={pval_gen:.3f})")
            log(f"Stable (rho > 0.7): {rho_gen > 0.7}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Saving figures...")
    log("=" * 60)

    _plot_learning_curves(
        train_records, val_records, logged_epochs,
        SIMPLE_CYTOKINES, COMPLEX_CYTOKINES, BOOTSTRAP_SEED, out_dir,
    )
    log(f"  Saved: learning_curves_stage3_ca_{BOOTSTRAP_SEED}.png")

    _plot_sa_vs_ca_entropy(
        train_records, logged_epochs,
        SIMPLE_CYTOKINES, COMPLEX_CYTOKINES, BOOTSTRAP_SEED, out_dir,
    )
    log(f"  Saved: sa_vs_ca_entropy_stage3_ca_{BOOTSTRAP_SEED}.png")

    _plot_ca_weight_norm(ca_weight_norm_trajectory, BOOTSTRAP_SEED, out_dir)
    log(f"  Saved: ca_weight_norm_stage3_ca_{BOOTSTRAP_SEED}.png")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    log()
    log(f"All results saved to: {out_dir}")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
