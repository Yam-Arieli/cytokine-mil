"""
Oesinghaus cytokine binary experiment — 17 independent binary models.

Trains one shared Stage 1 encoder (cell-type classification) across all 17
cytokines + PBS, then trains one binary AB-MIL per cytokine (cytokine vs PBS,
n_classes=2) with the shared frozen encoder.

Pre-registered hypothesis (CLAUDE.md Section 18):
    SIMPLE cytokines have higher normalized_AUC than COMPLEX cytokines.
    Tested post-training with a one-sided Mann-Whitney U test (alpha=0.05),
    run exactly once.

LR: 3e-5 fixed from Oelen binary sweep. No re-sweep on Oesinghaus.

Val donors: Donor2, Donor3 (held out — observer-only, no gradient updates).
Confusion entropy is NOT computed — meaningless for K=2.

Results saved to:
    results/oesinghaus_binary/run_{timestamp}/

Usage:
    python scripts/train_oesinghaus_binary.py
    python scripts/train_oesinghaus_binary.py --seed 123
    python scripts/train_oesinghaus_binary.py --output_dir /path/to/dir
"""

import argparse
import copy
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from torch.utils.data import DataLoader

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset
from cytokine_mil.models.instance_encoder import InstanceEncoder
from cytokine_mil.experiment_setup import (
    build_mil_model,
    build_stage1_manifest,
    filter_manifest,
    make_binary_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder
from cytokine_mil.training.train_mil import train_mil
from cytokine_mil.analysis.dynamics import aggregate_to_donor_level


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH      = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
OUTPUT_BASE   = Path(__file__).parent.parent / "results" / "oesinghaus_binary"
VAL_DONORS    = ["Donor2", "Donor3"]
CONTROL       = "PBS"

SIMPLE_POOL  = ["IL-4", "IL-10", "IL-2", "M-CSF", "TNF-alpha",
                "IL-1-beta", "IFN-beta", "IL-7", "G-CSF"]
COMPLEX_POOL = ["IL-12", "IL-32-beta", "OSM", "IL-22",
                "VEGF", "HGF", "TGF-beta1", "IL-6"]
ALL_TARGET_CYTOKINES = SIMPLE_POOL + COMPLEX_POOL

GROUP_MAP = {c: "SIMPLE" for c in SIMPLE_POOL}
GROUP_MAP.update({c: "COMPLEX" for c in COMPLEX_POOL})
COLOR_MAP = {c: "steelblue" for c in SIMPLE_POOL}
COLOR_MAP.update({c: "tomato" for c in COMPLEX_POOL})

EMBED_DIM            = 32
HIDDEN_DIMS          = (128, 64)
ATTENTION_HIDDEN_DIM = 16
STAGE1_EPOCHS        = 20
STAGE1_LR            = 0.005
STAGE1_MOMENTUM      = 0.9
STAGE2_EPOCHS        = 250
STAGE2_LR            = 0.00003   # FIXED from Oelen sweep — no CLI override
STAGE2_MOMENTUM      = 0.90
LOG_EVERY            = 1
SEED                 = 42


# ---------------------------------------------------------------------------
# Pre-registration statement
# ---------------------------------------------------------------------------

def _print_preregistration(log):
    """Print the pre-registration block before any data loading."""
    log("=" * 60)
    log("PRE-REGISTRATION STATEMENT")
    log("Experiment: Oesinghaus cytokine binary learnability test")
    log("Hypothesis (directional, one-sided):")
    log("  SIMPLE cytokines have higher normalized_AUC than COMPLEX cytokines.")
    log("Pre-registered pools (CLAUDE.md Section 18):")
    log("  SIMPLE (n=9): IL-4, IL-10, IL-2, M-CSF, TNF-alpha, IL-1-beta, IFN-beta, IL-7, G-CSF")
    log("  COMPLEX (n=8): IL-12, IL-32-beta, OSM, IL-22, VEGF, HGF, TGF-beta1, IL-6")
    log("Metric: normalized_AUC = AUC(p_correct(t) / max(p_correct(t))),")
    log("        train donors, aggregated to donor level (median per donor, mean across donors)")
    log("Statistical test: one-sided Mann-Whitney U (alpha=0.05), run exactly once.")
    log("LR: 3e-5 fixed from Oelen binary sweep. No re-sweep on Oesinghaus.")
    log("Val donors (observer-only): Donor2, Donor3")
    log("=" * 60)


# ---------------------------------------------------------------------------
# Normalized AUC helper
# ---------------------------------------------------------------------------

def _normalized_auc(dynamics_dict: dict, val: bool = False) -> float:
    """
    AUC(p_correct(t) / max(p_correct(t))), donor-aggregated.
    Median across tubes per donor, mean across donors.
    Returns float in [0, 1] (normalized by n_logged_epochs - 1 via trapz).
    """
    records = dynamics_dict["val_records"] if val else dynamics_dict["records"]
    if not records:
        return 0.0
    donor_traj = aggregate_to_donor_level(records, "p_correct_trajectory")
    # For binary, there are two cytokines: target and control.
    # Use only the target condition (positive class = index 0).
    target = dynamics_dict["condition"]
    if target not in donor_traj:
        return 0.0
    donor_aucs = []
    for traj in donor_traj[target].values():
        traj = np.array(traj)
        mx = traj.max()
        if mx < 1e-8:
            donor_aucs.append(0.0)
        else:
            norm_traj = traj / mx
            n = len(norm_traj)
            auc = np.trapz(norm_traj) / max(n - 1, 1)
            donor_aucs.append(float(auc))
    return float(np.mean(donor_aucs)) if donor_aucs else 0.0


def _final_p_train(dynamics_dict: dict) -> float:
    """Last value of donor-aggregated p_correct_trajectory for the target condition."""
    records = dynamics_dict["records"]
    if not records:
        return 0.0
    donor_traj = aggregate_to_donor_level(records, "p_correct_trajectory")
    target = dynamics_dict["condition"]
    if target not in donor_traj:
        return 0.0
    donor_finals = [traj[-1] for traj in donor_traj[target].values()]
    if not donor_finals:
        return 0.0
    return float(np.mean(donor_finals))


# ---------------------------------------------------------------------------
# Per-cytokine binary model training
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
    embed_dim: int = EMBED_DIM,
    attention_hidden_dim: int = ATTENTION_HIDDEN_DIM,
) -> dict:
    """
    Train a single binary AB-MIL model (Stage 2 only, shared frozen encoder)
    for target vs control.

    Returns the dynamics dict with keys:
        records, val_records, logged_epochs, condition, control
    """
    log(f"\n{'=' * 50}")
    log(f"Training binary model: {target} vs {control}  [{GROUP_MAP[target]}]")
    log("=" * 50)

    # ------------------------------------------------------------------
    # Binary manifest + donor split
    # ------------------------------------------------------------------
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
        json.dump({"positive": label_enc.positive, "negative": label_enc.negative}, fh)

    # ------------------------------------------------------------------
    # Tube datasets (preloaded)
    # ------------------------------------------------------------------
    train_dataset = PseudoTubeDataset(str(train_m_path), label_enc, gene_names=gene_names, preload=True)
    val_dataset   = PseudoTubeDataset(str(val_m_path),   label_enc, gene_names=gene_names, preload=True)

    # ------------------------------------------------------------------
    # Deep copy of shared encoder — each binary model gets its own copy
    # ------------------------------------------------------------------
    encoder = copy.deepcopy(shared_encoder)
    log(f"  Using deep copy of shared Stage 1 encoder")

    # ------------------------------------------------------------------
    # Stage 2 — Binary MIL (encoder frozen)
    # ------------------------------------------------------------------
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

    # Save dynamics pkl — drop confusion_entropy_trajectory (meaningless for K=2)
    pkl_path = out_dir / f"dynamics_{safe_target}.pkl"
    payload = {
        "records":       dynamics["records"],
        "val_records":   dynamics["val_records"],
        "logged_epochs": dynamics["logged_epochs"],
        "condition":     target,
        "control":       control,
    }
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh)

    log(f"  Saved: model_{safe_target}.pt, dynamics_{safe_target}.pkl")

    return payload


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _print_learnability_table(all_dynamics, log):
    """
    Sort ALL_TARGET_CYTOKINES by train normalized_AUC descending and print
    the learnability ranking table.
    """
    log()
    log("=" * 60)
    log("Cytokine learnability ranking — Oesinghaus binary (cytokine vs PBS)")
    log("Metric: AUC(p_correct(t) / max(p_correct(t))), train donors,")
    log("        aggregated to donor level (median per donor, mean across donors)")
    log()

    rows = []
    for target in ALL_TARGET_CYTOKINES:
        dyn = all_dynamics.get(target)
        if dyn is None:
            continue
        train_nauc = _normalized_auc(dyn, val=False)
        val_nauc   = _normalized_auc(dyn, val=True)
        final_p    = _final_p_train(dyn)
        rows.append((target, train_nauc, val_nauc, final_p))

    rows.sort(key=lambda x: -x[1])

    header = (
        f"{'Rank':<5}  {'Cytokine':<16}  {'Group':<8}"
        f"  {'Train nAUC':>10}  {'Val nAUC':>9}  {'Final P (train)':>15}"
    )
    log(header)
    log("-" * len(header))
    for rank, (cytokine, train_nauc, val_nauc, final_p) in enumerate(rows, 1):
        group = GROUP_MAP.get(cytokine, "")
        log(
            f"  {rank:>2}.  {cytokine:<16}  {group:<8}"
            f"  {train_nauc:>10.3f}  {val_nauc:>9.3f}  {final_p:>15.3f}"
        )
    log("=" * 60)


def _run_mannwhitney_test(all_dynamics, log):
    """
    Run the pre-registered one-sided Mann-Whitney U test:
    SIMPLE normalized_AUC > COMPLEX normalized_AUC (train donors).
    Run exactly once.
    """
    simple_aucs  = [_normalized_auc(all_dynamics[c])           for c in SIMPLE_POOL  if c in all_dynamics]
    complex_aucs = [_normalized_auc(all_dynamics[c])           for c in COMPLEX_POOL if c in all_dynamics]
    simple_val_aucs  = [_normalized_auc(all_dynamics[c], val=True) for c in SIMPLE_POOL  if c in all_dynamics]
    complex_val_aucs = [_normalized_auc(all_dynamics[c], val=True) for c in COMPLEX_POOL if c in all_dynamics]

    stat, p = mannwhitneyu(simple_aucs, complex_aucs, alternative="greater")
    n1, n2 = len(simple_aucs), len(complex_aucs)
    r_rb = 1 - (2 * stat) / (n1 * n2)
    supported = "SUPPORTED" if p < 0.05 else "NOT SUPPORTED"

    simple_labels  = ", ".join(
        f"{c}={_normalized_auc(all_dynamics[c]):.3f}" for c in SIMPLE_POOL  if c in all_dynamics
    )
    complex_labels = ", ".join(
        f"{c}={_normalized_auc(all_dynamics[c]):.3f}" for c in COMPLEX_POOL if c in all_dynamics
    )

    log()
    log("=" * 60)
    log("Pre-registered hypothesis test")
    log("Hypothesis: SIMPLE normalized_AUC > COMPLEX normalized_AUC")
    log(f"Test: one-sided Mann-Whitney U  (n_simple={n1}, n_complex={n2})")
    log("Metric: AUC(p_correct(t) / max(p_correct(t))), train donors,")
    log("        aggregated to donor level (median per donor, mean across donors)")
    log("-" * 60)
    log(f"SIMPLE AUCs (train):   {simple_labels}  median={np.median(simple_aucs):.3f}")
    log(f"COMPLEX AUCs (train):  {complex_labels}  median={np.median(complex_aucs):.3f}")
    log("-" * 60)
    log(f"Mann-Whitney U: {stat:.1f}   p-value (one-sided): {p:.4f}")
    log(f"Rank-biserial correlation: {r_rb:.3f}")
    log(f"Result: {supported} at alpha=0.05")
    log("-" * 60)
    log("Val donor check (secondary, NOT pre-registered):")
    log(
        f"SIMPLE val median: {np.median(simple_val_aucs):.3f}  |  "
        f"COMPLEX val median: {np.median(complex_val_aucs):.3f}"
    )
    log("NOTE: This test was run exactly once on the pre-registered pools.")
    log("=" * 60)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _donor_mean_trajectory(records, condition):
    """Return mean-across-donors p_correct_trajectory for one condition."""
    cond_records = [r for r in records if r["cytokine"] == condition]
    if not cond_records:
        return None
    donor_traj = aggregate_to_donor_level(cond_records, "p_correct_trajectory")
    if condition not in donor_traj or not donor_traj[condition]:
        return None
    return np.mean(list(donor_traj[condition].values()), axis=0)


def _donor_mean_entropy(records, condition):
    """Return mean-across-donors entropy_trajectory for one condition."""
    cond_records = [r for r in records if r["cytokine"] == condition]
    if not cond_records:
        return None
    donor_traj = aggregate_to_donor_level(cond_records, "entropy_trajectory")
    if condition not in donor_traj or not donor_traj[condition]:
        return None
    return np.mean(list(donor_traj[condition].values()), axis=0)


def _plot_learning_curves(all_dynamics, out_dir, seed, log):
    """
    All 17 target-condition p_correct trajectories on one axis.
    Train = solid, val = dashed. Donor-aggregated (mean across donors).
    Legend uses group proxy patches instead of per-cytokine entries.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for target in ALL_TARGET_CYTOKINES:
        dyn = all_dynamics.get(target)
        if dyn is None:
            continue
        color  = COLOR_MAP[target]
        logged = dyn["logged_epochs"]

        train_mean = _donor_mean_trajectory(dyn["records"],     target)
        val_mean   = _donor_mean_trajectory(dyn["val_records"], target)

        if train_mean is not None:
            ax.plot(
                logged, train_mean,
                color=color, linestyle="-", linewidth=2.0,
                label=f"{target} [{GROUP_MAP[target]}] train",
            )
        if val_mean is not None:
            ax.plot(
                logged, val_mean,
                color=color, linestyle="--", linewidth=1.2, alpha=0.6,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("P(Y_correct)")
    ax.set_title("Oesinghaus binary — cytokine vs PBS (shared encoder)")
    ax.set_ylim(0, 1.05)

    # Group proxy patches
    simple_patch  = mpatches.Patch(color="steelblue", label="SIMPLE")
    complex_patch = mpatches.Patch(color="tomato",    label="COMPLEX")
    # Train/val line proxies
    train_line = mlines.Line2D([], [], color="gray", linestyle="-",  linewidth=2.0, label="train (solid)")
    val_line   = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.2, alpha=0.6, label="val (dashed)")

    legend1 = ax.legend(
        handles=[simple_patch, complex_patch],
        loc="upper right",
        fontsize=9,
        title="Group",
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=[train_line, val_line],
        loc="lower right",
        fontsize=9,
        title="Split",
    )

    plt.suptitle(
        "Pre-registered hypothesis: SIMPLE AUC > COMPLEX AUC",
        fontsize=10,
    )
    plt.tight_layout()

    fname = f"learning_curves_binary_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")


def _plot_scatter(all_dynamics, out_dir, seed, log):
    """
    Scatter: Normalized AUC (x) vs Final P(correct) train (y) per cytokine.
    Group centroids overlaid. Reference lines at 0.75.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    group_xs = defaultdict(list)
    group_ys = defaultdict(list)

    for target in ALL_TARGET_CYTOKINES:
        dyn = all_dynamics.get(target)
        if dyn is None:
            continue
        color    = COLOR_MAP[target]
        group    = GROUP_MAP[target]
        train_mean = _donor_mean_trajectory(dyn["records"], target)
        if train_mean is None:
            continue

        x = _normalized_auc(dyn, val=False)
        y = float(train_mean[-1])

        ax.scatter(x, y, color=color, s=80, zorder=3)
        ax.annotate(target, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)

        group_xs[group].append(x)
        group_ys[group].append(y)

    # Group centroids
    for group, color in [("SIMPLE", "steelblue"), ("COMPLEX", "tomato")]:
        if group_xs[group]:
            cx = np.mean(group_xs[group])
            cy = np.mean(group_ys[group])
            ax.scatter(cx, cy, color=color, s=300, alpha=0.3, zorder=2)

    # Reference lines
    ax.axvline(0.75, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.75, color="gray", linestyle=":",  alpha=0.5)

    ax.set_xlabel("Normalized AUC (train)")
    ax.set_ylabel("Final P(correct) (train)")
    ax.set_title("Oesinghaus binary: normalized AUC vs final P(correct)")

    simple_patch  = mpatches.Patch(color="steelblue", label="SIMPLE")
    complex_patch = mpatches.Patch(color="tomato",    label="COMPLEX")
    ax.legend(handles=[simple_patch, complex_patch], fontsize=9)

    plt.tight_layout()

    fname = f"scatter_normauc_pmax_binary_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")


def _plot_entropy_grid(all_dynamics, out_dir, seed, log):
    """
    2x9 grid of attention entropy trajectories.
    Row 0: SIMPLE_POOL (9 cytokines).
    Row 1: COMPLEX_POOL (8 cytokines) + 1 blank panel.
    Train = solid, val = dashed.
    """
    n_cols = 9
    fig, axes = plt.subplots(2, n_cols, figsize=(26, 8), sharey=True)

    rows_targets = [SIMPLE_POOL, COMPLEX_POOL]

    for row_idx, pool in enumerate(rows_targets):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]

            if col_idx >= len(pool):
                # Blank panel for the 9th slot in the COMPLEX row
                ax.set_visible(False)
                continue

            target = pool[col_idx]
            dyn    = all_dynamics.get(target)
            color  = COLOR_MAP[target]

            if dyn is None:
                ax.set_title(target, fontsize=9)
                continue

            logged     = dyn["logged_epochs"]
            train_mean = _donor_mean_entropy(dyn["records"],     target)
            val_mean   = _donor_mean_entropy(dyn["val_records"], target)

            if train_mean is not None:
                ax.plot(logged, train_mean, color=color, linewidth=1.8, linestyle="-")
            if val_mean is not None:
                ax.plot(logged, val_mean,   color=color, linewidth=1.2, linestyle="--", alpha=0.7)

            ax.set_title(target, fontsize=9)

            # Only show tick labels on leftmost panels
            if col_idx != 0:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            else:
                ax.set_ylabel("H(a) [nats]")

    # Row labels
    # Compute vertical midpoints for each row
    n_rows = 2
    row_height = 1.0 / n_rows
    # Row 0 is at the top in figure coordinates (matplotlib rows go top to bottom)
    simple_y  = 1.0 - row_height * 0.5
    complex_y = 1.0 - row_height * 1.5
    fig.text(0.01, simple_y,  "SIMPLE",  color="steelblue", fontsize=12,
             va="center", ha="left", fontweight="bold")
    fig.text(0.01, complex_y, "COMPLEX", color="tomato",    fontsize=12,
             va="center", ha="left", fontweight="bold")

    fig.suptitle(
        "Attention entropy H(a) = -\u03a3 a\u1d62 log(a\u1d62)  |  "
        "Oesinghaus binary  |  SIMPLE (steelblue) / COMPLEX (tomato)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    fname = f"entropy_grid_binary_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oesinghaus cytokine binary experiment: 17 independent binary models vs PBS"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Training seed (default: {SEED})")
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM,
                        help=f"Encoder embedding dimension (default: {EMBED_DIM})")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=list(HIDDEN_DIMS),
                        help=f"Encoder hidden layer dims (default: {list(HIDDEN_DIMS)})")
    parser.add_argument("--attention_hidden_dim", type=int, default=ATTENTION_HIDDEN_DIM,
                        help=f"Attention hidden dimension (default: {ATTENTION_HIDDEN_DIM})")
    args = parser.parse_args()

    seed              = args.seed if args.seed is not None else SEED
    embed_dim         = args.embed_dim
    hidden_dims       = tuple(args.hidden_dims)
    attention_hidden_dim = args.attention_hidden_dim

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
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

    # Print pre-registration first — before any data loading
    _print_preregistration(log)

    # ------------------------------------------------------------------
    # Log hyperparameters
    # ------------------------------------------------------------------
    log()
    log(f"Output directory: {out_dir}")
    log(f"Started:          {timestamp}")
    log(f"Training seed:    {seed}")
    log(f"Val donors:       {VAL_DONORS}")
    log(f"Control:          {CONTROL}")
    log()
    log("Hyperparameters:")
    log(f"  embed_dim:            {embed_dim}")
    log(f"  hidden_dims:          {hidden_dims}")
    log(f"  attention_hidden_dim: {attention_hidden_dim}")
    log(f"  Stage 1 epochs:       {STAGE1_EPOCHS}  lr={STAGE1_LR}  momentum={STAGE1_MOMENTUM}")
    log(f"  Stage 2 epochs:       {STAGE2_EPOCHS}  lr={STAGE2_LR}  momentum={STAGE2_MOMENTUM}")
    log()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load manifest + HVG list
    # ------------------------------------------------------------------
    log()
    log("Loading manifest and HVG list...")
    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    with open(HVG_PATH) as fh:
        gene_names = json.load(fh)

    log(f"Manifest entries: {len(manifest)}")
    log(f"HVGs:             {len(gene_names)}")

    # ------------------------------------------------------------------
    # Verify all 17 cytokines present
    # ------------------------------------------------------------------
    manifest_cytokines = {e["cytokine"] for e in manifest}
    missing = [c for c in ALL_TARGET_CYTOKINES if c not in manifest_cytokines]
    if missing:
        for c in missing:
            log(f"WARNING: cytokine not found in manifest: {c}")
        raise ValueError(f"Missing cytokines: {missing}")

    # ------------------------------------------------------------------
    # Filter manifest to the 17 target cytokines + PBS
    # ------------------------------------------------------------------
    filtered_manifest = filter_manifest(manifest, ALL_TARGET_CYTOKINES, include_pbs=True)
    log(f"Filtered manifest size: {len(filtered_manifest)} entries "
        f"({len(ALL_TARGET_CYTOKINES)} cytokines + PBS)")

    # ------------------------------------------------------------------
    # SHARED STAGE 1 — train one encoder across all 17 cytokines + PBS
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("SHARED STAGE 1 — encoder pre-training (cell-type classification)")
    log("=" * 60)

    stage1_path = out_dir / "manifest_stage1_shared.json"
    stage1_manifest = build_stage1_manifest(filtered_manifest, save_path=str(stage1_path))
    log(f"Stage 1 manifest: {len(stage1_manifest)} entries saved to {stage1_path.name}")

    cell_dataset = CellDataset(str(stage1_path), gene_names=gene_names, preload=True)
    cell_loader  = DataLoader(cell_dataset, batch_size=256, shuffle=True, num_workers=0)
    log(f"Stage 1 cells: {len(cell_dataset)}  |  Cell types: {cell_dataset.n_cell_types()}")

    encoder = InstanceEncoder(
        input_dim=len(gene_names),
        embed_dim=embed_dim,
        n_cell_types=cell_dataset.n_cell_types(),
        hidden_dims=hidden_dims,
    )
    log(f"Stage 1: training shared encoder, n_cell_types={cell_dataset.n_cell_types()}")

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

    # ------------------------------------------------------------------
    # LOOP — 17 binary models, one per cytokine
    # ------------------------------------------------------------------
    all_dynamics = {}
    for target in ALL_TARGET_CYTOKINES:
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
            attention_hidden_dim=attention_hidden_dim,
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    _print_learnability_table(all_dynamics, log)
    _run_mannwhitney_test(all_dynamics, log)

    log()
    log("=" * 60)
    log("Saving figures...")
    log("=" * 60)

    _plot_learning_curves(all_dynamics, out_dir, seed, log)
    _plot_scatter(all_dynamics, out_dir, seed, log)
    _plot_entropy_grid(all_dynamics, out_dir, seed, log)

    log()
    log(f"Done. Results in: {out_dir}")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
