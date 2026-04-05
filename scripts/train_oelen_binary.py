"""
Oelen 2022 pathogen experiment — 3 independent binary models.

Trains one SA-only AB-MIL from scratch (encoder included) for each
pathogen condition against the unstimulated control (UT):

    Model 1: UT vs 24hPA
    Model 2: UT vs 24hMTB
    Model 3: UT vs 24hCA

Motivation: the joint 4-class model suffers from softmax competition —
MTB and PA are transcriptionally similar and compete for probability mass,
potentially inflating CA's apparent learnability. Independent binary models
eliminate this confound entirely.

Hyperparameters are deliberately conservative (narrow model, low LR) to
keep dynamics informative over 150 epochs rather than saturating in <10.

Val donors: donor_6.0 and donor_95.0 (held out — observer-only, no gradient
updates). Confusion entropy is NOT computed — meaningless for K=2.

Results saved to:
    results/oelen_binary/run_{timestamp}/

Usage:
    python scripts/train_oelen_binary.py
    python scripts/train_oelen_binary.py --seed 123
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
from torch.utils.data import DataLoader

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset
from cytokine_mil.models.instance_encoder import InstanceEncoder
from cytokine_mil.experiment_setup import (
    build_mil_model,
    build_stage1_manifest,
    make_binary_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder
from cytokine_mil.training.train_mil import train_mil
from cytokine_mil.analysis.dynamics import (
    aggregate_to_donor_level,
    rank_cytokines_by_learnability,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_PATH = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oelen_pseudotubes/manifest.json"
)
OUTPUT_BASE = str(Path(__file__).parent.parent / "results" / "oelen_binary")

VAL_DONORS = ["donor_6.0", "donor_95.0"]

# Each tuple: (target_condition, control_condition)
BINARY_PAIRS = [
    ("24hPA",  "UT"),
    ("24hMTB", "UT"),
    ("24hCA",  "UT"),
]

# Expected difficulty ordering (ground truth for biological validation)
DIFFICULTY_MAP = {
    "24hPA":  "EASY",
    "24hMTB": "HARD",
    "24hCA":  "MED",
    "UT":     "CONTROL",
}

COLOR_MAP = {
    "24hPA":  "steelblue",
    "24hMTB": "tomato",
    "24hCA":  "darkorange",
}

# ---------------------------------------------------------------------------
# Hyperparameters — identical for all 3 models
# ---------------------------------------------------------------------------

EMBED_DIM            = 64       # was 128 in 4-class experiment
HIDDEN_DIMS          = (256, 128)  # was (512, 256) hardcoded — halved at every layer
ATTENTION_HIDDEN_DIM = 32       # was 64

STAGE1_EPOCHS   = 20            # was 50 — weaker encoder = more MIL headroom
STAGE1_LR       = 0.005         # was 0.01
STAGE1_MOMENTUM = 0.9

STAGE2_EPOCHS   = 150           # was 100 — longer window for dynamics to unfold
STAGE2_LR       = 0.0001        # was 0.0002 — halved nominal LR
STAGE2_MOMENTUM = 0.90          # was 0.95 — halved effective-LR multiplier 1/(1-m)

LOG_EVERY = 1
SEED      = 42


# ---------------------------------------------------------------------------
# Per-model training
# ---------------------------------------------------------------------------

def _train_one_binary_model(
    target: str,
    control: str,
    manifest: list,
    gene_names: list,
    out_dir: Path,
    device: torch.device,
    seed: int,
    log,
) -> dict:
    """
    Train a single binary AB-MIL model (Stage 1 encoder + Stage 2 MIL)
    for target vs control from scratch.

    Returns the dynamics dict with keys:
        records, val_records, logged_epochs, condition, control
    """
    log()
    log("=" * 60)
    log(f"Binary model: {target} vs {control}")
    log("=" * 60)

    # ------------------------------------------------------------------
    # Binary manifest + donor split
    # ------------------------------------------------------------------
    bin_manifest, label_enc = make_binary_manifest(manifest, target, control=control)
    train_manifest, val_manifest = split_manifest_by_donor(bin_manifest, VAL_DONORS)

    train_donors = sorted({e["donor"] for e in train_manifest})
    val_donors   = sorted({e["donor"] for e in val_manifest})
    log(f"  Binary entries total: {len(bin_manifest)}")
    log(f"  Train tubes: {len(train_manifest)}  |  Val tubes: {len(val_manifest)}")
    log(f"  Train donors ({len(train_donors)}): {train_donors}")
    log(f"  Val   donors ({len(val_donors)}):   {val_donors}")
    log(f"  Label encoding: {target}→{label_enc.encode(target)}, "
        f"{control}→{label_enc.encode(control)}, n_classes={label_enc.n_classes()}")

    # Save manifests to disk (PseudoTubeDataset constructor takes a path)
    train_m_path = str(out_dir / f"manifest_train_{target}.json")
    val_m_path   = str(out_dir / f"manifest_val_{target}.json")
    with open(train_m_path, "w") as fh:
        json.dump(train_manifest, fh)
    with open(val_m_path, "w") as fh:
        json.dump(val_manifest, fh)

    # Save label encoder (BinaryLabel has no .save() — write JSON manually)
    with open(out_dir / f"label_encoder_{target}.json", "w") as fh:
        json.dump({"positive": label_enc.positive, "negative": label_enc.negative}, fh)

    # ------------------------------------------------------------------
    # Tube datasets (preloaded)
    # ------------------------------------------------------------------
    log("  Preloading tube datasets...")
    train_tube_dataset = PseudoTubeDataset(
        train_m_path, label_enc, gene_names=gene_names, preload=True
    )
    val_tube_dataset = PseudoTubeDataset(
        val_m_path, label_enc, gene_names=gene_names, preload=True
    )
    log(f"  Train tubes loaded: {len(train_tube_dataset)}")
    log(f"  Val   tubes loaded: {len(val_tube_dataset)}")

    # ------------------------------------------------------------------
    # Stage 1 — Encoder pre-training (cell-type classification)
    # ------------------------------------------------------------------
    log()
    log(f"  Stage 1 — Encoder pre-training ({STAGE1_EPOCHS} epochs, "
        f"lr={STAGE1_LR}, momentum={STAGE1_MOMENTUM})")

    stage1_m_path = str(out_dir / f"manifest_stage1_{target}.json")
    build_stage1_manifest(train_manifest, save_path=stage1_m_path)
    cell_dataset = CellDataset(stage1_m_path, gene_names=gene_names, preload=True)
    cell_loader  = DataLoader(
        cell_dataset, batch_size=256, shuffle=True, num_workers=0
    )
    log(f"  Stage 1 cells: {len(cell_dataset)}  |  "
        f"Cell types: {cell_dataset.n_cell_types()}")

    # Build encoder directly (build_encoder wrapper doesn't accept hidden_dims)
    encoder = InstanceEncoder(
        input_dim=len(gene_names),
        embed_dim=EMBED_DIM,
        n_cell_types=cell_dataset.n_cell_types(),
        hidden_dims=HIDDEN_DIMS,
    )
    n_enc_params = sum(p.numel() for p in encoder.parameters())
    log(f"  Encoder params: {n_enc_params:,}  "
        f"(embed_dim={EMBED_DIM}, hidden_dims={HIDDEN_DIMS})")

    encoder = train_encoder(
        encoder,
        cell_loader,
        n_epochs=STAGE1_EPOCHS,
        lr=STAGE1_LR,
        momentum=STAGE1_MOMENTUM,
        device=device,
        verbose=True,
    )
    enc_path = str(out_dir / f"encoder_{target}_stage1.pt")
    torch.save(encoder.state_dict(), enc_path)
    log(f"  Encoder saved: {enc_path}")

    # ------------------------------------------------------------------
    # Stage 2 — Binary MIL (encoder frozen)
    # ------------------------------------------------------------------
    log()
    log(f"  Stage 2 — Binary MIL ({STAGE2_EPOCHS} epochs, "
        f"lr={STAGE2_LR}, momentum={STAGE2_MOMENTUM})")

    mil_model = build_mil_model(
        encoder,
        embed_dim=EMBED_DIM,
        attention_hidden_dim=ATTENTION_HIDDEN_DIM,
        n_classes=label_enc.n_classes(),   # 2
        encoder_frozen=True,
    )
    n_total   = sum(p.numel() for p in mil_model.parameters())
    n_trainable = sum(p.numel() for p in mil_model.parameters() if p.requires_grad)
    log(f"  MIL total params:     {n_total:,}")
    log(f"  MIL trainable params: {n_trainable:,}  "
        f"(attention_hidden_dim={ATTENTION_HIDDEN_DIM})")

    dynamics = train_mil(
        mil_model,
        train_tube_dataset,
        n_epochs=STAGE2_EPOCHS,
        lr=STAGE2_LR,
        momentum=STAGE2_MOMENTUM,
        log_every_n_epochs=LOG_EVERY,
        device=device,
        seed=seed,
        verbose=True,
        val_dataset=val_tube_dataset,
    )

    model_path = str(out_dir / f"model_{target}.pt")
    torch.save(mil_model.state_dict(), model_path)
    log(f"  Model saved: {model_path}")
    log(f"  Train records: {len(dynamics['records'])}")
    log(f"  Val   records: {len(dynamics['val_records'])}")

    # Save dynamics pkl — drop confusion_entropy_trajectory (meaningless for K=2)
    pkl_path = out_dir / f"dynamics_{target}.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump({
            "records":        dynamics["records"],
            "val_records":    dynamics["val_records"],
            "logged_epochs":  dynamics["logged_epochs"],
            "condition":      target,
            "control":        control,
        }, fh)
    log(f"  Dynamics saved: {pkl_path.name}")

    return {
        "records":       dynamics["records"],
        "val_records":   dynamics["val_records"],
        "logged_epochs": dynamics["logged_epochs"],
        "condition":     target,
        "control":       control,
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _donor_mean(records, condition):
    """Return mean-across-donors p_correct_trajectory for one condition."""
    cond_records = [r for r in records if r["cytokine"] == condition]
    if not cond_records:
        return None
    donor_traj = aggregate_to_donor_level(cond_records)
    if condition not in donor_traj or not donor_traj[condition]:
        return None
    return np.mean(list(donor_traj[condition].values()), axis=0)


def _compute_normalized_auc(traj):
    """AUC(p_correct(t) / max(p_correct(t))), normalized by (n-1)."""
    arr = np.array(traj, dtype=float)
    peak = arr.max()
    if peak < 1e-8:
        return 0.0
    normalized = arr / peak
    return float(np.trapz(normalized) / max(len(normalized) - 1, 1))


def _print_learnability_table(all_dynamics, log):
    """Print learnability ranking table for all 3 binary models."""
    log()
    log("=" * 60)
    log("Condition learnability ranking — Binary SA (UT vs condition)")
    log("Metric: AUC of mean p_correct_trajectory across pseudo-tubes, "
        "aggregated to donor level")
    log("        (median across pseudo-tubes per donor, then mean across donors)")
    log()

    rows = []
    for target, dyn in all_dynamics.items():
        train_mean = _donor_mean(dyn["records"],     target)
        val_mean   = _donor_mean(dyn["val_records"], target)
        logged     = dyn["logged_epochs"]

        train_auc = _compute_normalized_auc(train_mean) * 100 if train_mean is not None else float("nan")
        val_auc   = _compute_normalized_auc(val_mean)   * 100 if val_mean   is not None else float("nan")
        rows.append((target, train_auc, val_auc))

    rows.sort(key=lambda x: -x[1])

    header = f"{'Rank':<5}  {'Condition':<10}  {'Train AUC':>10}  {'Val AUC':>9}  {'Difficulty'}"
    sep    = "-" * len(header)
    log(header)
    log(sep)
    for rank, (cond, train_auc, val_auc) in enumerate(rows, 1):
        diff = DIFFICULTY_MAP.get(cond, "")
        log(f"  {rank}.  {cond:<10}  {train_auc:>10.3f}  {val_auc:>9.3f}  {diff}")
    log()
    log("Expected ordering: PA > CA > MTB")
    ranked_conds = [r[0] for r in rows]
    pa_idx  = ranked_conds.index("24hPA")  if "24hPA"  in ranked_conds else None
    ca_idx  = ranked_conds.index("24hCA")  if "24hCA"  in ranked_conds else None
    mtb_idx = ranked_conds.index("24hMTB") if "24hMTB" in ranked_conds else None
    if all(x is not None for x in [pa_idx, ca_idx, mtb_idx]):
        pa_gt_ca  = pa_idx  < ca_idx
        ca_gt_mtb = ca_idx  < mtb_idx
        pa_gt_mtb = pa_idx  < mtb_idx
        log(f"  PA > CA:  {'True ' if pa_gt_ca  else 'False'}  (expected: True)")
        log(f"  CA > MTB: {'True ' if ca_gt_mtb else 'False'}  (expected: True)")
        log(f"  PA > MTB: {'True ' if pa_gt_mtb else 'False'}  (expected: True)")
        n_pass = sum([pa_gt_ca, ca_gt_mtb, pa_gt_mtb])
        log(f"  Ordering checks passed: {n_pass}/3")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_learning_curves(all_dynamics, out_dir, seed, log):
    """
    All 3 target-condition p_correct trajectories on one plot.
    Train = solid, val = dashed. Donor-aggregated.
    UT lines are omitted — each model has a separate UT baseline,
    adding them would clutter without interpretive value.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for target, dyn in all_dynamics.items():
        color        = COLOR_MAP.get(target, "gray")
        difficulty   = DIFFICULTY_MAP.get(target, "")
        logged       = dyn["logged_epochs"]

        train_mean = _donor_mean(dyn["records"],     target)
        val_mean   = _donor_mean(dyn["val_records"], target)

        label_base = f"{target} ({difficulty})"
        if train_mean is not None:
            ax.plot(logged, train_mean, color=color, linestyle="-",
                    linewidth=2, label=f"{label_base} — train")
        if val_mean is not None:
            ax.plot(logged, val_mean, color=color, linestyle="--",
                    linewidth=1.5, alpha=0.85, label=f"{label_base} — val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("P(Y_correct | t)")
    ax.set_title("Binary SA — Oelen 2022: each condition vs UT (independent models)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.3)

    plt.suptitle(
        "Metric: mean p_correct_trajectory(t), aggregated to donor level\n"
        f"(median per donor, mean across donors). "
        f"Val donors (held out, observer-only): {VAL_DONORS}",
        fontsize=8,
    )
    plt.tight_layout()

    fname = f"learning_curves_binary_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


def _plot_scatter(all_dynamics, out_dir, seed, log):
    """
    Scatter: Normalized AUC (x) vs Final P(correct) (y) for all 3 conditions.
    One point per condition, colored and labeled.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for target, dyn in all_dynamics.items():
        color      = COLOR_MAP.get(target, "gray")
        difficulty = DIFFICULTY_MAP.get(target, "")
        train_mean = _donor_mean(dyn["records"], target)
        if train_mean is None:
            continue

        norm_auc = _compute_normalized_auc(train_mean)
        final_p  = float(train_mean[-1])

        ax.scatter(norm_auc, final_p, color=color, s=120, zorder=3,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(
            f"{target}\n({difficulty})",
            (norm_auc, final_p),
            textcoords="offset points", xytext=(8, 4), fontsize=8,
        )

    ax.axvline(0.75, color="gray", ls="--", lw=0.8, alpha=0.6, label="AUC=0.75")
    ax.axhline(0.75, color="gray", ls=":",  lw=0.8, alpha=0.6, label="Final P=0.75")
    ax.set_xlabel("Normalized AUC  [AUC(p_correct / max(p_correct))]")
    ax.set_ylabel("Final P(Y_correct)")
    ax.set_title("Binary learnability — Normalized AUC vs Final P\n(train donors)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.suptitle(
        "Metric: AUC(p_correct(t) / max(p_correct(t))), aggregated to donor level\n"
        "(median across pseudo-tubes per donor, then mean across donors)",
        fontsize=8,
    )
    plt.tight_layout()

    fname = f"scatter_normauc_pmax_binary_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


def _plot_entropy(all_dynamics, out_dir, seed, log):
    """
    1x3 grid: attention entropy trajectory per condition.
    Mean across all train pseudo-tubes; val mean overlaid as dashed.
    """
    targets = [pair[0] for pair in BINARY_PAIRS]
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)

    for ax, target in zip(axes, targets):
        dyn        = all_dynamics.get(target)
        color      = COLOR_MAP.get(target, "gray")
        difficulty = DIFFICULTY_MAP.get(target, "")

        if dyn is None:
            ax.set_title(f"{target} — no data")
            continue

        logged = dyn["logged_epochs"]

        # Mean entropy across all train tubes (not donor-aggregated here)
        train_entropies = [
            r["entropy_trajectory"] for r in dyn["records"]
            if r["cytokine"] == target and r.get("entropy_trajectory")
        ]
        val_entropies = [
            r["entropy_trajectory"] for r in dyn["val_records"]
            if r["cytokine"] == target and r.get("entropy_trajectory")
        ]

        if train_entropies:
            mean_train_ent = np.mean(train_entropies, axis=0)
            ax.plot(logged, mean_train_ent, color=color, lw=2, label="train")

        if val_entropies:
            mean_val_ent = np.mean(val_entropies, axis=0)
            ax.plot(logged, mean_val_ent, color=color, lw=1.5,
                    linestyle="--", alpha=0.8, label="val")

        ax.set_title(f"{target}\n({difficulty})", fontsize=9)
        ax.set_xlabel("Epoch")
        if ax == axes[0]:
            ax.set_ylabel("H(a) [nats]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Binary SA — Attention entropy per condition  |  Oelen 2022\n"
        "Metric: H(a) = -sum_i a_i log(a_i), mean across pseudo-tubes per condition.",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    fname = f"attention_entropy_binary_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oelen 2022 binary experiment: 3 independent binary models vs UT"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Training seed (default: {SEED})")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else SEED

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(OUTPUT_BASE) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    log("Oelen 2022 pathogen experiment — 3 independent binary models")
    log(f"Output directory: {out_dir}")
    log(f"Started:          {timestamp}")
    log(f"Device:           {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    log(f"Training seed:    {seed}")
    log(f"Val donors:       {VAL_DONORS}")
    log()
    log("Hyperparameters (identical for all 3 models):")
    log(f"  embed_dim:            {EMBED_DIM}  (was 128)")
    log(f"  hidden_dims:          {HIDDEN_DIMS}  (was (512, 256))")
    log(f"  attention_hidden_dim: {ATTENTION_HIDDEN_DIM}  (was 64)")
    log(f"  Stage 1 epochs:       {STAGE1_EPOCHS}  (was 50)  lr={STAGE1_LR}")
    log(f"  Stage 2 epochs:       {STAGE2_EPOCHS}  (was 100)  lr={STAGE2_LR}  "
        f"momentum={STAGE2_MOMENTUM}")
    log()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load manifest + gene list
    # ------------------------------------------------------------------
    log("=" * 60)
    log("Loading data")
    log("=" * 60)

    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    hvg_path = str(Path(MANIFEST_PATH).parent / "hvg_list.json")
    with open(hvg_path) as fh:
        gene_names = json.load(fh)

    log(f"Manifest entries: {len(manifest)}")
    log(f"HVGs:             {len(gene_names)}")
    conditions = sorted({e["cytokine"] for e in manifest})
    donors     = sorted({e["donor"] for e in manifest})
    log(f"Conditions: {conditions}")
    log(f"Donors:     {len(donors)} total")
    log()

    # ------------------------------------------------------------------
    # Train all 3 binary models
    # ------------------------------------------------------------------
    all_dynamics = {}
    for target, control in BINARY_PAIRS:
        dyn = _train_one_binary_model(
            target=target,
            control=control,
            manifest=manifest,
            gene_names=gene_names,
            out_dir=out_dir,
            device=device,
            seed=seed,
            log=log,
        )
        all_dynamics[target] = dyn

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    _print_learnability_table(all_dynamics, log)

    log()
    log("=" * 60)
    log("Saving figures...")
    log("=" * 60)

    _plot_learning_curves(all_dynamics, out_dir, seed, log)
    _plot_scatter(all_dynamics, out_dir, seed, log)
    _plot_entropy(all_dynamics, out_dir, seed, log)

    log()
    log(f"All results saved to: {out_dir}")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
