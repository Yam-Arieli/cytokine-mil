"""
Two-layer attention bootstrap experiment — cluster job script.

Same cytokine sampling and training protocol as run_bootstrap.py, but uses
CytokineABMIL_V2 (TwoLayerAttentionModule: SA + CA) instead of the standard
single-layer CytokineABMIL.

Additional outputs (unique to this script):
  - sa_vs_ca_entropy_2al_bootstrap_{seed}.png
  - attention_overlap_2al_bootstrap_{seed}.png

Results are saved to:
    results/2al_bootstrap_seed{BOOTSTRAP_SEED}_{timestamp}/

Usage:
    python scripts/run_2al_bootstrap.py --bootstrap_seed 42
    python scripts/run_2al_bootstrap.py --bootstrap_seed 123 --n_sample 5
"""

import argparse
import json
import os
import pickle
import random
import sys
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for cluster
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from torch.utils.data import DataLoader

from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.data.dataset import PseudoTubeDataset, CellDataset
from cytokine_mil.models.two_layer_attention import TwoLayerAttentionModule
from cytokine_mil.models.cytokine_abmil_v2 import CytokineABMIL_V2
from cytokine_mil.training.train_encoder import train_encoder
from cytokine_mil.training.train_mil import train_mil
from cytokine_mil.experiment_setup import (
    build_stage1_manifest,
    filter_manifest,
    split_manifest_by_donor,
    build_encoder,
)
from cytokine_mil.analysis.dynamics import (
    aggregate_to_donor_level,
    rank_cytokines_by_learnability,
    compute_cytokine_entropy_summary,
    compute_confusion_entropy_summary,
)
from cytokine_mil.analysis.validation import check_functional_groupings

# ---------------------------------------------------------------------------
# Cytokine pools (identical to run_bootstrap.py)
# ---------------------------------------------------------------------------
SIMPLE_POOL = [
    "IL-4",
    "IL-10",
    "IL-2",
    "M-CSF",
    "TNF-alpha",
    "IL-1-beta",
    "IFN-beta",
    "IL-7",
    "G-CSF",
]

COMPLEX_POOL = [
    "IL-12",
    "IL-32-beta",
    "OSM",
    "IL-22",
    "VEGF",
    "HGF",
    "TGF-beta1",
    "IL-6",
]

VAL_DONORS = ["Donor2", "Donor3"]

FEMALE_DONORS = {f"Donor{i}" for i in [1, 3, 4, 7, 8, 12]}
MALE_DONORS   = {f"Donor{i}" for i in [2, 5, 6, 9, 10, 11]}

# ---------------------------------------------------------------------------
# Extra metrics helpers (identical to run_bootstrap.py)
# ---------------------------------------------------------------------------

def _compute_normalized_auc(traj):
    arr = np.array(traj, dtype=float)
    max_val = arr.max()
    if max_val < 1e-10:
        return 0.0
    norm = arr / max_val
    return float(np.trapezoid(norm) / max(len(norm) - 1, 1))


def _compute_pmax(traj):
    return float(np.max(traj))


def _aggregate_extra_metrics(records, exclude=("PBS",)):
    exclude_set = set(exclude)
    by_cyt_donor = {}
    for r in records:
        cyt = r["cytokine"]
        if cyt in exclude_set:
            continue
        donor = r["donor"]
        by_cyt_donor.setdefault(cyt, {}).setdefault(donor, []).append(r)

    result = {}
    for cyt, by_donor in by_cyt_donor.items():
        donor_norm_aucs, donor_pmaxes = [], []
        for recs in by_donor.values():
            donor_norm_aucs.append(float(np.median(
                [_compute_normalized_auc(r["p_correct_trajectory"]) for r in recs]
            )))
            donor_pmaxes.append(float(np.median(
                [_compute_pmax(r["p_correct_trajectory"]) for r in recs]
            )))
        result[cyt] = {
            "norm_auc": float(np.mean(donor_norm_aucs)),
            "pmax":     float(np.mean(donor_pmaxes)),
        }
    return result


# ---------------------------------------------------------------------------
# Plotting helpers — shared with run_bootstrap.py
# ---------------------------------------------------------------------------

def _plot_learning_curves(dynamics, simple_cyts, complex_cyts, bootstrap_seed, out_dir):
    donor_traj     = aggregate_to_donor_level(dynamics["records"])
    val_donor_traj = aggregate_to_donor_level(dynamics["val_records"])
    epochs = dynamics["logged_epochs"]
    tab10  = plt.cm.tab10.colors

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
                ax.plot(epochs, train_mean, color=color, linestyle="-",
                        alpha=0.9, label=f"{cyt} (train)")
            if cyt in val_donor_traj:
                val_mean = np.mean(list(val_donor_traj[cyt].values()), axis=0)
                ax.plot(epochs, val_mean, color=color, linestyle=":",
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
        f"Stage 2 learning curves — 2-layer attention  |  Bootstrap seed: {bootstrap_seed}\n"
        "Metric: mean p_correct_trajectory(t), aggregated to donor level "
        "(median per donor, mean across donors)",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"learning_curves_2al_bootstrap_{bootstrap_seed}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_donor_curves(dynamics, subset_cyts, simple_cyts, complex_cyts,
                           bootstrap_seed, out_dir):
    donor_traj     = aggregate_to_donor_level(dynamics["records"])
    val_donor_traj = aggregate_to_donor_level(dynamics["val_records"])
    epochs = dynamics["logged_epochs"]

    female_sorted = sorted(FEMALE_DONORS, key=lambda d: int(d.replace("Donor", "")))
    male_sorted   = sorted(MALE_DONORS,   key=lambda d: int(d.replace("Donor", "")))
    female_palette = plt.cm.Reds(np.linspace(0.4, 0.9, len(female_sorted)))
    male_palette   = plt.cm.Blues(np.linspace(0.4, 0.9, len(male_sorted)))
    donor_color = (
        {d: c for d, c in zip(female_sorted, female_palette)}
        | {d: c for d, c in zip(male_sorted, male_palette)}
    )

    def sex_label(donor):
        return "F" if donor in FEMALE_DONORS else "M"

    for cyt in subset_cyts:
        group = "SIMPLE" if cyt in simple_cyts else "COMPLEX"
        fig, ax = plt.subplots(figsize=(8, 4))
        for donor, traj in sorted(
            donor_traj.get(cyt, {}).items(),
            key=lambda x: int(x[0].replace("Donor", ""))
        ):
            ax.plot(epochs, traj,
                    color=donor_color.get(donor, "gray"),
                    linestyle="-", linewidth=1.5, alpha=0.75,
                    label=f"{donor} ({sex_label(donor)})")
        for donor, traj in sorted(
            val_donor_traj.get(cyt, {}).items(),
            key=lambda x: int(x[0].replace("Donor", ""))
        ):
            ax.plot(epochs, traj,
                    color=donor_color.get(donor, "gray"),
                    linestyle=":", linewidth=2.5, alpha=1.0,
                    label=f"{donor} ({sex_label(donor)}, val)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("P(Y_correct | t)")
        ax.set_title(f"{cyt}  [{group}]")
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        fig.suptitle(
            "Per-donor learning trajectory — Stage 2 (2-layer attention)"
            "  |  Color: sex (red=female, blue=male)  |  Dotted=val donor\n"
            "Metric: median p_correct_trajectory across pseudo-tubes per donor, per epoch",
            fontsize=8, y=1.02,
        )
        plt.tight_layout()
        safe_name = cyt.replace("/", "_").replace(" ", "_")
        fig.savefig(
            out_dir / f"per_donor_curves_{safe_name}_2al_bootstrap_{bootstrap_seed}.png",
            dpi=120, bbox_inches="tight",
        )
        plt.close(fig)


def _plot_scatter(extra_train, extra_val, subset_cyts, simple_cyts, complex_cyts,
                  bootstrap_seed, out_dir):
    group_color  = {"SIMPLE": "steelblue", "COMPLEX": "tomato"}
    group_marker = {"SIMPLE": "o",         "COMPLEX": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (split_label, extra_dict) in zip(axes, [
        ("Train (10 donors)", extra_train),
        ("Val (D2, D3)",      extra_val),
    ]):
        for cyt in subset_cyts:
            if cyt not in extra_dict:
                continue
            m     = extra_dict[cyt]
            group = "SIMPLE" if cyt in simple_cyts else "COMPLEX"
            ax.scatter(m["norm_auc"], m["pmax"],
                       color=group_color[group], marker=group_marker[group],
                       s=90, zorder=3)
            ax.annotate(cyt, (m["norm_auc"], m["pmax"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax.set_xlabel(
            "Normalized trajectory AUC\n"
            "AUC(p_correct(t) / max(p_correct(t))), trapz / (n-1)\n"
            "aggregated to donor level (median per donor, mean across donors)",
            fontsize=8,
        )
        ax.set_ylabel(
            "P_max  max(p_correct_trajectory)\n"
            "aggregated to donor level (median per donor, mean across donors)",
            fontsize=8,
        )
        ax.set_title(f"{split_label}\nShape (x-axis) vs Ceiling (y-axis)", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    legend_handles = [
        plt.scatter([], [], color=group_color["SIMPLE"],  marker=group_marker["SIMPLE"],
                    s=60, label="SIMPLE (direct PBMC response)"),
        plt.scatter([], [], color=group_color["COMPLEX"], marker=group_marker["COMPLEX"],
                    s=60, label="COMPLEX (indirect / non-PBMC)"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="lower right")
    plt.suptitle(
        f"Stage 2 — Normalized AUC vs P_max per cytokine  |  2-layer attention  |  Bootstrap seed: {bootstrap_seed}\n"
        "Metric x: AUC(p_correct(t) / max(p_correct(t))), trapz / (n-1), aggregated to donor level\n"
        "Metric y: max(p_correct_trajectory), aggregated to donor level\n"
        "Left = train donors (10)  |  Right = val donors (D2, D3)",
        fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"scatter_normauc_pmax_2al_bootstrap_{bootstrap_seed}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting helpers — v2-specific (SA vs CA)
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


def _compute_attention_overlap(records):
    """
    For each cytokine: overlap(t) = mean_tubes sum_i min(a_SA_i(t), a_CA_i(t)).

    Recovers a_SA and a_CA from stored confidence trajectories by dividing by
    p_correct(t). Epochs where p_correct == 0 are skipped.

    Returns:
        overlap_per_cytokine : {cytokine -> float}   mean overlap across all epochs/tubes.
        overlap_trajectory   : {cytokine -> ndarray} mean overlap per epoch.
    """
    raw = defaultdict(list)
    for rec in records:
        csa = rec.get("confidence_trajectory_sa")
        cca = rec.get("confidence_trajectory_ca")
        if csa is None or cca is None:
            continue
        p_traj = np.array(rec["p_correct_trajectory"])
        safe_p = np.where(p_traj > 1e-10, p_traj, np.nan)
        a_sa = np.array(csa) / safe_p[np.newaxis, :]
        a_ca = np.array(cca) / safe_p[np.newaxis, :]
        overlap_traj = np.nansum(np.minimum(a_sa, a_ca), axis=0)
        raw[rec["cytokine"]].append(overlap_traj)

    overlap_per_cytokine = {}
    overlap_trajectory   = {}
    for cytokine, trajs in raw.items():
        stacked = np.stack(trajs, axis=0)
        mean_traj = np.nanmean(stacked, axis=0)
        overlap_trajectory[cytokine]   = mean_traj
        overlap_per_cytokine[cytokine] = float(np.nanmean(mean_traj))
    return overlap_per_cytokine, overlap_trajectory


def _plot_loss_components(dynamics, bootstrap_seed, kl_lambda, aux_loss_weight, out_dir):
    """Two-panel figure: classification loss components and KL regularization loss per epoch."""
    lc = dynamics.get("loss_components", {})
    if not lc or not lc.get("total"):
        return
    epochs_all = list(range(1, len(lc["total"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.plot(epochs_all, lc["total"], label="total", color="black", lw=2)
    if lc.get("main"):
        ax.plot(epochs_all, lc["main"], label="main (combined)", color="steelblue", ls="--")
    if lc.get("sa_aux"):
        ax.plot(epochs_all, lc["sa_aux"], label="SA aux", color="darkorange", ls="--")
    if lc.get("ca_aux"):
        ax.plot(epochs_all, lc["ca_aux"], label="CA aux", color="seagreen", ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Classification Loss Components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if lc.get("kl"):
        ax.plot(epochs_all, lc["kl"], label="KL(a_CA || a_SA)", color="crimson", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL divergence (batchmean / n_cytokines)")
    ax.set_title("KL Regularization Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Loss components — bootstrap seed {bootstrap_seed}\n"
        f"kl_lambda={kl_lambda}  aux_loss_weight={aux_loss_weight}",
        fontsize=10,
    )
    plt.tight_layout()
    fname = out_dir / f"loss_components_2al_bootstrap_{bootstrap_seed}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_entropy_gap(dynamics, simple_cyts, complex_cyts, bootstrap_seed, out_dir, last_n=5):
    """Bar chart: H(a_CA) - H(a_SA) at convergence per cytokine."""
    records = dynamics["records"]
    all_cyts = simple_cyts + complex_cyts
    colors = ["steelblue" if c in simple_cyts else "tomato" for c in all_cyts]

    gaps_mean = []
    gaps_sem = []
    for cytokine in all_cyts:
        gaps = []
        for rec in records:
            if rec["cytokine"] != cytokine:
                continue
            sa_traj = rec.get("entropy_trajectory") or []
            ca_traj = rec.get("entropy_trajectory_ca") or []
            if not sa_traj or not ca_traj:
                continue
            sa_end = float(np.mean(sa_traj[-last_n:]))
            ca_end = float(np.mean(ca_traj[-last_n:]))
            gaps.append(ca_end - sa_end)
        if gaps:
            gaps_mean.append(float(np.mean(gaps)))
            gaps_sem.append(float(np.std(gaps) / max(len(gaps) ** 0.5, 1)))
        else:
            gaps_mean.append(0.0)
            gaps_sem.append(0.0)

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(all_cyts))
    ax.bar(x, gaps_mean, yerr=gaps_sem, color=colors, capsize=4, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(all_cyts, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean H(a_CA) - H(a_SA) at convergence (nats)")
    ax.set_title(
        f"SA vs CA entropy gap at convergence — bootstrap seed {bootstrap_seed}\n"
        "Positive = CA more diffuse (cascade signal); blue = SIMPLE, red = COMPLEX"
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fname = out_dir / f"entropy_gap_2al_bootstrap_{bootstrap_seed}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sa_vs_ca_entropy(dynamics, simple_cyts, complex_cyts, bootstrap_seed, out_dir):
    """2×5 grid: per cytokine, SA (blue) vs CA (orange) attention entropy over training."""
    epochs    = dynamics["logged_epochs"]
    records   = dynamics["records"]
    all_cyts  = simple_cyts + complex_cyts
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
            ax.plot(epochs, mean_sa, color="steelblue",  linewidth=2, label="SA")
            ax.plot(epochs, mean_ca, color="darkorange", linewidth=2, linestyle="--", label="CA")
        ax.set_title(f"{cyt}\n({group})", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("H(a) [nats]")
        ax.legend(fontsize=7)

    fig.suptitle(
        f"SA vs CA attention entropy — 2-layer attention  |  Bootstrap seed: {bootstrap_seed}\n"
        "Metric: H(a) = -sum_i a_i log(a_i) for SA layer (blue) and CA layer (orange).\n"
        "Mean across pseudo-tubes per cytokine.",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"sa_vs_ca_entropy_2al_bootstrap_{bootstrap_seed}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_attention_overlap(dynamics, simple_cyts, complex_cyts, bootstrap_seed, out_dir):
    """2×5 grid: SA vs CA attention overlap trajectory over training per cytokine."""
    epochs   = dynamics["logged_epochs"]
    records  = dynamics["records"]
    all_cyts = simple_cyts + complex_cyts
    group_map = {c: "SIMPLE" for c in simple_cyts}
    group_map.update({c: "COMPLEX" for c in complex_cyts})

    _overlap_scores, overlap_trajs = _compute_attention_overlap(records)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    for ax, cyt in zip(axes, all_cyts):
        group = group_map.get(cyt, "")
        color = "steelblue" if group == "SIMPLE" else "tomato"
        traj  = overlap_trajs.get(cyt)
        if traj is not None and len(traj) == len(epochs):
            ax.plot(epochs, traj, color=color, linewidth=2)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"{cyt}\n({group})", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("overlap score")

    fig.suptitle(
        f"SA vs CA attention overlap — 2-layer attention  |  Bootstrap seed: {bootstrap_seed}\n"
        "Metric: mean_tubes sum_i min(a_SA_i(t), a_CA_i(t)).  1=identical, 0=fully specialized.",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"attention_overlap_2al_bootstrap_{bootstrap_seed}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-layer attention bootstrap cytokine MIL experiment"
    )
    parser.add_argument("--bootstrap_seed", type=int, default=42,
                        help="Seed for cytokine pool sampling (default: 42)")
    parser.add_argument("--n_sample", type=int, default=5,
                        help="Number of cytokines per group (default: 5)")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent.parent / "configs" / "default.yaml"),
                        help="Path to YAML config")
    args = parser.parse_args()

    BOOTSTRAP_SEED     = args.bootstrap_seed
    N_SAMPLE_PER_GROUP = args.n_sample

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    out_dir = Path(__file__).parent.parent / "results" / \
              f"2al_bootstrap_seed{BOOTSTRAP_SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    log(f"Two-layer attention bootstrap experiment — seed={BOOTSTRAP_SEED}  n_sample={N_SAMPLE_PER_GROUP}")
    log(f"Output directory: {out_dir}")
    log(f"Started: {timestamp}")
    log()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(f"Device:         {DEVICE}")
    log(f"Bootstrap seed: {BOOTSTRAP_SEED}  (controls cytokine sampling and training seed)")
    log()

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    MANIFEST_PATH = cfg["data"]["manifest_path"]
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    HVG_PATH = str(Path(MANIFEST_PATH).parent / "hvg_list.json")
    with open(HVG_PATH) as f:
        gene_names = json.load(f)

    log(f"Full manifest entries: {len(manifest)}")
    log(f"HVGs: {len(gene_names)}")

    # Verify pool names
    manifest_cytokines = {e["cytokine"] for e in manifest}
    missing_simple  = [c for c in SIMPLE_POOL  if c not in manifest_cytokines]
    missing_complex = [c for c in COMPLEX_POOL if c not in manifest_cytokines]
    if missing_simple or missing_complex:
        log("WARNING — some pool cytokines not found in manifest:")
        if missing_simple:  log(f"  Simple pool:  {missing_simple}")
        if missing_complex: log(f"  Complex pool: {missing_complex}")
        sys.exit(1)

    # Seeded sampling
    _rng = random.Random(BOOTSTRAP_SEED)
    SIMPLE_CYTOKINES  = sorted(_rng.sample(SIMPLE_POOL,  N_SAMPLE_PER_GROUP))
    COMPLEX_CYTOKINES = sorted(_rng.sample(COMPLEX_POOL, N_SAMPLE_PER_GROUP))
    SUBSET_CYTOKINES  = SIMPLE_CYTOKINES + COMPLEX_CYTOKINES

    log(f"Sampled simple  ({len(SIMPLE_CYTOKINES)}): {SIMPLE_CYTOKINES}")
    log(f"Sampled complex ({len(COMPLEX_CYTOKINES)}): {COMPLEX_CYTOKINES}")
    log(f"Full subset     ({len(SUBSET_CYTOKINES)}): {SUBSET_CYTOKINES}")
    log()

    with open(out_dir / "cytokine_groups.json", "w") as f:
        json.dump({
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_sample_per_group": N_SAMPLE_PER_GROUP,
            "simple_cytokines": SIMPLE_CYTOKINES,
            "complex_cytokines": COMPLEX_CYTOKINES,
            "simple_pool": SIMPLE_POOL,
            "complex_pool": COMPLEX_POOL,
        }, f, indent=2)

    # Filter manifest
    subset_manifest = filter_manifest(manifest, cytokines=SUBSET_CYTOKINES, include_pbs=True)
    log(f"Subset manifest entries: {len(subset_manifest)}")

    # Label encoder
    label_encoder = CytokineLabel().fit(subset_manifest)
    label_encoder.save(str(out_dir / "label_encoder.json"))
    log(f"Classes: {label_encoder.n_classes()}  (PBS at index {label_encoder.encode('PBS')})")

    # Donor-level train/val split
    train_manifest, val_manifest = split_manifest_by_donor(subset_manifest, val_donors=VAL_DONORS)
    log(f"Train donors: {sorted({e['donor'] for e in train_manifest})}  ({len(train_manifest)} tubes)")
    log(f"Val donors:   {sorted({e['donor'] for e in val_manifest})}  ({len(val_manifest)} tubes)")

    TRAIN_MANIFEST_PATH = str(out_dir / "manifest_train.json")
    VAL_MANIFEST_PATH   = str(out_dir / "manifest_val.json")
    with open(TRAIN_MANIFEST_PATH, "w") as f:
        json.dump(train_manifest, f)
    with open(VAL_MANIFEST_PATH, "w") as f:
        json.dump(val_manifest, f)

    log("Preloading tube datasets...")
    train_tube_dataset = PseudoTubeDataset(
        TRAIN_MANIFEST_PATH, label_encoder, gene_names=gene_names, preload=True)
    val_tube_dataset = PseudoTubeDataset(
        VAL_MANIFEST_PATH, label_encoder, gene_names=gene_names, preload=True)
    log(f"Train tubes: {len(train_tube_dataset)}")
    log(f"Val tubes:   {len(val_tube_dataset)}")

    # Stage 1 manifest
    STAGE1_MANIFEST_PATH = str(out_dir / "manifest_stage1.json")
    build_stage1_manifest(train_manifest, save_path=STAGE1_MANIFEST_PATH)
    cell_dataset = CellDataset(STAGE1_MANIFEST_PATH, gene_names=gene_names, preload=True)
    log(f"Cells: {len(cell_dataset)}  |  Cell types: {cell_dataset.n_cell_types()}")
    log(f"X range: [{cell_dataset._X.min():.3f}, {cell_dataset._X.max():.3f}]")
    cell_loader = DataLoader(cell_dataset, batch_size=256, shuffle=True, num_workers=0)

    # ------------------------------------------------------------------
    # 2. Stage 1 — Encoder pre-training
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Stage 1 — Encoder pre-training")
    log("=" * 60)

    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=cell_dataset.n_cell_types(),
        embed_dim=cfg["model"]["embedding_dim"],
    )
    encoder = train_encoder(
        encoder,
        cell_loader,
        n_epochs=8,
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        device=DEVICE,
        verbose=True,
    )
    torch.save(encoder.state_dict(),
               str(out_dir / f"encoder_stage1_2al_bootstrap_{BOOTSTRAP_SEED}.pt"))
    log("Encoder saved.")

    # ------------------------------------------------------------------
    # 3. Stage 2 — MIL training with two-layer attention (encoder frozen)
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Stage 2 — Two-layer MIL training (encoder frozen)")
    log("=" * 60)

    stage2_lr        = 0.001
    stage2_n_epochs  = 40
    stage2_scheduler = "cosine"

    embed_dim          = cfg["model"]["embedding_dim"]
    attention_hidden   = cfg["model"]["attention_hidden_dim"]
    n_classes          = label_encoder.n_classes()

    attention = TwoLayerAttentionModule(
        embed_dim=embed_dim,
        attention_hidden_dim=attention_hidden,
    )
    mil_model = CytokineABMIL_V2(
        encoder=encoder,
        attention=attention,
        n_classes=n_classes,
        embed_dim=embed_dim,
        encoder_frozen=True,
    )

    dynamics_stage2 = train_mil(
        mil_model,
        train_tube_dataset,
        n_epochs=stage2_n_epochs,
        lr=stage2_lr,
        momentum=cfg["training"]["momentum"],
        lr_scheduler=stage2_scheduler,
        lr_warmup_epochs=cfg["training"]["lr_warmup_epochs"],
        log_every_n_epochs=cfg["dynamics"]["log_every_n_epochs"],
        device=DEVICE,
        seed=BOOTSTRAP_SEED,
        verbose=True,
        val_dataset=val_tube_dataset,
        kl_lambda=cfg["training"]["kl_lambda"],
        aux_loss_weight=cfg["training"]["aux_loss_weight"],
    )
    torch.save(mil_model.state_dict(),
               str(out_dir / f"mil_stage2_2al_bootstrap_{BOOTSTRAP_SEED}.pt"))
    log(f"Stage 2 model saved.")
    log(f"Train records: {len(dynamics_stage2['records'])}")
    log(f"Val records:   {len(dynamics_stage2['val_records'])}")

    with open(out_dir / "dynamics_stage2.pkl", "wb") as _fh:
        pickle.dump({
            "records":                        dynamics_stage2["records"],
            "val_records":                    dynamics_stage2["val_records"],
            "logged_epochs":                  dynamics_stage2["logged_epochs"],
            "confusion_entropy_trajectory":   dynamics_stage2["confusion_entropy_trajectory"],
            "val_confusion_entropy_trajectory": dynamics_stage2["val_confusion_entropy_trajectory"],
            "loss_components":                dynamics_stage2["loss_components"],
        }, _fh)
    log("Dynamics Stage 2 saved (dynamics_stage2.pkl).")

    # ------------------------------------------------------------------
    # 4. Dynamics analysis
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Dynamics analysis — Stage 2 (primary)")
    log("=" * 60)

    donor_traj     = aggregate_to_donor_level(dynamics_stage2["records"])
    val_donor_traj = aggregate_to_donor_level(dynamics_stage2["val_records"])

    learnability_result     = rank_cytokines_by_learnability(donor_traj,     exclude=["PBS"])
    val_learnability_result = rank_cytokines_by_learnability(val_donor_traj, exclude=["PBS"])
    ranking     = learnability_result["ranking"]
    val_ranking = val_learnability_result["ranking"]
    val_auc_map = {cyt: auc for cyt, auc in val_ranking}

    extra_train = _aggregate_extra_metrics(dynamics_stage2["records"],     exclude=("PBS",))
    extra_val   = _aggregate_extra_metrics(dynamics_stage2["val_records"], exclude=("PBS",))

    log()
    log(f"Cytokine learnability ranking — Stage 2  |  Bootstrap seed: {BOOTSTRAP_SEED}")
    log(f"Metric (AUC):     {learnability_result['metric_description']}")
    log( "Metric (NormAUC): AUC(p_correct(t) / max(p_correct(t))), trapz / (n-1), aggregated to donor level")
    log( "Metric (Pmax):    max(p_correct_trajectory), aggregated to donor level")
    log()
    hdr = (f"{'Rank':>4}  {'Cytokine':<14}  {'Train AUC':>9}  {'Val AUC':>8}"
           f"  {'NormAUC(T)':>10}  {'NormAUC(V)':>10}  {'Pmax(T)':>7}  {'Pmax(V)':>7}  Group")
    log(hdr)
    log("-" * 100)
    for i, (cyt, auc) in enumerate(ranking, 1):
        group   = "SIMPLE" if cyt in SIMPLE_CYTOKINES else "COMPLEX"
        val_auc = val_auc_map.get(cyt, float("nan"))
        t_norm  = extra_train.get(cyt, {}).get("norm_auc", float("nan"))
        v_norm  = extra_val.get(cyt,   {}).get("norm_auc", float("nan"))
        t_pmax  = extra_train.get(cyt, {}).get("pmax",     float("nan"))
        v_pmax  = extra_val.get(cyt,   {}).get("pmax",     float("nan"))
        log(f"  {i:2d}.  {cyt:<14}  {auc:>9.3f}  {val_auc:>8.3f}"
            f"  {t_norm:>10.3f}  {v_norm:>10.3f}  {t_pmax:>7.3f}  {v_pmax:>7.3f}  {group}")

    simple_aucs  = [auc for cyt, auc in ranking if cyt in SIMPLE_CYTOKINES]
    complex_aucs = [auc for cyt, auc in ranking if cyt in COMPLEX_CYTOKINES]
    log()
    log(f"Group summary (train AUC):")
    log(f"  SIMPLE  mean={np.mean(simple_aucs):.3f}  median={np.median(simple_aucs):.3f}"
        f"  values={[f'{x:.2f}' for x in sorted(simple_aucs, reverse=True)]}")
    log(f"  COMPLEX mean={np.mean(complex_aucs):.3f}  median={np.median(complex_aucs):.3f}"
        f"  values={[f'{x:.2f}' for x in sorted(complex_aucs, reverse=True)]}")

    # SA vs CA attention entropy summary
    log()
    log("-" * 60)
    log("SA vs CA attention entropy — Stage 2")
    log("Metric: mean H(a) = -sum_i a_i log(a_i) per layer, mean across tubes")
    log()
    log(f"{'Cytokine':<20}  {'SA entropy':>10}  {'CA entropy':>10}  {'CA-SA diff':>10}  Group")
    log("-" * 64)
    for cyt in [c for c, _ in ranking]:
        group = "SIMPLE" if cyt in SIMPLE_CYTOKINES else "COMPLEX"
        sa_curves, ca_curves = _extract_layer_entropy(dynamics_stage2["records"], cyt)
        if not sa_curves:
            continue
        sa_mean = float(np.mean([np.mean(c) for c in sa_curves]))
        ca_mean = float(np.mean([np.mean(c) for c in ca_curves]))
        log(f"  {cyt:<20}  {sa_mean:>10.3f}  {ca_mean:>10.3f}  {ca_mean - sa_mean:>10.3f}  {group}")

    # Attention overlap summary
    log()
    log("-" * 60)
    overlap_scores, _overlap_trajs = _compute_attention_overlap(dynamics_stage2["records"])
    log("SA vs CA attention overlap — Stage 2")
    log("Metric: mean_t mean_tubes sum_i min(a_SA_i(t), a_CA_i(t))")
    log("  1=identical (no specialization), 0=fully specialized (attend different cells)")
    log()
    log(f"{'Cytokine':<20}  {'Overlap':>10}  Group")
    log("-" * 40)
    for cyt, score in sorted(overlap_scores.items(), key=lambda x: x[1]):
        if cyt == "PBS":
            continue
        group = "SIMPLE" if cyt in SIMPLE_CYTOKINES else "COMPLEX"
        log(f"  {cyt:<20}  {score:>10.4f}  {group}")

    # Attention entropy (standard)
    log()
    log("-" * 60)
    entropy_result     = compute_cytokine_entropy_summary(dynamics_stage2["records"])
    val_entropy_result = compute_cytokine_entropy_summary(dynamics_stage2["val_records"])
    entropy_sorted = sorted(entropy_result["summary"].items(),
                            key=lambda x: x[1]["mean_entropy"])
    log("SA attention entropy summary (same as single-layer entropy)")
    log(f"Metric: {entropy_result['metric_description']}")
    log()
    log(f"{'Cytokine':<20}  {'Train H':>10}  {'Val H':>8}  Group")
    log("-" * 52)
    for cyt, stats in entropy_sorted:
        group = "SIMPLE" if cyt in SIMPLE_CYTOKINES else ("COMPLEX" if cyt in COMPLEX_CYTOKINES else "PBS")
        val_h = val_entropy_result["summary"].get(cyt, {}).get("mean_entropy", float("nan"))
        log(f"  {cyt:<20}  {stats['mean_entropy']:>10.3f}  {val_h:>8.3f}  {group}")

    # Confusion entropy
    log()
    log("-" * 60)
    confusion_result     = compute_confusion_entropy_summary(
        dynamics_stage2["confusion_entropy_trajectory"], exclude=["PBS"])
    val_confusion_result = compute_confusion_entropy_summary(
        dynamics_stage2["val_confusion_entropy_trajectory"], exclude=["PBS"])
    val_conf_map = {cyt: auc for cyt, auc in val_confusion_result["ranking"]}
    log("Cytokine confusion entropy ranking")
    log(f"Metric: {confusion_result['metric_description']}")
    log()
    log(f"{'Cytokine':<20}  {'Train AUC(H_c)':>14}  {'Val AUC(H_c)':>12}  Group")
    log("-" * 64)
    for cyt, auc in confusion_result["ranking"]:
        group   = "SIMPLE" if cyt in SIMPLE_CYTOKINES else ("COMPLEX" if cyt in COMPLEX_CYTOKINES else "PBS")
        val_auc = val_conf_map.get(cyt, float("nan"))
        log(f"  {cyt:<20}  {auc:>14.3f}  {val_auc:>12.3f}  {group}")

    # ------------------------------------------------------------------
    # 6. Hypothesis test
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Hypothesis test — Stage 2 (train donors)")
    log("=" * 60)

    auc_map = {cyt: auc for cyt, auc in ranking}
    simple_auc_vals  = [auc_map[c] for c in SIMPLE_CYTOKINES  if c in auc_map]
    complex_auc_vals = [auc_map[c] for c in COMPLEX_CYTOKINES if c in auc_map]

    u_stat, p_one_sided = mannwhitneyu(simple_auc_vals, complex_auc_vals, alternative="greater")
    n1, n2 = len(simple_auc_vals), len(complex_auc_vals)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)

    log(f"Bootstrap seed: {BOOTSTRAP_SEED}")
    log()
    log(f"  SIMPLE  AUCs: {[f'{x:.3f}' for x in sorted(simple_auc_vals, reverse=True)]}")
    log(f"  COMPLEX AUCs: {[f'{x:.3f}' for x in sorted(complex_auc_vals, reverse=True)]}")
    log()
    log(f"  Mann-Whitney U statistic: {u_stat:.1f}")
    log(f"  One-sided p-value (simple > complex): {p_one_sided:.4f}")
    log(f"  Rank-biserial correlation r = {r_rb:.3f}")
    log()
    alpha = 0.05
    if p_one_sided < alpha:
        log(f"  Result: p < {alpha} → hypothesis SUPPORTED (simple > complex, train donors)")
    else:
        log(f"  Result: p >= {alpha} → hypothesis NOT SUPPORTED at alpha={alpha}")
    log()
    log("  Note: n=5 per group → low power. Report effect size alongside p-value.")

    val_auc_vals_simple  = [val_auc_map.get(c, float("nan")) for c in SIMPLE_CYTOKINES]
    val_auc_vals_complex = [val_auc_map.get(c, float("nan")) for c in COMPLEX_CYTOKINES]
    val_auc_vals_simple  = [x for x in val_auc_vals_simple  if not np.isnan(x)]
    val_auc_vals_complex = [x for x in val_auc_vals_complex if not np.isnan(x)]
    if val_auc_vals_simple and val_auc_vals_complex:
        u_val, p_val = mannwhitneyu(val_auc_vals_simple, val_auc_vals_complex, alternative="greater")
        r_val = 1 - (2 * u_val) / (len(val_auc_vals_simple) * len(val_auc_vals_complex))
        log(f"  Val donors (D2, D3) — informative only:")
        log(f"    One-sided p = {p_val:.4f}  |  rank-biserial r = {r_val:.3f}")

    with open(out_dir / "hypothesis_test.json", "w") as f:
        json.dump({
            "bootstrap_seed": BOOTSTRAP_SEED,
            "simple_cytokines": SIMPLE_CYTOKINES,
            "complex_cytokines": COMPLEX_CYTOKINES,
            "simple_aucs_train": simple_auc_vals,
            "complex_aucs_train": complex_auc_vals,
            "u_stat": float(u_stat),
            "p_one_sided": float(p_one_sided),
            "rank_biserial_r": float(r_rb),
            "supported_at_alpha_0.05": bool(p_one_sided < 0.05),
        }, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Validation checks
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Validation")
    log("=" * 60)

    train_ranking_list = rank_cytokines_by_learnability(donor_traj,     exclude=["PBS"])["ranking"]
    val_ranking_list   = rank_cytokines_by_learnability(val_donor_traj, exclude=["PBS"])["ranking"]
    train_order_v  = [c for c, _ in train_ranking_list]
    val_order_v    = [c for c, _ in val_ranking_list]
    val_rank_by_cyt   = {c: i for i, c in enumerate(val_order_v)}
    val_ranks_aligned = [val_rank_by_cyt[c] for c in train_order_v]
    rho_gen, pval_gen = spearmanr(range(len(train_order_v)), val_ranks_aligned)

    log(f"Donor-level generalization — Stage 2  |  Bootstrap seed: {BOOTSTRAP_SEED}")
    log(f"  Train/val rank correlation: Spearman rho = {rho_gen:.3f}  (p={pval_gen:.3f})")
    log(f"  Stable (rho > 0.7): {rho_gen > 0.7}")
    log()
    log(f"  {'Cytokine':<20}  {'Train AUC':>10}  {'Val AUC':>9}  {'Ratio V/T':>10}  Group")
    log("  " + "-" * 64)
    val_auc_map2 = {c: a for c, a in val_ranking_list}
    for cyt, train_auc in train_ranking_list:
        val_auc_v = val_auc_map2.get(cyt, float("nan"))
        ratio     = val_auc_v / train_auc if train_auc > 0 else float("nan")
        group     = "SIMPLE" if cyt in SIMPLE_CYTOKINES else "COMPLEX"
        flag      = "  ← possible overfit" if ratio < 0.6 else ""
        log(f"  {cyt:<20}  {train_auc:>10.3f}  {val_auc_v:>9.3f}  {ratio:>10.2f}  {group}{flag}")

    log()
    sampled_groups = {
        "simple_sampled":  SIMPLE_CYTOKINES,
        "complex_sampled": COMPLEX_CYTOKINES,
    }
    grouping_result = check_functional_groupings(donor_traj, sampled_groups)
    log("Known functional groupings — within-group consistency")
    for group, result in grouping_result.items():
        log(f"\n  {group}:")
        for k, v in result.items():
            log(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # 8. Figures
    # ------------------------------------------------------------------
    log()
    log("=" * 60)
    log("Saving figures...")
    log("=" * 60)

    _plot_learning_curves(
        dynamics_stage2, SIMPLE_CYTOKINES, COMPLEX_CYTOKINES, BOOTSTRAP_SEED, out_dir)
    log(f"  Saved: learning_curves_2al_bootstrap_{BOOTSTRAP_SEED}.png")

    _plot_per_donor_curves(
        dynamics_stage2, SUBSET_CYTOKINES, SIMPLE_CYTOKINES, COMPLEX_CYTOKINES,
        BOOTSTRAP_SEED, out_dir)
    log(f"  Saved: per_donor_curves_<cytokine>_2al_bootstrap_{BOOTSTRAP_SEED}.png (x{len(SUBSET_CYTOKINES)})")

    _plot_scatter(extra_train, extra_val, SUBSET_CYTOKINES,
                  SIMPLE_CYTOKINES, COMPLEX_CYTOKINES, BOOTSTRAP_SEED, out_dir)
    log(f"  Saved: scatter_normauc_pmax_2al_bootstrap_{BOOTSTRAP_SEED}.png")

    _plot_sa_vs_ca_entropy(
        dynamics_stage2, SIMPLE_CYTOKINES, COMPLEX_CYTOKINES, BOOTSTRAP_SEED, out_dir)
    log(f"  Saved: sa_vs_ca_entropy_2al_bootstrap_{BOOTSTRAP_SEED}.png")

    _plot_attention_overlap(
        dynamics_stage2, SIMPLE_CYTOKINES, COMPLEX_CYTOKINES, BOOTSTRAP_SEED, out_dir)
    log(f"  Saved: attention_overlap_2al_bootstrap_{BOOTSTRAP_SEED}.png")

    _plot_loss_components(
        dynamics_stage2, BOOTSTRAP_SEED,
        kl_lambda=cfg["training"]["kl_lambda"],
        aux_loss_weight=cfg["training"]["aux_loss_weight"],
        out_dir=out_dir,
    )
    log(f"  Saved: loss_components_2al_bootstrap_{BOOTSTRAP_SEED}.png")

    _plot_entropy_gap(
        dynamics_stage2, SIMPLE_CYTOKINES, COMPLEX_CYTOKINES, BOOTSTRAP_SEED, out_dir)
    log(f"  Saved: entropy_gap_2al_bootstrap_{BOOTSTRAP_SEED}.png")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    log()
    log(f"All results saved to: {out_dir}")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
