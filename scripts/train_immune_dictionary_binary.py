"""
Train binary AB-MIL models (stimulus vs PBS) for the Immune Dictionary
(Cui Nature 2024) — the Bridge step that discovers each cytokine's gene
signature S_X^binary for the cross_asym pipeline (CLAUDE.md §26).

Clone of train_oesinghaus_binary_missing16.py with:
  * ID manifest/HVG paths
  * Targets = 12 benchmark cytokines locked at pre-registration time
    (see reports/immune_dictionary/PRE_REGISTRATION.md and
    scripts/finalize_id_labels.py BENCHMARK_CYTOKINES)
  * Val mouse auto-detected from <manifest_dir>/build_metadata.json:val_mouse
  * Same WIDE HPs as Oes-wide (embed_dim=512, hidden_dims=(512,512),
    attention_hidden_dim=128, Stage2 LR 3e-5) — ID is 4000 noisy HVG like Oes
    so the Oes-wide config applies (M4 part 2 in reports/method_deep_dive/).

Outputs (one run dir):
  encoder_shared_stage1.pt
  manifest_stage1_shared.json
  model_<cytokine>.pt           one per benchmark cytokine
  dynamics_<cytokine>.pkl       one per benchmark cytokine
  manifest_train_<cyt>.json, manifest_val_<cyt>.json
  label_encoder_<cyt>.json
  learning_curves_id_binary_<seed>.png

Usage:
  python scripts/train_immune_dictionary_binary.py
  python scripts/train_immune_dictionary_binary.py --seed 42 --output_dir /path
"""

import argparse
import copy
import json
import os
import pickle
import sys
import traceback
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

MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/manifest.json"
HVG_PATH      = "/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/hvg_list.json"
OUTPUT_BASE   = REPO_ROOT / "results" / "id_cascade" / "binary"
CONTROL       = "PBS"

# Fallback val mouse if build_metadata.json absent (the ID adapter writes it).
# Each benchmark cytokine spans rep01/rep02/rep03 -> hold out rep03 as val.
DEFAULT_VAL_DONORS = ["rep03"]

# Pre-registered 12 benchmark cytokines.  SCP `cyt` machine names; MUST match
# scripts/finalize_id_labels.py BENCHMARK_CYTOKINES and the cytokine column
# strings in the ID manifest (built by build_pseudotubes_immune_dictionary.py).
BENCHMARK_CYTOKINES = [
    "IFNb", "IFNg", "IL1b", "IL10", "IL12", "IL13",
    "IL15", "IL18", "IL2", "IL4", "IL6", "TNFa",
]

# Wide HPs — match the Oes-wide bridge variant used to produce the published
# cross_asym 88% on Oesinghaus (M4 part 2, train_oesinghaus_binary_missing16.py).
EMBED_DIM            = 512
HIDDEN_DIMS          = (512, 512)
ATTENTION_HIDDEN_DIM = 128
STAGE1_EPOCHS        = 20
STAGE1_LR            = 0.005
STAGE1_MOMENTUM      = 0.9
STAGE2_EPOCHS        = 250
STAGE2_LR            = 0.00003
STAGE2_MOMENTUM      = 0.90
LOG_EVERY            = 1
SEED                 = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_val_mouse_from_metadata(manifest_path: str):
    """If <manifest_dir>/build_metadata.json has a 'val_mouse' key, return it.

    The ID adapter (build_pseudotubes_immune_dictionary.py) picks the outlier-
    PBS-PCA mouse at build time and writes it here.  Returns None if missing.
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


def _train_one_binary_model(
    target,
    control,
    manifest,
    gene_names,
    shared_encoder,
    out_dir,
    device,
    seed,
    log,
    embed_dim,
    hidden_dims,
    attention_hidden_dim,
    val_donors,
):
    """Train one binary AB-MIL (target vs control), shared frozen encoder."""
    log(f"\n{'=' * 50}")
    log(f"Training binary model: {target} vs {control}")
    log("=" * 50)

    bin_manifest, label_enc = make_binary_manifest(manifest, target, control=control)
    train_m, val_m = split_manifest_by_donor(bin_manifest, val_donors)
    n_pos_train = sum(1 for e in train_m if e["cytokine"] == target)
    if n_pos_train == 0:
        log(f"  SKIP {target}: no positive tubes in train split "
            f"(val donors hold them all)")
        return None
    log(f"  {target}: {len(train_m)} train tubes ({n_pos_train} positive), "
        f"{len(val_m)} val tubes")

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
    with open(out_dir / f"dynamics_{safe_target}.pkl", "wb") as fh:
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
    colors = plt.cm.tab20(np.linspace(0, 1, len(BENCHMARK_CYTOKINES)))
    for color, target in zip(colors, BENCHMARK_CYTOKINES):
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
    ax.set_title("Immune Dictionary binary AB-MIL — cytokine vs PBS")
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
    fname = f"learning_curves_id_binary_{seed}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train binary AB-MIL for 12 ID benchmark cytokines vs PBS.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=MANIFEST_PATH)
    parser.add_argument("--hvg_path", type=str, default=HVG_PATH)
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Training seed (default: {SEED})")
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--hidden_dims", type=int, nargs="+",
                        default=list(HIDDEN_DIMS))
    parser.add_argument("--attention_hidden_dim", type=int,
                        default=ATTENTION_HIDDEN_DIM)
    parser.add_argument("--val_donors", nargs="*", default=None,
                        help="Mouse names to hold out for val.  If unset, read "
                             "from <manifest_dir>/build_metadata.json:val_mouse "
                             f"and fall back to {DEFAULT_VAL_DONORS}.")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Override benchmark cytokine list (for smoke runs); "
                             "default is BENCHMARK_CYTOKINES from this file.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else SEED
    embed_dim = args.embed_dim
    hidden_dims = tuple(args.hidden_dims)
    attention_hidden_dim = args.attention_hidden_dim
    targets = list(args.targets) if args.targets else list(BENCHMARK_CYTOKINES)

    # Resolve val mouse
    if args.val_donors is None:
        from_meta = _read_val_mouse_from_metadata(args.manifest_path)
        if from_meta is not None:
            val_donors = [from_meta]
        else:
            val_donors = list(DEFAULT_VAL_DONORS)
    else:
        val_donors = list(args.val_donors)

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
    log("Immune Dictionary binary AB-MIL training (Bridge for cross_asym)")
    log("=" * 60)
    log(f"Benchmark cytokines ({len(targets)}): {targets}")
    log(f"Val donors (held out):                {val_donors}")
    log(f"Control:                              {CONTROL}")
    log(f"Output dir:                           {out_dir}")
    log(f"Started:                              {timestamp}")
    log(f"Training seed:                        {seed}")
    log("")
    log("Hyperparameters (Oes-wide variant; 4000 noisy HVG like ID):")
    log(f"  embed_dim:            {embed_dim}")
    log(f"  hidden_dims:          {hidden_dims}")
    log(f"  attention_hidden_dim: {attention_hidden_dim}")
    log(f"  Stage 1 epochs:       {STAGE1_EPOCHS}  lr={STAGE1_LR}")
    log(f"  Stage 2 epochs:       {STAGE2_EPOCHS}  lr={STAGE2_LR}")
    log("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    log("\nLoading manifest and HVG list...")
    with open(args.manifest_path) as fh:
        manifest = json.load(fh)
    with open(args.hvg_path) as fh:
        gene_names = json.load(fh)
    log(f"Manifest entries: {len(manifest)}")
    log(f"HVGs:             {len(gene_names)}")

    manifest_cytokines = {e["cytokine"] for e in manifest}
    missing = [c for c in targets if c not in manifest_cytokines]
    if missing:
        for c in missing:
            log(f"FATAL: cytokine not found in manifest: {c}")
        log(f"Available cytokines in manifest (first 30): "
            f"{sorted(manifest_cytokines)[:30]}")
        raise ValueError(f"Missing cytokines: {missing}")

    filtered_manifest = filter_manifest(manifest, targets, include_pbs=True)
    log(f"Filtered manifest size: {len(filtered_manifest)} entries "
        f"({len(targets)} cytokines + PBS)")

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
    # Loop — one binary model per benchmark cytokine
    # -----------------------------------------------------------------
    all_dynamics = {}
    for i, target in enumerate(targets, start=1):
        log(f"\n[{i}/{len(targets)}] {target} START")
        try:
            payload = _train_one_binary_model(
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
                val_donors=val_donors,
            )
            if payload is not None:
                all_dynamics[target] = payload
            log(f"[{i}/{len(targets)}] {target} DONE")
        except Exception as e:
            log(f"[{i}/{len(targets)}] {target} ERROR: "
                f"{type(e).__name__}: {e}")
            log(traceback.format_exc())
            # Defensive: don't kill the whole run; continue.
            continue

    # -----------------------------------------------------------------
    # Learning-curve plot
    # -----------------------------------------------------------------
    log("\n" + "=" * 60)
    log("Saving learning-curve plot...")
    log("=" * 60)
    _plot_learning_curves(all_dynamics, out_dir, seed, log)

    log("")
    log(f"Done. {len(all_dynamics)}/{len(targets)} cytokines trained.")
    log(f"Results in: {out_dir}")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
