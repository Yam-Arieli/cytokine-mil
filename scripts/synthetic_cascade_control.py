"""
Experiment 0: Synthetic Positive Control for Confusion Dynamics (GO/NO-GO gate).

Constructs synthetic pseudo-tubes where a known mixture exists:
  - Take A-tubes and inject alpha fraction of cells from B-tubes.
  - This simulates a cascade signal: A-tubes with a controlled weak B signal.

Trains an AB-MIL on the modified dataset and checks whether the model shows
late-onset confusion C(A,B,t) > C(B,A,t) for synthetic mixtures but not controls.

If this test passes (late-onset confusion detectable at alpha >= 0.1):
  → Method can detect cascade signals → proceed to real data analysis.
If this test fails:
  → Signal too weak at 24h / 90-class setting → reconsider approach.

Results saved to:
    results/synthetic_cascade_control/run_{timestamp}/

Usage:
    python scripts/synthetic_cascade_control.py --pair_a IL-12 --pair_b IFN-gamma
    python scripts/synthetic_cascade_control.py --pair_a IL-12 --pair_b IFN-gamma --alpha 0.1 0.2 0.3
    python scripts/synthetic_cascade_control.py --pair_a IL-6  --pair_b IL-10 --seed 123
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import (
    build_encoder,
    build_mil_model,
    build_stage1_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder
from cytokine_mil.training.train_mil import train_mil
from cytokine_mil.analysis.confusion_dynamics import (
    compute_confusion_trajectory,
    compute_asymmetry_score,
    compute_temporal_profile,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH      = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
OUTPUT_BASE   = Path(__file__).parent.parent / "results" / "synthetic_cascade_control"
VAL_DONORS    = ["Donor2", "Donor3"]

EMBED_DIM            = 128
ATTENTION_HIDDEN_DIM = 64
STAGE1_EPOCHS        = 30
STAGE1_LR            = 0.01
STAGE2_EPOCHS        = 100
STAGE2_LR            = 0.01
LOG_EVERY            = 1
SEED                 = 42


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Synthetic cascade positive control.")
    p.add_argument("--pair_a",         type=str,   default="IL-12",
                   help="True cytokine label A (cascade source).")
    p.add_argument("--pair_b",         type=str,   default="IFN-gamma",
                   help="Downstream cytokine label B (cascade target).")
    p.add_argument("--alpha",          type=float, nargs="+", default=[0.1, 0.2, 0.3],
                   help="Mixing fractions to test (fraction of A-tube cells replaced by B).")
    p.add_argument("--seed",                type=int,   default=SEED)
    p.add_argument("--stage2_epochs",       type=int,   default=STAGE2_EPOCHS)
    p.add_argument("--lr",                  type=float, default=STAGE2_LR)
    p.add_argument("--output_dir",          type=str,   default=None)
    p.add_argument("--subset_n_cytokines",  type=int,   default=None,
                   help="If set, randomly sample this many cytokines (always including "
                        "pair_a, pair_b, PBS) to reduce training time for the GO/NO-GO test.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic tube construction
# ---------------------------------------------------------------------------

def _build_synthetic_manifest(
    manifest: list,
    pair_a: str,
    pair_b: str,
    alpha: float,
    out_dir: Path,
    rng: np.random.Generator,
    gene_names: list,
) -> list:
    """
    Construct synthetic A-tubes by replacing alpha fraction of cells with B cells.

    For each A-tube:
      1. Load A-tube and B-tube from the same donor (or any B-tube if unavailable).
      2. Randomly select ceil(alpha * N_A) cells to replace with randomly sampled B cells.
      3. Write synthetic tube to out_dir/synthetic_tubes/.
      4. Return a new manifest entry pointing to the synthetic tube.

    The original (unmodified) A-tubes and all other cytokine tubes are retained as-is,
    so the classifier sees both the synthetic and unmodified A-tubes during training.

    Returns the synthetic manifest (full original manifest + synthetic A entries).
    """
    syn_dir = out_dir / "synthetic_tubes"
    syn_dir.mkdir(parents=True, exist_ok=True)

    # Group B-tubes by donor for matched mixing.
    b_tubes_by_donor = {}
    a_entries = []
    for entry in manifest:
        if entry["cytokine"] == pair_b:
            donor = entry["donor"]
            b_tubes_by_donor.setdefault(donor, []).append(entry)
        if entry["cytokine"] == pair_a:
            a_entries.append(entry)

    all_b_entries = [e for entries in b_tubes_by_donor.values() for e in entries]
    if not all_b_entries:
        raise ValueError(f"No tubes found for cytokine '{pair_b}' in manifest.")

    synthetic_entries = []
    for a_entry in a_entries:
        donor = a_entry["donor"]
        b_pool = b_tubes_by_donor.get(donor, all_b_entries)
        b_entry = b_pool[rng.integers(len(b_pool))]

        a_tube = anndata.read_h5ad(a_entry["path"])
        b_tube = anndata.read_h5ad(b_entry["path"])

        # Filter to shared genes (should already match HVG list).
        shared_genes = list(set(a_tube.var_names) & set(b_tube.var_names))
        a_tube = a_tube[:, shared_genes]
        b_tube = b_tube[:, shared_genes]

        N_a = a_tube.n_obs
        n_replace = max(1, int(np.ceil(alpha * N_a)))
        n_replace = min(n_replace, b_tube.n_obs)

        # Sample cells to replace from A and cells from B.
        a_keep_idx = rng.choice(N_a, N_a - n_replace, replace=False)
        b_sample_idx = rng.choice(b_tube.n_obs, n_replace, replace=False)

        a_keep = a_tube[a_keep_idx, :]
        b_sample = b_tube[b_sample_idx, :]

        # Align B obs columns to A (drop extra, fill missing with NA).
        b_obs = b_sample.obs.copy()
        for col in a_keep.obs.columns:
            if col not in b_obs.columns:
                b_obs[col] = "unknown"
        b_sample_aligned = anndata.AnnData(
            X=b_sample.X, obs=b_obs[a_keep.obs.columns], var=a_keep.var,
        )

        syn_tube = anndata.concat(
            [a_keep, b_sample_aligned], axis=0, join="outer",
        )

        syn_fname = (
            f"synthetic_{pair_a}_{pair_b}_alpha{alpha:.2f}"
            f"_{donor}_tube{a_entry['tube_idx']}.h5ad"
        ).replace("/", "_")
        syn_path = syn_dir / syn_fname
        syn_tube.write_h5ad(str(syn_path))

        synthetic_entries.append({
            "path":               str(syn_path),
            "donor":              donor,
            "cytokine":           f"{pair_a}_synthetic_{alpha:.2f}",
            "n_cells":            syn_tube.n_obs,
            "cell_types_included": a_entry.get("cell_types_included", []),
            "tube_idx":           a_entry["tube_idx"],
            "is_synthetic":       True,
            "alpha":              alpha,
            "source_a":           a_entry["path"],
            "source_b":           b_entry["path"],
        })

    return manifest + synthetic_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_BASE / f"run_{ts}_seed{args.seed}_{args.pair_a}_{args.pair_b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(out_dir / "train.log", "w")
    def log(msg=""):
        print(msg)
        print(msg, file=log_file, flush=True)

    log("=" * 70)
    log("EXPERIMENT 0: Synthetic Positive Control — Confusion Dynamics GO/NO-GO")
    log(f"Pair: {args.pair_a} → {args.pair_b}")
    log(f"Mixing fractions: {args.alpha}")
    log(f"Seed: {args.seed}")
    log("Expected result: synthetic A-tubes show elevated C(A,B,t) with late onset.")
    log("Failure: signal not detectable → method cannot detect cascade signals.")
    log("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Load manifest & gene names
    # ------------------------------------------------------------------
    log("\nLoading manifest...")
    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    with open(HVG_PATH) as fh:
        gene_names = json.load(fh)

    # ------------------------------------------------------------------
    # Optional: subset manifest to N cytokines (always keep pair_a, pair_b, PBS)
    # ------------------------------------------------------------------
    if args.subset_n_cytokines is not None:
        all_cyts = sorted({e["cytokine"] for e in manifest if e["cytokine"] != "PBS"})
        required = {args.pair_a, args.pair_b}
        pool = [c for c in all_cyts if c not in required]
        n_extra = max(0, args.subset_n_cytokines - len(required))
        sampled = sorted(rng.choice(pool, size=min(n_extra, len(pool)), replace=False).tolist())
        keep = required | set(sampled) | {"PBS"}
        manifest = [e for e in manifest if e["cytokine"] in keep]
        log(f"  Subset: {len(keep)-1} cytokines + PBS "
            f"({args.pair_a}, {args.pair_b} + {len(sampled)} random)")
        log(f"  Manifest after subset: {len(manifest)} tubes")

    # ------------------------------------------------------------------
    # Label encoder (full 91 classes + synthetic labels)
    # ------------------------------------------------------------------
    all_cytokines = sorted({e["cytokine"] for e in manifest if e["cytokine"] != "PBS"})
    label_enc_base = CytokineLabel().fit(manifest)
    log(f"  Base: {label_enc_base.n_classes()} classes")

    # ------------------------------------------------------------------
    # Stage 1 encoder (trained once on base cytokines)
    # ------------------------------------------------------------------
    log("\nStage 1: training shared encoder...")
    stage1_manifest = build_stage1_manifest(manifest)
    stage1_path = out_dir / "manifest_stage1.json"
    with open(stage1_path, "w") as fh:
        json.dump(stage1_manifest, fh)

    cell_dataset = CellDataset(
        str(stage1_path), gene_names=gene_names, preload=True,
    )
    cell_loader = DataLoader(cell_dataset, batch_size=256, shuffle=True, num_workers=0)
    n_cell_types = len(cell_dataset.cell_type_to_idx)

    encoder = build_encoder(
        n_input_genes=len(gene_names), n_cell_types=n_cell_types, embed_dim=EMBED_DIM,
    )
    train_encoder(
        encoder=encoder, dataloader=cell_loader, n_epochs=STAGE1_EPOCHS,
        lr=STAGE1_LR, momentum=0.9, device=device, verbose=True,
    )
    torch.save(encoder.state_dict(), out_dir / "encoder_stage1.pt")
    log("  Saved: encoder_stage1.pt")

    # ------------------------------------------------------------------
    # Run one training per alpha value + a control (alpha=0)
    # ------------------------------------------------------------------
    alphas_to_run = [0.0] + list(args.alpha)
    all_results = {}

    for alpha in alphas_to_run:
        label = f"alpha={alpha:.2f}"
        log(f"\n{'=' * 60}")
        log(f"Training with {label} mixing ({args.pair_a} ← {int(alpha*100)}% {args.pair_b} cells)")
        log("=" * 60)

        alpha_dir = out_dir / label.replace("=", "").replace(".", "p")
        alpha_dir.mkdir(exist_ok=True)

        # Build modified manifest
        if alpha == 0.0:
            modified_manifest = manifest
            # Label encoder same as base
            cytokines_for_enc = all_cytokines
        else:
            modified_manifest = _build_synthetic_manifest(
                manifest, args.pair_a, args.pair_b, alpha, alpha_dir, rng, gene_names,
            )
            syn_label = f"{args.pair_a}_synthetic_{alpha:.2f}"
            cytokines_for_enc = sorted(
                {e["cytokine"] for e in modified_manifest if e["cytokine"] != "PBS"}
            )

        label_enc = CytokineLabel().fit(modified_manifest)

        train_m, val_m = split_manifest_by_donor(modified_manifest, VAL_DONORS)
        train_path = alpha_dir / "manifest_train.json"
        val_path   = alpha_dir / "manifest_val.json"
        with open(train_path, "w") as fh:
            json.dump(train_m, fh)
        with open(val_path, "w") as fh:
            json.dump(val_m, fh)

        train_dataset = PseudoTubeDataset(
            str(train_path), label_enc, gene_names=gene_names, preload=False,
        )
        val_dataset = PseudoTubeDataset(
            str(val_path), label_enc, gene_names=gene_names, preload=False,
        )

        import copy
        enc_copy = copy.deepcopy(encoder)
        model = build_mil_model(
            enc_copy, embed_dim=EMBED_DIM, attention_hidden_dim=ATTENTION_HIDDEN_DIM,
            n_classes=label_enc.n_classes(), encoder_frozen=True,
        )

        dynamics = train_mil(
            model, train_dataset, n_epochs=args.stage2_epochs,
            lr=args.lr, momentum=0.9, log_every_n_epochs=LOG_EVERY,
            device=device, seed=args.seed, verbose=True, val_dataset=val_dataset,
        )

        # Compute confusion trajectory
        confusion, cyt_names = compute_confusion_trajectory(
            dynamics["records"], label_enc,
        )

        a_idx = label_enc.encode(args.pair_a) if args.pair_a in label_enc.cytokines else None
        b_idx = label_enc.encode(args.pair_b) if args.pair_b in label_enc.cytokines else None

        result = {
            "alpha": alpha,
            "dynamics": dynamics,
            "confusion": confusion,
            "cytokine_names": cyt_names,
            "a_idx": a_idx,
            "b_idx": b_idx,
            "label_encoder": label_enc,
        }

        if a_idx is not None and b_idx is not None:
            prof_ab = compute_temporal_profile(confusion, a_idx, b_idx)
            prof_ba = compute_temporal_profile(confusion, b_idx, a_idx)
            asym = compute_asymmetry_score(confusion)
            result["profile_ab"] = prof_ab
            result["profile_ba"] = prof_ba
            result["asymmetry_ab"] = float(asym[a_idx, b_idx])
            log(f"  C({args.pair_a},{args.pair_b}): "
                f"peak_epoch={prof_ab['peak_epoch']}, "
                f"profile_type={prof_ab['profile_type']}, "
                f"max={prof_ab['max_value']:.4f}")
            log(f"  C({args.pair_b},{args.pair_a}): "
                f"peak_epoch={prof_ba['peak_epoch']}, "
                f"profile_type={prof_ba['profile_type']}, "
                f"max={prof_ba['max_value']:.4f}")
            log(f"  Asymmetry({args.pair_a}→{args.pair_b}): {result['asymmetry_ab']:+.4f}")

        pkl_path = alpha_dir / "result.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump({k: v for k, v in result.items() if k != "label_encoder"}, fh)

        all_results[alpha] = result

    # ------------------------------------------------------------------
    # Plot confusion trajectories
    # ------------------------------------------------------------------
    _plot_results(all_results, args, out_dir, log)

    log("\nDone. Check figures in:", str(out_dir))
    log_file.close()


def _plot_results(all_results, args, out_dir, log):
    """Plot C(A,B,t) and C(B,A,t) for all alpha values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(all_results)))
    alphas_sorted = sorted(all_results.keys())

    for i, alpha in enumerate(alphas_sorted):
        res = all_results[alpha]
        if res.get("profile_ab") is None:
            continue
        traj_ab = res["profile_ab"]["trajectory"]
        traj_ba = res["profile_ba"]["trajectory"]
        T = len(traj_ab)
        xs = np.arange(T)
        lbl = f"alpha={alpha:.2f}" if alpha > 0 else "control (alpha=0)"
        axes[0].plot(xs, traj_ab, color=colors[i], label=lbl, linewidth=1.5)
        axes[1].plot(xs, traj_ba, color=colors[i], label=lbl, linewidth=1.5)

    axes[0].set_title(f"C({args.pair_a} → {args.pair_b}, t)")
    axes[0].set_xlabel("Logged epoch")
    axes[0].set_ylabel("Mean softmax P(predict B | true A)")
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"C({args.pair_b} → {args.pair_a}, t)  [reverse; should stay low]")
    axes[1].set_xlabel("Logged epoch")
    axes[1].set_ylabel("Mean softmax P(predict A | true B)")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"Synthetic Cascade Control: {args.pair_a}→{args.pair_b}\n"
        "Late-onset elevation in left plot = method detects cascade signal (GO).",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_trajectory.png", dpi=150)
    plt.close(fig)
    log(f"  Saved: confusion_trajectory.png")

    # Asymmetry summary
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    valid = [(a, r["asymmetry_ab"]) for a, r in all_results.items() if "asymmetry_ab" in r]
    if valid:
        xs2, ys2 = zip(*sorted(valid))
        ax2.bar([str(x) for x in xs2], ys2, color="steelblue")
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("Alpha (mixing fraction)")
        ax2.set_ylabel(f"Asymmetry score  Asym({args.pair_a}→{args.pair_b})")
        ax2.set_title("Asymmetry by mixing fraction\n(positive = late-epoch A confuses toward B more than B toward A)")
        fig2.tight_layout()
        fig2.savefig(out_dir / "asymmetry_by_alpha.png", dpi=150)
        plt.close(fig2)
        log(f"  Saved: asymmetry_by_alpha.png")


if __name__ == "__main__":
    main()
