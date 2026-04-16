"""
Extract per-cytokine learnability AUC for all 91 cytokines across seeds,
and the IL-12→IFN-γ asymmetry trajectory for 3 seeds.
Outputs a small JSON file for local figure generation.

Usage:
  python scripts/extract_figure_data.py \
    --results_dir results/oesinghaus_full_v2 \
    --seeds 42 123 7 \
    --output results/figure_data.json
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results")


def compute_donor_auc(records, cytokine):
    """AUC of mean donor p_correct_trajectory for one cytokine."""
    # Group by donor
    donor_trajs = defaultdict(list)
    for r in records:
        if r["cytokine"] == cytokine:
            donor_trajs[r["donor"]].append(np.array(r["p_correct_trajectory"]))
    if not donor_trajs:
        return float("nan")
    donor_means = [np.mean(trajs, axis=0) for trajs in donor_trajs.values()]
    grand_mean = np.mean(donor_means, axis=0)
    T = len(grand_mean)
    auc = float(np.trapz(grand_mean) / (T - 1)) if T > 1 else 0.0
    return auc


def compute_confusion_trajectory(records, label_encoder_cytokines, cyt_a, cyt_b):
    """
    C(A, B, t) = mean over A-tubes of softmax[B, t].
    Returns np.array of shape (T,).
    """
    try:
        b_idx = label_encoder_cytokines.index(cyt_b)
    except ValueError:
        return None
    trajs = []
    for r in records:
        if r["cytokine"] == cyt_a:
            st = np.array(r["softmax_trajectory"])  # shape (K, T)
            trajs.append(st[b_idx])
    if not trajs:
        return None
    return np.mean(trajs, axis=0)


def load_seed(results_dir, seed):
    pkl_path = results_dir / f"seed_{seed}" / "dynamics.pkl"
    print(f"  Loading {pkl_path} ...", flush=True)
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/oesinghaus_full_v2")
    parser.add_argument("--seeds", nargs="+", default=["42", "123", "7"])
    parser.add_argument("--output", default="results/figure_data.json")
    args = parser.parse_args()

    results_dir = RESULTS_DIR / Path(args.results_dir).name
    output_path = RESULTS_DIR.parent / args.output

    print(f"Results dir: {results_dir}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = {}

    for seed in args.seeds:
        print(f"\nProcessing seed {seed} ...", flush=True)
        d = load_seed(results_dir, seed)
        records = d["records"]
        label_enc = d["label_encoder_cytokines"]

        # All unique cytokines from records
        cytokines = sorted(set(r["cytokine"] for r in records))
        print(f"  {len(cytokines)} cytokines, {len(records)} records", flush=True)

        # Per-cytokine AUC
        print("  Computing learnability AUCs...", flush=True)
        aucs = {}
        for cyt in cytokines:
            aucs[cyt] = compute_donor_auc(records, cyt)

        # IL-12 → IFN-γ confusion trajectories
        print("  Computing IL-12/IFN-gamma confusion trajectories...", flush=True)
        c_il12_ifng = compute_confusion_trajectory(
            records, label_enc, "IL-12", "IFN-gamma"
        )
        c_ifng_il12 = compute_confusion_trajectory(
            records, label_enc, "IFN-gamma", "IL-12"
        )

        logged_epochs = d.get("logged_epochs", list(range(1, d.get("stage2_epochs", 100) + 1)))

        out[seed] = {
            "learnability_auc": aucs,
            "il12_ifng": c_il12_ifng.tolist() if c_il12_ifng is not None else None,
            "ifng_il12": c_ifng_il12.tolist() if c_ifng_il12 is not None else None,
            "logged_epochs": logged_epochs,
            "n_epochs": d.get("stage2_epochs", len(logged_epochs)),
        }
        print(f"  Done. Top 5 AUC: {sorted(aucs.items(), key=lambda x: -x[1])[:5]}", flush=True)

    print(f"\nSaving to {output_path} ...", flush=True)
    with open(output_path, "w") as f:
        json.dump(out, f)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
