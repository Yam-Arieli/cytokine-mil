"""
Analyze centroid trajectory results from the 10-seed centroid_traj_array experiment.

Run on cluster:
    /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
        scripts/analyze_centroid_trajectory.py \
        --results_dir /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory \
        --output_dir /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/analysis
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

WORKDIR = "/cs/labs/mornitzan/yam.arieli/cytokine-mil"
sys.path.insert(0, WORKDIR)

from cytokine_mil.analysis.latent_geometry import (
    compute_trajectory_bias_per_donor,
    compute_trajectory_slope_per_donor,
    test_trajectory_slope_significance,
)
from cytokine_mil.data.label_encoder import CytokineLabel

# Ground-truth benchmark pairs (from CLAUDE.md — 11 curated pairs)
# Format: (upstream, downstream)
# Names must match the label encoder exactly (e.g. IFN-alpha1 not IFN-alpha;
# IL-17E not IL-25 — these are the same cytokine, the dataset uses IL-17E).
BENCHMARK_PAIRS = [
    ("IL-12", "IFN-gamma"),
    ("IL-1-beta", "IL-6"),
    ("IL-18", "IFN-gamma"),
    ("IFN-alpha1", "IFN-gamma"),   # IFN-alpha1 in this dataset
    ("IL-33", "IL-13"),
    ("IL-17E", "IL-13"),           # IL-17E == IL-25
    ("TSLP", "IL-13"),
    ("IL-2", "IFN-gamma"),
    ("IL-15", "IFN-gamma"),
    ("IL-6", "IL-10"),
    ("TNF-alpha", "IL-6"),
]


def load_seed_data(seed_dir):
    stage3_pkl = os.path.join(seed_dir, "dynamics_stage3.pkl")
    label_enc_path = os.path.join(seed_dir, "label_encoder.json")
    pbs_ct_means_path = os.path.join(seed_dir, "pbs_ct_means.pkl")
    manifest_train_path = os.path.join(seed_dir, "manifest_train.json")

    with open(stage3_pkl, "rb") as f:
        dynamics = pickle.load(f)

    label_encoder = CytokineLabel.load(label_enc_path)

    with open(pbs_ct_means_path, "rb") as f:
        pbs_ct_means = pickle.load(f)

    with open(manifest_train_path) as f:
        train_manifest = json.load(f)

    train_donors = sorted(set(e["donor"] for e in train_manifest))
    train_donors = [d for d in train_donors if d not in ("Donor2", "Donor3")]

    return dynamics, label_encoder, pbs_ct_means, train_donors


def inspect_structure(dynamics, seed):
    ct = dynamics.get("centroid_trajectory", [])
    cle = dynamics.get("centroid_logged_epochs", [])
    print(f"\n[Seed {seed}] centroid_logged_epochs: {cle}")
    print(f"[Seed {seed}] n_snapshots: {len(ct)}")
    if ct:
        snap = ct[0]
        k0 = list(snap.keys())[0]
        print(f"[Seed {seed}] example key: {k0}, embed shape: {snap[k0].shape}")
    records = dynamics.get("records", [])
    val_records = dynamics.get("val_records", [])
    if records:
        final_p = np.mean([r["p_correct_trajectory"][-1] for r in records])
        print(f"[Seed {seed}] final train p_correct (mean over tubes): {final_p:.4f}")
    if val_records:
        final_val_p = np.mean([r["p_correct_trajectory"][-1] for r in val_records])
        print(f"[Seed {seed}] final val   p_correct (mean over tubes): {final_val_p:.4f}")


def analyze_benchmark_recovery(slope_sig, label_encoder):
    """Check how many benchmark pairs are recovered by the slope signal."""
    print("\n" + "=" * 60)
    print("BENCHMARK PAIR RECOVERY — TRAJECTORY SLOPE SIGNAL")
    print("=" * 60)

    cytokines = set(label_encoder.cytokines)
    n_found = 0

    for A, B in BENCHMARK_PAIRS:
        # Try to find these cytokines in the label encoder
        A_match = A if A in cytokines else None
        B_match = B if B in cytokines else None

        if A_match is None or B_match is None:
            print(f"  {A} -> {B}: CYTOKINE NOT IN ENCODER")
            continue

        call = slope_sig["cascade_call_slope"].get((A_match, B_match), "none")
        q = slope_sig["q_pair_slope"].get((A_match, B_match), 1.0)
        relay = slope_sig["relay_T_slope"].get((A_match, B_match), None)
        print(f"  {A_match} -> {B_match}: call={call}, q={q:.4f}, relay={relay}")
        if call in ("A->B", "shared"):
            n_found += 1

    print(f"\nRecovered {n_found}/{len(BENCHMARK_PAIRS)} benchmark pairs")
    return n_found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory",
    )
    parser.add_argument(
        "--output_dir",
        default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/analysis",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--direction_mode", default="global")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seed_dirs = sorted(
        d
        for d in [
            os.path.join(args.results_dir, x) for x in os.listdir(args.results_dir)
        ]
        if os.path.isdir(d) and os.path.basename(d).startswith("seed_")
    )

    print(f"Found {len(seed_dirs)} seed directories: {[os.path.basename(d) for d in seed_dirs]}")

    # ---- Per-seed analysis ----
    seed_slope_sigs = []

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        print(f"\n{'=' * 60}")
        print(f"Processing {seed_name}")
        print("=" * 60)

        try:
            dynamics, label_encoder, pbs_ct_means, train_donors = load_seed_data(seed_dir)
        except Exception as e:
            print(f"  ERROR loading {seed_name}: {e}")
            continue

        ct = dynamics.get("centroid_trajectory", [])
        cle = dynamics.get("centroid_logged_epochs", [])

        inspect_structure(dynamics, seed_name)

        if len(ct) < 2:
            print(f"  WARNING: only {len(ct)} centroid snapshots — skipping slope analysis")
            continue

        # Go/no-go: print epoch-0 Stage-3 centroids for IL-12 and IFN-gamma
        snap0 = ct[0]
        il12_keys = [(k, v) for k, v in snap0.items() if k[0] == "IL-12"]
        ifng_keys = [(k, v) for k, v in snap0.items() if k[0] == "IFN-gamma"]
        print(f"  IL-12 cell types at epoch 0: {[k[1] for k, _ in il12_keys]}")
        print(f"  IFN-gamma cell types at epoch 0: {[k[1] for k, _ in ifng_keys]}")

        print(f"  Computing trajectory bias per donor (mode={args.direction_mode})...")
        try:
            b_fwd_result = compute_trajectory_bias_per_donor(
                centroid_trajectory=ct,
                centroid_logged_epochs=cle,
                label_encoder=label_encoder,
                pbs_ct_means=pbs_ct_means,
                train_donors=train_donors,
                direction_mode=args.direction_mode,
            )
        except Exception as e:
            print(f"  ERROR in compute_trajectory_bias_per_donor: {e}")
            import traceback; traceback.print_exc()
            continue

        # Extract the flat (A, B, T, d) -> np.ndarray trajectory dict
        flat_traj = b_fwd_result["b_fwd_trajectory"]
        print(f"  b_fwd_trajectory keys count: {len(flat_traj)}")
        print(f"  Donors seen: {b_fwd_result['donors']}")
        print(f"  Cell types: {b_fwd_result['cell_types']}")
        print(f"  Logged epochs: {b_fwd_result['logged_epochs']}")

        # Go/no-go: IL-12 → IFN-gamma NK-cell slope per donor
        il12_ifng_nk_keys = [k for k in flat_traj if k[0] == "IL-12" and k[1] == "IFN-gamma" and "NK" in k[2]]
        if il12_ifng_nk_keys:
            k = il12_ifng_nk_keys[0]
            traj = flat_traj[k]
            print(f"\n  GO/NO-GO: IL-12->IFN-gamma, cell type={k[2]}, donor={k[3]}")
            print(f"    trajectory (n={len(traj)}): {np.round(traj, 4)}")

        print(f"  Computing trajectory slopes...")
        slopes = compute_trajectory_slope_per_donor(flat_traj)
        print(f"  slopes keys count: {len(slopes)}")

        # Show IL-12 → IFN-gamma NK slopes
        il12_ifng_nk_slope_keys = [k for k in slopes if k[0] == "IL-12" and k[1] == "IFN-gamma" and "NK" in k[2]]
        if il12_ifng_nk_slope_keys:
            k = il12_ifng_nk_slope_keys[0]
            donor_slopes = slopes[k]
            n_pos = sum(1 for v in donor_slopes.values() if v > 0)
            print(f"\n  GO/NO-GO IL-12->IFN-gamma NK slopes: {n_pos}/{len(donor_slopes)} donors positive")
            for d, s in sorted(donor_slopes.items()):
                print(f"    donor={d}: slope={s:.6f}")

        print(f"  Running significance test...")
        try:
            slope_sig = test_trajectory_slope_significance(slopes, label_encoder, alpha=args.alpha)
        except Exception as e:
            print(f"  ERROR in test_trajectory_slope_significance: {e}")
            import traceback; traceback.print_exc()
            continue

        # Save per-seed result
        sig_path = os.path.join(args.output_dir, f"{seed_name}_slope_sig.pkl")
        with open(sig_path, "wb") as f:
            pickle.dump(slope_sig, f)
        print(f"  Saved slope significance to {sig_path}")

        # Benchmark recovery for this seed
        analyze_benchmark_recovery(slope_sig, label_encoder)
        seed_slope_sigs.append((seed_name, slope_sig, label_encoder))

    # ---- Cross-seed summary ----
    if len(seed_slope_sigs) > 1:
        print("\n" + "=" * 60)
        print("CROSS-SEED BENCHMARK RECOVERY SUMMARY")
        print("=" * 60)
        cytokines = set(seed_slope_sigs[0][2].cytokines)
        pair_calls = defaultdict(list)
        for seed_name, sig, le in seed_slope_sigs:
            for A, B in BENCHMARK_PAIRS:
                if A in cytokines and B in cytokines:
                    call = sig["cascade_call_slope"].get((A, B), "none")
                    pair_calls[(A, B)].append(call)

        for (A, B), calls in pair_calls.items():
            n_pos = sum(1 for c in calls if c in ("A->B", "shared"))
            print(f"  {A} -> {B}: {n_pos}/{len(calls)} seeds recalled")

    print("\nDone.")


if __name__ == "__main__":
    main()
