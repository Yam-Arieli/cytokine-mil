"""
Aggregate asymmetry matrices across the 15-seed ensemble.

Loads all latent_geometry.pkl files from experiment3_v2/dec*/
under each MIL run directory, averages the asymmetry matrices,
and reports:
  - Top-20 cascade pairs from the averaged matrix
  - Per-pair standard deviation (stability proxy)
  - Spearman rho between individual matrices and the ensemble mean

Usage:
    python scripts/aggregate_ensemble.py \
        --results_dir /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full \
        --exp_name experiment3_v2
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_dir", type=str, required=True,
        help="Top-level results directory containing run_* subdirectories.",
    )
    p.add_argument(
        "--exp_name", type=str, default="experiment3_v2",
        help="Experiment subdirectory name inside each run_dir (default: experiment3_v2).",
    )
    p.add_argument(
        "--top_n", type=int, default=20,
        help="Number of top cascade pairs to report (default: 20).",
    )
    p.add_argument(
        "--min_std_ratio", type=float, default=2.0,
        help="Report pairs where mean/std >= this ratio (signal-to-noise filter).",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    results_dir = Path(args.results_dir)

    pkls = sorted(results_dir.glob(f"*/experiment3_v2/dec*/latent_geometry.pkl"))
    if not pkls:
        # Also accept flat exp_name paths
        pkls = sorted(results_dir.glob(f"*/{args.exp_name}/latent_geometry.pkl"))

    if not pkls:
        print(f"No latent_geometry.pkl files found under {results_dir}/*/{args.exp_name}/")
        return

    print(f"Found {len(pkls)} pkl files:")
    for p in pkls:
        print(f"  {p}")

    mats = []
    cyt_names = None
    for pkl in pkls:
        with open(pkl, "rb") as f:
            r = pickle.load(f)
        m = r["asymmetry"]["asymmetry_matrix"]
        if cyt_names is None:
            cyt_names = r["asymmetry"]["cytokine_names"]
        mats.append(m)

    mats = np.stack(mats, axis=0)  # (n_runs, K, K)

    # --- Exclude PBS from all analysis (no cascade interpretation) ---
    pbs_idx = next((i for i, n in enumerate(cyt_names) if n == "PBS"), None)
    if pbs_idx is not None:
        keep = [i for i in range(len(cyt_names)) if i != pbs_idx]
        mats = mats[:, keep, :][:, :, keep]
        cyt_names = [cyt_names[i] for i in keep]
        print(f"  PBS excluded from analysis (was index {pbs_idx}). "
              f"Remaining cytokines: {len(cyt_names)}")

    mean_mat = mats.mean(axis=0)   # (K, K)
    std_mat  = mats.std(axis=0)    # (K, K)

    K = mean_mat.shape[0]
    mask = ~np.eye(K, dtype=bool)

    print(f"\nEnsemble summary: {len(mats)} runs, K={K} cytokines (PBS excluded)")
    print(f"  off-diag mean asymmetry: {mean_mat[mask].mean():.4f}")
    print(f"  off-diag mean std:       {std_mat[mask].mean():.4f}")

    # Spearman rho of each run vs ensemble mean
    print("\nSpearman rho (each run vs ensemble mean, off-diagonal):")
    mean_flat = mean_mat[mask]
    for i, m in enumerate(mats):
        rho, _ = spearmanr(m[mask], mean_flat)
        print(f"  run {i:2d}: rho={rho:.4f}  [{pkls[i].parts[-3]}/{pkls[i].parts[-2]}]")

    # Top-N pairs from ensemble mean
    rows, cols = np.where(mask)
    flat = mean_mat[mask]
    std_flat = std_mat[mask]
    order = np.argsort(flat)[::-1]

    print(f"\nTop-{args.top_n} cascade pairs (ensemble mean asymmetry, PBS excluded):")
    print(f"  {'Source':<25}  {'Target':<25}  {'mean':>8}  {'std':>7}  {'SNR':>6}")
    print("  " + "-" * 75)
    shown = 0
    for idx in order:
        a, b = rows[idx], cols[idx]
        mean_val = float(flat[idx])
        std_val  = float(std_flat[idx])
        snr = mean_val / std_val if std_val > 0 else float("inf")
        marker = " *" if snr >= args.min_std_ratio else ""
        print(
            f"  {cyt_names[a]:<25}  {cyt_names[b]:<25}  "
            f"{mean_val:8.4f}  {std_val:7.4f}  {snr:6.2f}{marker}"
        )
        shown += 1
        if shown >= args.top_n:
            break

    # Cross-run pair overlap at top-50
    print(f"\nTop-50 pair overlap across all {len(mats)} runs:")
    top50s = []
    for m in mats:
        f = m[mask]
        ord_ = np.argsort(f)[-50:]
        top50s.append(set(zip(rows[ord_].tolist(), cols[ord_].tolist())))

    pair_counts = {}
    for ts in top50s:
        for pair in ts:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    stable_pairs = [(cnt, pair) for pair, cnt in pair_counts.items() if cnt >= len(mats) // 2]
    stable_pairs.sort(reverse=True)
    print(f"  Pairs appearing in >= {len(mats) // 2}/{len(mats)} runs: {len(stable_pairs)}")
    for cnt, (a, b) in stable_pairs[:20]:
        print(
            f"    {cyt_names[a]:<25} -> {cyt_names[b]:<25}  "
            f"in {cnt}/{len(mats)} runs  mean_asym={mean_mat[a,b]:.4f}"
        )


if __name__ == "__main__":
    main()
