"""
Per-seed 2D PCA distance heatmap for a cytokine cascade pair.

For each seed and each cell type:
  - Fit PCA on that seed's centroids (source + target, all cell types, all epochs)
  - In this cascade-relevant 2D space, compute:
      d_start = distance between source and target centroids at ep_start
      d_end   = distance between source and target centroids at ep_end
      delta   = (d_start - d_end) / d_start   [relative convergence, +ve = converged]

Result: 18-cell-type × 10-seed heatmap, rows sorted by mean delta.

Usage:
    python scripts/plot_pca_distance_heatmap.py \
        --seed_dirs results/centroid_trajectory/seed_42 ... \
        --source "IL-12" --target "IFN-gamma" \
        --output_dir results/centroid_trajectory/figures
"""
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path


KNOWN_RELAYS = {"NK", "NK CD56bright", "NKT", "CD8 Memory", "CD8 Naive", "ILC", "MAIT"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs", nargs='+',
                   default=["/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/seed_42"])
    p.add_argument("--source", default="IL-12")
    p.add_argument("--target", default="IFN-gamma")
    p.add_argument("--trajectory_type", default="attn_weighted",
                   choices=["encoder", "attn_weighted"])
    p.add_argument("--output_dir",
                   default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/figures")
    return p.parse_args()


def load_data(seed_dir, trajectory_type):
    pkl_path = Path(seed_dir) / "dynamics_stage3.pkl"
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    traj_key = "attn_centroid_trajectory" if trajectory_type == "attn_weighted" else "centroid_trajectory"
    return d[traj_key], d["centroid_logged_epochs"]


def get_meta(traj):
    keys       = list(traj[0].keys())
    cell_types = sorted(set(ct for (_, ct, _) in keys))
    cytokines  = sorted(set(cy for (cy, _, _) in keys))
    donors     = sorted(set(d  for (_, _, d)  in keys))
    return cell_types, cytokines, donors


def mean_centroid(snap, cyto, ct, train_donors):
    vecs = [snap[(cyto, ct, d)] for d in train_donors if (cyto, ct, d) in snap]
    return np.mean(vecs, axis=0) if vecs else None


def main():
    args      = parse_args()
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_donors = {"Donor2", "Donor3"}
    source, target = args.source, args.target

    seed_dirs = [Path(d) for d in args.seed_dirs]
    seed_names = [sd.name for sd in seed_dirs]   # e.g. ["seed_42", "seed_123", ...]

    cell_types_ref = None
    # delta[ct][seed_idx] = relative convergence in 2D PCA
    results = {}

    for s_idx, sd in enumerate(seed_dirs):
        traj, epochs = load_data(sd, args.trajectory_type)
        cell_types, cytos, donors = get_meta(traj)
        train_donors = [d for d in donors if d not in val_donors]

        if cell_types_ref is None:
            cell_types_ref = cell_types

        if source not in cytos or target not in cytos:
            print(f"WARNING: pair not found in {sd.name}, skipping")
            continue

        # Fit PCA on all centroids from this seed (cascade-relevant subspace)
        pool_vecs = []
        for ct in cell_types:
            for cyto in [source, target]:
                for snap in traj:
                    v = mean_centroid(snap, cyto, ct, train_donors)
                    if v is not None:
                        pool_vecs.append(v)
        pca = PCA(n_components=2)
        pca.fit(np.vstack(pool_vecs))

        snap_start = traj[0]
        snap_end   = traj[-1]

        for ct in cell_types:
            c_src_start = mean_centroid(snap_start, source, ct, train_donors)
            c_tgt_start = mean_centroid(snap_start, target, ct, train_donors)
            c_src_end   = mean_centroid(snap_end,   source, ct, train_donors)
            c_tgt_end   = mean_centroid(snap_end,   target, ct, train_donors)

            if any(v is None for v in [c_src_start, c_tgt_start, c_src_end, c_tgt_end]):
                continue

            # Project into cascade-relevant 2D space
            p_src_s = pca.transform(c_src_start[None])[0]
            p_tgt_s = pca.transform(c_tgt_start[None])[0]
            p_src_e = pca.transform(c_src_end[None])[0]
            p_tgt_e = pca.transform(c_tgt_end[None])[0]

            d_start = np.linalg.norm(p_src_s - p_tgt_s)
            d_end   = np.linalg.norm(p_src_e - p_tgt_e)

            # Relative convergence, clipped to [-3, 3] to avoid blow-up when d_start ≈ 0
            raw_delta = (d_start - d_end) / (d_start + 1e-10)
            delta = float(np.clip(raw_delta, -3.0, 3.0))

            results.setdefault(ct, {})[s_idx] = delta

        print(f"Done: {sd.name}")

    n_seeds = len(seed_dirs)
    cell_types = cell_types_ref

    # Build matrix (n_ct, n_seeds)
    mat = np.full((len(cell_types), n_seeds), np.nan)
    for i, ct in enumerate(cell_types):
        for s_idx in range(n_seeds):
            if ct in results and s_idx in results[ct]:
                mat[i, s_idx] = results[ct][s_idx]

    # Sort rows by mean delta (descending)
    row_means = np.nanmean(mat, axis=1)
    order     = np.argsort(row_means)[::-1]
    mat_s     = mat[order]
    cts_s     = [cell_types[i] for i in order]

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                             gridspec_kw={'width_ratios': [n_seeds, 1.5, 1.5]})

    ax_heat = axes[0]
    ax_bar  = axes[1]
    ax_frac = axes[2]

    vmax = 3.0  # clipped to ±3
    im = ax_heat.imshow(mat_s, aspect='auto', cmap='RdYlGn',
                        vmin=-vmax, vmax=vmax)

    ax_heat.set_xticks(range(n_seeds))
    ax_heat.set_xticklabels([sn.replace('seed_', '') for sn in seed_names],
                             rotation=45, ha='right', fontsize=9)
    ax_heat.set_yticks(range(len(cts_s)))
    ax_heat.set_yticklabels(
        [f"★ {ct}" if ct in KNOWN_RELAYS else ct for ct in cts_s],
        fontsize=9
    )
    ax_heat.set_xlabel("Seed", fontsize=10)
    ax_heat.set_title(
        f"2D PCA relative convergence\n"
        f"(d_start − d_end) / d_start per seed\n"
        f"Green = converged, Red = diverged",
        fontsize=10, fontweight='bold'
    )

    # Annotate cells with value
    for i in range(len(cts_s)):
        for j in range(n_seeds):
            v = mat_s[i, j]
            if not np.isnan(v):
                ax_heat.text(j, i, f"{v:.2f}", ha='center', va='center',
                             fontsize=7, color='black' if abs(v) < 0.5 else 'white')

    plt.colorbar(im, ax=ax_heat, label='relative convergence')

    # Mean ± SEM bar chart
    means = np.nanmean(mat_s, axis=1)
    sems  = np.nanstd(mat_s, axis=1) / np.sqrt(n_seeds)
    colors = ["#2ecc71" if ct in KNOWN_RELAYS else "#95a5a6" for ct in cts_s]

    ax_bar.barh(range(len(cts_s)), means, xerr=sems,
                color=colors, alpha=0.85, edgecolor='white',
                error_kw={'ecolor': '#2c3e50', 'capsize': 3})
    ax_bar.axvline(0, color='black', lw=0.8, ls='--')
    ax_bar.set_yticks(range(len(cts_s)))
    ax_bar.set_yticklabels([])
    ax_bar.set_xlabel("Mean ± SEM", fontsize=9)
    ax_bar.set_title("Mean\nacross seeds", fontsize=9, fontweight='bold')
    ax_bar.grid(axis='x', alpha=0.3)

    # Fraction of seeds showing convergence (sign test)
    frac_converge = np.array([
        np.sum(mat_s[i, :] > 0) / np.sum(~np.isnan(mat_s[i, :]))
        for i in range(len(cts_s))
    ])
    ax_frac.barh(range(len(cts_s)), frac_converge, color=colors,
                 alpha=0.85, edgecolor='white')
    ax_frac.axvline(0.5, color='black', lw=0.8, ls='--')
    ax_frac.set_xlim(0, 1)
    ax_frac.set_xticks([0, 0.5, 1.0])
    ax_frac.set_xticklabels(['0', '5/10', '10/10'], fontsize=8)
    ax_frac.set_yticks(range(len(cts_s)))
    ax_frac.set_yticklabels([])
    ax_frac.set_xlabel("Fraction of seeds\nconverging", fontsize=9)
    ax_frac.set_title("Sign test\n(robust)", fontsize=9, fontweight='bold')
    ax_frac.grid(axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    ax_frac.legend(handles=[
        Patch(facecolor='#2ecc71', label='★ Known relay'),
        Patch(facecolor='#95a5a6', label='Non-relay'),
    ], fontsize=8, loc='lower right')

    fig.suptitle(
        f"Centroid Convergence Heatmap  |  {source} → {target}  |  {n_seeds} seeds\n"
        f"Metric: (d_start − d_end) / d_start  in per-seed cascade-relevant PCA 2D subspace  "
        f"[clipped to ±3 to prevent blow-up when d_start≈0]",
        fontsize=10, fontweight='bold'
    )

    plt.tight_layout()
    n = n_seeds
    fname = out_dir / f"pca_distance_heatmap_{source.replace('-','_')}_{target.replace('-','_')}_{n}seeds.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fname}")
    plt.close()

    # Print table
    print(f"\n{'Cell type':<25}  {'Mean':>8}  {'SEM':>7}  {'relay?':>8}")
    print("-" * 55)
    for ct, m, s in zip(cts_s, means, sems):
        tag = "★ RELAY" if ct in KNOWN_RELAYS else ""
        print(f"{ct:<25}  {m:>8.3f}  {s:>7.3f}  {tag:>8}")


if __name__ == "__main__":
    main()
