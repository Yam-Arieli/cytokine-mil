"""
Visualize centroid trajectories in PCA-2D space for a known cascade pair.

For each cell type: plot how the mean embedding of (source cytokine, cell_type)
and (target cytokine, cell_type) move over Stage-2b training epochs.

Supports multiple seed dirs: PCA is fitted on pooled vectors from all seeds,
giving a shared embedding space. Per-seed trajectories are shown as thin lines;
the cross-seed mean is shown bold.

Usage:
    # Single seed
    python scripts/plot_centroid_pca_trajectories.py \
        --seed_dirs results/centroid_trajectory/seed_42 \
        --source "IL-12" --target "IFN-gamma" \
        --output_dir results/centroid_trajectory/figures

    # Multi-seed (shared PCA)
    python scripts/plot_centroid_pca_trajectories.py \
        --seed_dirs results/centroid_trajectory/seed_42 results/centroid_trajectory/seed_123 \
        --source "IL-12" --target "IFN-gamma" \
        --output_dir results/centroid_trajectory/figures
"""
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs", nargs='+',
                   default=["/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/seed_42"])
    p.add_argument("--source", default="IL-12")
    p.add_argument("--target", default="IFN-gamma")
    p.add_argument("--output_dir",
                   default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/figures")
    p.add_argument("--cell_type", default=None,
                   help="If set, plot only this cell type (large single panel)")
    p.add_argument("--trajectory_type", default="attn_weighted",
                   choices=["encoder", "attn_weighted"],
                   help="Which centroid trajectory to plot")
    return p.parse_args()


def load_data(seed_dir, trajectory_type="attn_weighted"):
    pkl_path = Path(seed_dir) / "dynamics_stage3.pkl"
    with open(pkl_path, "rb") as f:
        dynamics = pickle.load(f)
    traj_key = "attn_centroid_trajectory" if trajectory_type == "attn_weighted" else "centroid_trajectory"
    traj   = dynamics[traj_key]
    epochs = dynamics["centroid_logged_epochs"]
    return traj, epochs


def get_meta(traj):
    """Return sorted (cell_types, cytokines, donors) from trajectory keys."""
    keys = list(traj[0].keys())
    cell_types = sorted(set(ct for (_, ct, _) in keys))
    cytokines  = sorted(set(cy for (cy, _, _) in keys))
    donors     = sorted(set(d  for (_, _, d)  in keys))
    return cell_types, cytokines, donors


def collect_trajectory(traj, cyto, ct, donors):
    """Return (n_epochs, dim): mean across train donors for (cyto, ct)."""
    dummy = np.zeros(list(traj[0].values())[0].shape)
    vecs = []
    for snap in traj:
        dvecs = [snap[(cyto, ct, d)] for d in donors if (cyto, ct, d) in snap]
        vecs.append(np.mean(dvecs, axis=0) if dvecs else dummy)
    return np.array(vecs)


def main():
    args   = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = [Path(d) for d in args.seed_dirs]
    source, target = args.source, args.target
    val_donors = {"Donor2", "Donor3"}

    # ── Load all seeds ─────────────────────────────────────────────────────────
    all_seed_trajs = []
    epochs = None
    cell_types = None

    for sd in seed_dirs:
        traj, ep = load_data(sd, args.trajectory_type)
        cts, cytos, donors = get_meta(traj)
        train_donors = [d for d in donors if d not in val_donors]

        if epochs is None:
            epochs     = ep
            cell_types = cts
        assert ep == epochs, f"Epoch mismatch in {sd}"

        if source not in cytos or target not in cytos:
            print(f"WARNING: {source} or {target} not in {sd}, skipping")
            continue

        all_seed_trajs.append((traj, train_donors))

    n_seeds = len(all_seed_trajs)
    n_ep    = len(epochs)
    print(f"Loaded {n_seeds} seeds  |  epochs: {epochs}")
    print(f"Cell types ({len(cell_types)}): {cell_types}")

    # ── Fit shared PCA on pooled vectors from all seeds ────────────────────────
    pool = []
    for traj, train_donors in all_seed_trajs:
        for ct in cell_types:
            for cyto in [source, target]:
                pool.append(collect_trajectory(traj, cyto, ct, train_donors))
    pool = np.vstack(pool)   # (n_seeds * 2 * n_ct * n_ep, dim)

    pca = PCA(n_components=2)
    pca.fit(pool)
    ev = pca.explained_variance_ratio_
    print(f"Shared PCA fitted on {pool.shape[0]} vectors (dim={pool.shape[1]})")
    print(f"Explained variance: PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%")

    # ── Filter cell types ──────────────────────────────────────────────────────
    focus_ct = args.cell_type
    plot_cts = [ct for ct in cell_types if (not focus_ct or ct == focus_ct)]
    if focus_ct and not plot_cts:
        print(f"ERROR: cell_type '{focus_ct}' not found"); return

    n_ct  = len(plot_cts)
    ncols = min(4, n_ct)
    nrows = (n_ct + ncols - 1) // ncols
    fw    = 10 if n_ct == 1 else 5 * ncols
    fh    = 8  if n_ct == 1 else 4 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
    axes_flat = [axes] if n_ct == 1 else (axes.flatten() if nrows > 1 else list(axes))

    src_colors_bold = cm.Reds(np.linspace(0.4, 0.95, n_ep))
    tgt_colors_bold = cm.Blues(np.linspace(0.4, 0.95, n_ep))

    for i, ct in enumerate(plot_cts):
        ax = axes_flat[i]

        for cyto, colors_bold, label in [
            (source, src_colors_bold, f"{source}  [source →]"),
            (target, tgt_colors_bold, f"{target}  [← target]"),
        ]:
            mid_color = colors_bold[n_ep // 2]

            # Collect per-seed 2D trajectories
            per_seed_pts = []
            for traj, train_donors in all_seed_trajs:
                mat = collect_trajectory(traj, cyto, ct, train_donors)  # (n_ep, dim)
                per_seed_pts.append(pca.transform(mat))                  # (n_ep, 2)

            # Thin per-seed lines (faint)
            for pts in per_seed_pts:
                ax.plot(pts[:, 0], pts[:, 1], '-', color=mid_color,
                        alpha=0.15, lw=0.8)

            # Bold mean trajectory
            mean_pts = np.mean(per_seed_pts, axis=0)   # (n_ep, 2)
            ax.plot(mean_pts[:, 0], mean_pts[:, 1], '-',
                    color=mid_color, alpha=0.7, lw=2.0)

            # Epoch dots on mean (color encodes time)
            ax.scatter(mean_pts[:, 0], mean_pts[:, 1],
                       c=colors_bold, s=45, zorder=4, edgecolors='none', label=label)

            # Start / end markers on mean
            ax.scatter(mean_pts[0, 0], mean_pts[0, 1], s=130,
                       facecolors='white', edgecolors=colors_bold[0],
                       linewidths=2, zorder=5)
            ax.scatter(mean_pts[-1, 0], mean_pts[-1, 1], s=200,
                       marker='*', color=colors_bold[-1], zorder=5)

            # Epoch labels
            for ep_idx, ep_lbl in [(0, f"ep{epochs[0]}"), (-1, f"ep{epochs[-1]}")]:
                ax.annotate(ep_lbl, (mean_pts[ep_idx, 0], mean_pts[ep_idx, 1]),
                            fontsize=6, alpha=0.75,
                            xytext=(4, 4), textcoords='offset points')

        ax.set_title(ct, fontsize=10, fontweight='bold')
        ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)", fontsize=8)
        ax.legend(fontsize=7, loc='best', framealpha=0.7)
        ax.grid(True, alpha=0.25)

    for j in range(n_ct, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if n_seeds == 1:
        seed_label = f"Seed {seed_dirs[0].name.split('_')[-1]}"
        suffix     = f"_{seed_dirs[0].name}"   # e.g. _seed_123
    else:
        seed_label = f"{n_seeds} seeds"
        suffix     = f"_{n_seeds}seeds"

    fig.suptitle(
        f"Attn-Weighted Centroid Trajectories in PCA Space  |  "
        f"{source} (source)  →  {target} (target)\n"
        f"Open circle = epoch {epochs[0]}  |  Star = epoch {epochs[-1]}  |  "
        f"Colour darkness encodes training time\n"
        f"PCA fitted on both cytokines, all cell types  |  {seed_label}",
        fontsize=10, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    fname  = out_dir / f"centroid_pca_traj_{source.replace('-','_')}_{target.replace('-','_')}{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fname}")
    plt.close()


if __name__ == "__main__":
    main()
