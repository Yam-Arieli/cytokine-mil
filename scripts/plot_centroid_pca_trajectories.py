"""
Visualize centroid trajectories in PCA-2D space for a known cascade pair.

For each cell type: plot how the mean embedding of (source cytokine, cell_type)
and (target cytokine, cell_type) move over Stage-2b training epochs.

Usage:
    python scripts/plot_centroid_pca_trajectories.py \
        --seed_dir results/centroid_trajectory/seed_42 \
        --source "IL-12" \
        --target "IFN-gamma" \
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
    p.add_argument("--seed_dir", default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/seed_42")
    p.add_argument("--source", default="IL-12")
    p.add_argument("--target", default="IFN-gamma")
    p.add_argument("--output_dir", default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/figures")
    p.add_argument("--cell_type", default=None, help="If set, plot only this cell type (large single panel)")
    p.add_argument("--trajectory_type", default="encoder", choices=["encoder", "attn_weighted"],
                   help="Which centroid trajectory to plot: 'encoder' (h_i) or 'attn_weighted' (a_i*h_i)")
    return p.parse_args()


def load_data(seed_dir, trajectory_type="encoder"):
    pkl_path = Path(seed_dir) / "dynamics_stage3.pkl"
    with open(pkl_path, "rb") as f:
        dynamics = pickle.load(f)
    traj_key = "attn_centroid_trajectory" if trajectory_type == "attn_weighted" else "centroid_trajectory"
    traj = dynamics[traj_key]                    # list of dicts: {(cyto, ct, donor): vec}
    epochs = dynamics["centroid_logged_epochs"]  # list of int
    print(f"Loaded [{traj_key}]: {len(traj)} snapshots, epochs: {epochs}")
    return traj, epochs


def get_cell_types_and_cytokines(traj):
    # Keys are 3-tuples: (cytokine, cell_type, donor)
    keys = list(traj[0].keys())
    cell_types = sorted(set(ct for (_, ct, _) in keys))
    cytokines  = sorted(set(cy for (cy, _, _) in keys))
    donors     = sorted(set(d  for (_, _, d)  in keys))
    return cell_types, cytokines, donors


def collect_trajectory(traj, cyto, ct, donors):
    """Return array (n_epochs, dim): mean across donors for (cyto, ct)."""
    vecs = []
    dummy = np.zeros(list(traj[0].values())[0].shape)
    for snap in traj:
        donor_vecs = [snap[(cyto, ct, d)] for d in donors if (cyto, ct, d) in snap]
        if donor_vecs:
            vecs.append(np.mean(donor_vecs, axis=0))
        else:
            vecs.append(dummy)
    return np.array(vecs)   # (n_epochs, dim)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj, epochs = load_data(args.seed_dir, args.trajectory_type)
    cell_types, cytokines, all_donors = get_cell_types_and_cytokines(traj)

    # Use only training donors (exclude val donors D2, D3)
    val_donors = {"Donor2", "Donor3"}
    train_donors = [d for d in all_donors if d not in val_donors]
    print(f"All donors: {all_donors}")
    print(f"Train donors used for averaging: {train_donors}")

    source, target = args.source, args.target

    # Validate
    if source not in cytokines:
        print(f"ERROR: '{source}' not found. Available (first 10): {cytokines[:10]}")
        return
    if target not in cytokines:
        print(f"ERROR: '{target}' not found. Available (first 10): {cytokines[:10]}")
        return

    print(f"Pair: {source} → {target}")
    print(f"Cell types ({len(cell_types)}): {cell_types}")

    # ── Fit a shared PCA on ALL centroids from both cytokines ──────────────────
    all_vecs = []
    for ct in cell_types:
        for cyto in [source, target]:
            traj_mat = collect_trajectory(traj, cyto, ct, train_donors)
            all_vecs.append(traj_mat)
    all_vecs = np.vstack(all_vecs)   # (2 * n_ct * n_epochs, dim)

    pca = PCA(n_components=2)
    pca.fit(all_vecs)
    ev = pca.explained_variance_ratio_
    print(f"PCA fitted on {all_vecs.shape[0]} vectors (dim={all_vecs.shape[1]})")
    print(f"Explained variance: PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    # If a single cell type is requested via --cell_type, plot only that one large
    focus_ct = getattr(args, 'cell_type', None)
    if focus_ct:
        cell_types = [ct for ct in cell_types if ct == focus_ct]
        if not cell_types:
            print(f"ERROR: cell_type '{focus_ct}' not found")
            return

    n_ct   = len(cell_types)
    ncols  = min(4, n_ct)
    nrows  = (n_ct + ncols - 1) // ncols
    fw     = 10 if n_ct == 1 else 5 * ncols
    fh     = 8  if n_ct == 1 else 4 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
    axes_flat = [axes] if n_ct == 1 else (axes.flatten() if nrows > 1 else list(axes))

    # Color maps: source = reds, target = blues — darker = later epoch
    n_ep = len(epochs)
    src_colors = cm.Reds(np.linspace(0.35, 0.95, n_ep))
    tgt_colors = cm.Blues(np.linspace(0.35, 0.95, n_ep))

    for i, ct in enumerate(cell_types):
        ax = axes_flat[i]

        for cyto, colors, label, marker in [
            (source, src_colors, f"{source}  [source →]", "o"),
            (target, tgt_colors, f"{target}  [← target]", "s"),
        ]:
            traj_mat = collect_trajectory(traj, cyto, ct, train_donors)   # (n_ep, dim)
            pts = pca.transform(traj_mat)                    # (n_ep, 2)

            # Trajectory line (thin, semi-transparent)
            ax.plot(pts[:, 0], pts[:, 1], '-', color=colors[n_ep // 2], alpha=0.4, lw=1.2)

            # Epoch dots: color encodes time (light→early, dark→late)
            ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=40, zorder=4,
                       edgecolors='none', label=label)

            # Start marker (open circle, larger)
            ax.scatter(pts[0, 0], pts[0, 1], s=120, facecolors='white',
                       edgecolors=colors[0], linewidths=2, zorder=5)
            # End marker (star)
            ax.scatter(pts[-1, 0], pts[-1, 1], s=180, marker='*',
                       color=colors[-1], zorder=5)

            # Label first and last epoch
            for ep_idx, ep_label in [(0, f"ep{epochs[0]}"), (-1, f"ep{epochs[-1]}")]:
                ax.annotate(ep_label, (pts[ep_idx, 0], pts[ep_idx, 1]),
                            fontsize=6, alpha=0.75,
                            xytext=(4, 4), textcoords='offset points')

        ax.set_title(ct, fontsize=10, fontweight='bold')
        ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)", fontsize=8)
        ax.legend(fontsize=7, loc='best', framealpha=0.7)
        ax.grid(True, alpha=0.25)

    # Hide unused panels
    for j in range(n_ct, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Centroid Trajectories in PCA Space  |  {source} (source)  →  {target} (target)\n"
        f"Each dot = one logged epoch  |  Open circle = epoch {epochs[0]}  |  Star = epoch {epochs[-1]}\n"
        f"Colour darkness encodes training time (light = early, dark = late)\n"
        f"PCA fitted jointly on both cytokines across all cell types  |  Seed 42",
        fontsize=11, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    fname = out_dir / f"centroid_pca_traj_{source.replace('-','_')}_{target.replace('-','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fname}")
    plt.close()


if __name__ == "__main__":
    main()
