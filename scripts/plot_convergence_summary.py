"""
Multi-seed convergence summary for a cytokine cascade pair.

For each cell type: compute Δdist = dist(ep_start) - dist(ep_end)
where dist = L2 distance between source and target centroids in 128D.

Positive Δdist = converged (source moved toward target over training).
Negative Δdist = diverged.

Aggregated across seeds with mean ± SEM bar chart.

Usage:
    python scripts/plot_convergence_summary.py \
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


# Known relay cell types for IL-12 → IFN-γ (for coloring only)
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
    """Mean 128D centroid for (cyto, ct) over train donors at one epoch snapshot."""
    vecs = [snap[(cyto, ct, d)] for d in train_donors if (cyto, ct, d) in snap]
    return np.mean(vecs, axis=0) if vecs else None


def main():
    args      = parse_args()
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_donors = {"Donor2", "Donor3"}
    source, target = args.source, args.target

    seed_dirs = [Path(d) for d in args.seed_dirs]
    all_delta = {}   # cell_type -> list of Δdist across seeds

    cell_types_ref = None
    epochs_ref     = None

    for sd in seed_dirs:
        traj, epochs = load_data(sd, args.trajectory_type)
        cell_types, cytos, donors = get_meta(traj)
        train_donors = [d for d in donors if d not in val_donors]

        if cell_types_ref is None:
            cell_types_ref = cell_types
            epochs_ref     = epochs
        assert cell_types == cell_types_ref

        if source not in cytos or target not in cytos:
            print(f"WARNING: pair not found in {sd.name}, skipping")
            continue

        # ── Fit per-seed PCA on this seed's relevant subspace ──────────────────
        # PCA fitted on all centroids (source + target, all cell types, all epochs)
        # → captures the cascade-relevant variance; filters out noise dimensions.
        pool_vecs = []
        for ct in cell_types:
            for cyto in [source, target]:
                for snap in traj:
                    v = mean_centroid(snap, cyto, ct, train_donors)
                    if v is not None:
                        pool_vecs.append(v)
        pool_arr = np.vstack(pool_vecs)
        pca = PCA(n_components=2)
        pca.fit(pool_arr)

        snap_start = traj[0]    # epoch 10
        snap_end   = traj[-1]   # epoch 100

        for ct in cell_types:
            c_src_start = mean_centroid(snap_start, source, ct, train_donors)
            c_tgt_start = mean_centroid(snap_start, target, ct, train_donors)
            c_src_end   = mean_centroid(snap_end,   source, ct, train_donors)

            if any(v is None for v in [c_src_start, c_tgt_start, c_src_end]):
                continue

            # Project into the cascade-relevant 2D PCA subspace
            p_src_start = pca.transform(c_src_start[None])[0]
            p_tgt_start = pca.transform(c_tgt_start[None])[0]
            p_src_end   = pca.transform(c_src_end[None])[0]

            # Directional score in PCA space: did source move toward target?
            direction = p_tgt_start - p_src_start   # 2D vector toward target
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            movement  = p_src_end - p_src_start     # how source centroid moved in 2D
            dir_score = np.dot(movement, direction) / norm

            all_delta.setdefault(ct, []).append(dir_score)

    print(f"Loaded {len(seed_dirs)} seeds  |  epochs {epochs_ref[0]}→{epochs_ref[-1]}")
    print(f"Cell types: {cell_types_ref}")

    # ── Bar chart ──────────────────────────────────────────────────────────────
    cts    = cell_types_ref
    means  = np.array([np.mean(all_delta.get(ct, [0])) for ct in cts])
    sems   = np.array([np.std(all_delta.get(ct, [0])) / max(len(all_delta.get(ct, [1])), 1)**0.5
                       for ct in cts])
    n_per  = [len(all_delta.get(ct, [])) for ct in cts]

    # Sort by mean convergence descending
    order  = np.argsort(means)[::-1]
    cts_s  = [cts[i] for i in order]
    means_s = means[order]
    sems_s  = sems[order]

    colors = ["#2ecc71" if ct in KNOWN_RELAYS else "#95a5a6" for ct in cts_s]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(cts_s)), means_s, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    ax.errorbar(range(len(cts_s)), means_s, yerr=sems_s,
                fmt='none', color='#2c3e50', capsize=4, linewidth=1.2)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(range(len(cts_s)))
    ax.set_xticklabels(cts_s, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Directional score  =  dot(Δc_src, c_tgt−c_src) / ‖c_tgt−c_src‖   [per-seed PCA 2D]",
                  fontsize=10)
    ax.set_title(
        f"Directional Convergence in PCA Subspace: {source} → {target}  |  {len(seed_dirs)} seeds\n"
        f"Positive = {source} centroid moved TOWARD {target} in cascade-relevant 2D subspace (ep10→ep100)\n"
        f"Per-seed PCA fitted on source+target centroids — filters noise, retains cascade signal\n"
        f"Green = known relay  |  Gray = non-relay / producers",
        fontsize=11, fontweight='bold'
    )
    ax.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#2ecc71', label='Known relay (NK, NKT, CD8, ILC, MAIT)'),
        Patch(facecolor='#95a5a6', label='Non-relay / IL-12 producers'),
    ], fontsize=9, loc='upper right')

    plt.tight_layout()
    n = len(seed_dirs)
    fname = out_dir / f"convergence_summary_{source.replace('-','_')}_{target.replace('-','_')}_{n}seeds.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fname}")
    plt.close()

    # Print table
    print(f"\n{'Cell type':<25}  {'Mean dir_score':>14}  {'SEM':>8}  {'n':>3}  {'relay?':>8}")
    print("-" * 65)
    for ct, m, s, n in zip(cts_s, means_s, sems_s, [n_per[list(cts).index(c)] for c in cts_s]):
        tag = "RELAY" if ct in KNOWN_RELAYS else ""
        print(f"{ct:<25}  {m:>12.4f}  {s:>8.4f}  {n:>3}  {tag:>8}")


if __name__ == "__main__":
    main()
