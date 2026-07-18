"""
Plot per-label training accuracy trajectories (x=epoch, y=p_correct), one line per
cascade_forge label, colored by cascade_size (number of downstream labels, INCLUDING
transitive downstream-of-downstream -- the full reachable set, not just direct out_degree).

Visualizes the ceiling-effect finding directly: does more downstream reach correlate with
a later/lower trajectory, or does everything saturate together regardless of cascade role?

Usage (run where results/cascade_forge_potency/seed_*/dynamics.pkl live, i.e. on the cluster):
    python scripts/plot_cascade_forge_trajectories.py \
        --seeds_dir results/cascade_forge_potency --seeds 42 123 7 \
        --out reports/cascade_forge_potency/label_trajectories.png
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from cascade_forge.graph import CascadeGraph
from cytokine_mil.analysis.dynamics import aggregate_to_donor_level

LARGE_CASCADES = {
    "A": {"B": (0.75, 2.0)}, "B": {"C": (0.65, 2.0)}, "C": {"D": (0.55, 2.0)},
    "E": {"F": (0.70, 1.5)}, "F": {"G": (0.60, 1.5)},
    "H": {"I": (0.70, 1.0), "J": (0.60, 2.0), "K": (0.50, 3.0)},
    "L": {"N": (0.65, 1.0)}, "M": {"N": (0.60, 1.0)},
    "O": {"P": (0.60, 1.0)}, "P": {"O": (0.45, 1.0)},
}
ISOLATED_LABELS = ("Q", "R", "S", "T")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds_dir", default="results/cascade_forge_potency")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--exclude", nargs="+", default=["PBS"])
    p.add_argument("--zoom_epochs", type=int, default=50,
                    help="right-panel x-limit (everything saturates fast)")
    p.add_argument("--out", default=str(REPO / "reports/cascade_forge_potency/label_trajectories.png"))
    return p.parse_args()


def main():
    args = _parse_args()
    graph = CascadeGraph.from_dict(LARGE_CASCADES, isolated_labels=ISOLATED_LABELS)
    reach_by_src = {}
    for a, b in graph.reachable:
        reach_by_src.setdefault(a, set()).add(b)
    cascade_size = {lab: len(reach_by_src.get(lab, set())) for lab in graph.labels}

    per_seed_traj = []
    epochs_ref = None
    for s in args.seeds:
        dp = Path(args.seeds_dir) / f"seed_{s}" / "dynamics.pkl"
        if not dp.exists():
            print(f"skip (missing): {dp}"); continue
        with open(dp, "rb") as fh:
            d = pickle.load(fh)
        recs, epochs = d.get("records") or [], d.get("logged_epochs") or []
        if not recs or not epochs:
            print(f"skip (no records/epochs): {dp}"); continue
        epochs_ref = epochs
        donor_trajs = aggregate_to_donor_level(recs, "p_correct_trajectory")
        seed_traj = {}
        for cyt, by_donor in donor_trajs.items():
            if cyt in args.exclude:
                continue
            arrs = [np.asarray(v, dtype=np.float64) for v in by_donor.values()]
            n = min(a.size for a in arrs)
            seed_traj[cyt] = np.mean(np.stack([a[:n] for a in arrs]), axis=0)
        per_seed_traj.append(seed_traj)
        print(f"loaded {dp}: {len(seed_traj)} labels, {len(epochs)} epochs")
    if not per_seed_traj:
        sys.exit("No usable dynamics.pkl found.")

    all_labels = sorted({c for st in per_seed_traj for c in st})
    traj = {}
    for c in all_labels:
        arrs = [st[c] for st in per_seed_traj if c in st]
        n = min(a.size for a in arrs)
        traj[c] = np.mean(np.stack([a[:n] for a in arrs]), axis=0)

    vmax = max(cascade_size.values())
    cmap = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, xlim, title in [
        (axes[0], None, "Full training (400 epochs)"),
        (axes[1], args.zoom_epochs, f"Zoomed: first {args.zoom_epochs} epochs"),
    ]:
        for c in all_labels:
            color = cmap(norm(cascade_size.get(c, 0)))
            n = min(len(epochs_ref), traj[c].size)
            ax.plot(epochs_ref[:n], traj[c][:n], color=color, lw=1.4, alpha=0.9)
        if xlim:
            ax.set_xlim(1, xlim)
        ax.set_xlabel("epoch")
        ax.set_title(title)
    axes[0].set_ylabel("p_correct (3-seed mean, donor-aggregated)")

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label("cascade_size (downstream reach, incl. transitive)")

    fig.suptitle("cascade_forge: per-label training accuracy vs cascade depth "
                 "(color = downstream reach)")
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outp}")
    print("cascade_size per label:", cascade_size)


if __name__ == "__main__":
    main()
