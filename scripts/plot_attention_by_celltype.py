"""
Plot attention budget per cell type over training (CLAUDE.md §33, diagnostic).

x-axis = epoch, y-axis = mean attention a cell type receives (mean of per-cell
attention within the type, averaged across donors AND across all cytokines), one
line per cell type. Visualizes the late-training collapse: a few distinctive
cell types' lines rise while the rest fall.

Reads the saved attention_trajectory.pkl per seed (no retraining).

Usage:
    python scripts/plot_attention_by_celltype.py \
        --base_dir results/attention_dynamics --seeds 42 123 7
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _mean_over_cytokines(traj):
    """traj: {cytokine: {cell_type: array(n_epochs)}} ->
    {cell_type: array(n_epochs)} averaged over all cytokines that contain it."""
    by_ct = {}
    for _cyt, ct_dict in traj.items():
        for ct, arr in ct_dict.items():
            by_ct.setdefault(ct, []).append(np.asarray(arr, dtype=float))
    return {ct: np.mean(np.vstack(v), axis=0) for ct, v in by_ct.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="results/attention_dynamics")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    base = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [s for s in args.seeds
             if (base / f"seed_{s}" / "attention_trajectory.pkl").exists()]
    if not seeds:
        print("No attention_trajectory.pkl found.")
        sys.exit(1)

    fig, axes = plt.subplots(1, len(seeds), figsize=(6.2 * len(seeds), 5.2),
                             squeeze=False, sharey=True)
    # Stable color per cell type across panels.
    all_cts, cmap = None, None

    for ax, seed in zip(axes[0], seeds):
        with open(base / f"seed_{seed}" / "attention_trajectory.pkl", "rb") as f:
            at = pickle.load(f)
        epochs = at["epochs"]
        by_ct = _mean_over_cytokines(at["trajectory"])
        if all_cts is None:
            all_cts = sorted(by_ct, key=lambda c: by_ct[c][-1], reverse=True)
            cmap = {ct: plt.cm.tab20(i % 20) for i, ct in enumerate(all_cts)}
        # rank by final value for labeling the top lines
        ranked = sorted(by_ct, key=lambda c: by_ct[c][-1], reverse=True)
        for ct in ranked:
            ax.plot(epochs, by_ct[ct], color=cmap.get(ct, "gray"), lw=1.4,
                    label=ct)
        # annotate the 4 highest-final lines
        for ct in ranked[:4]:
            ax.annotate(ct, (epochs[-1], by_ct[ct][-1]), fontsize=7,
                        xytext=(3, 0), textcoords="offset points", va="center")
        ax.set_title(f"seed {seed}", fontsize=10)
        ax.set_xlabel("epoch")
    axes[0][0].set_ylabel("mean attention per cell type\n(avg over cells, donors, cytokines)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=6, ncol=1,
               bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Attention budget per cell type over training "
                 "(rise = collapse target; mean over all cytokines)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    out = out_dir / "attention_by_celltype_over_epochs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # Also print the top-5 collapse targets per seed (final vs epoch-10).
    for seed in seeds:
        with open(base / f"seed_{seed}" / "attention_trajectory.pkl", "rb") as f:
            at = pickle.load(f)
        epochs = at["epochs"]
        by_ct = _mean_over_cytokines(at["trajectory"])
        e10 = min(range(len(epochs)), key=lambda i: abs(epochs[i] - 10))
        top = sorted(by_ct, key=lambda c: by_ct[c][-1], reverse=True)[:5]
        print(f"\nseed {seed} top-5 by FINAL attention (final / @ep10):")
        for ct in top:
            print(f"  {ct:<22} {by_ct[ct][-1]:.4f} / {by_ct[ct][e10]:.4f}")


if __name__ == "__main__":
    main()
