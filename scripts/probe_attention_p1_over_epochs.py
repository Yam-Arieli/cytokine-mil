"""
Exploratory probe (CLAUDE.md §33, post-hoc): recompute the P1 "attention-primary
recovers known direct responders" metric along the FULL epoch axis, and under
rare-cell-type exclusion — directly from the saved attention_trajectory.pkl
(no checkpoints, no retraining).

Motivation: the locked P1 reads attention-primary only at the FINAL epoch (250),
which is the most attention-collapsed point. The multi-seed result showed attention
collapses onto rare discriminative cell types (cDC/HSPC/ILC/pDC/Plasmablast),
crowding the textbook responders out of the top-3 (seed 123 → 0/5). This probe asks:
  (A) is P1 higher EARLIER in training, before the collapse?
  (B) does excluding the paper-dropped rare cell types recover P1?

"rare" = the 4 cell types Oesinghaus dropped from gene-expression analysis
(<10 cells/condition, Fig S3): pDC, ILC, Granulocyte, Plasmablast. STRICT additionally
drops cDC + HSPC (the dominant collapse targets, though not <10/condition).

This does NOT change the pre-registered P1 (final epoch, all cell types). It is a
diagnostic to decide whether the signal is present underneath the collapse.

Usage:
    python scripts/probe_attention_p1_over_epochs.py \
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

from cytokine_mil.analysis.attention_dynamics import EXPECTED_DOMINANT  # noqa: E402

RARE_PAPER_DROPPED = {"pDC", "ILC", "Granulocyte", "Plasmablast"}
RARE_STRICT = RARE_PAPER_DROPPED | {"cDC", "HSPC"}
KEY = list(EXPECTED_DOMINANT.keys())  # IL-12, IFN-gamma, IL-4, IL-2, TNF-alpha


def _frac_match_at_epoch(traj, epoch_idx, exclude, top_k=3):
    """frac of KEY cytokines whose top-k attention cell types (at epoch_idx,
    after removing `exclude`) include a known direct responder."""
    matched, n = 0, 0
    for cyt in KEY:
        by_ct = traj.get(cyt)
        if not by_ct:
            continue
        n += 1
        ranked = sorted(
            ((ct, float(np.asarray(a)[epoch_idx])) for ct, a in by_ct.items()
             if ct not in exclude),
            key=lambda kv: kv[1], reverse=True,
        )
        top = [ct for ct, _ in ranked[:top_k]]
        if any(ct in set(EXPECTED_DOMINANT[cyt]) for ct in top):
            matched += 1
    return matched / n if n else float("nan")


def _curve(traj, epochs, exclude):
    return np.array([_frac_match_at_epoch(traj, i, exclude) for i in range(len(epochs))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="results/attention_dynamics")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    base = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {"all_celltypes": set(),
                "drop_paper_rare": RARE_PAPER_DROPPED,
                "drop_rare+cDC+HSPC": RARE_STRICT}

    rows = []
    curves = {v: [] for v in variants}     # per-seed curves (aligned epoch index)
    epochs_ref = None
    print("="*78)
    print("P1 (attention-primary top-3 vs known responders) over the epoch axis")
    print("KEY cytokines:", KEY)
    print("="*78)

    for seed in args.seeds:
        pkl = base / f"seed_{seed}" / "attention_trajectory.pkl"
        if not pkl.exists():
            print(f"seed {seed}: MISSING {pkl}")
            continue
        with open(pkl, "rb") as f:
            at = pickle.load(f)
        traj, epochs = at["trajectory"], at["epochs"]
        epochs_ref = epochs
        print(f"\nseed {seed}  ({len(epochs)} epochs, {len(at['cell_types'])} cell types)")
        print(f"  {'variant':<20} {'final':>6} {'best':>6} {'@best_ep':>9} {'@ep10':>6} {'mean':>6}")
        for v, excl in variants.items():
            c = _curve(traj, epochs, excl)
            curves[v].append(c)
            best_i = int(np.nanargmax(c))
            ep10_i = min(range(len(epochs)), key=lambda i: abs(epochs[i] - 10))
            print(f"  {v:<20} {c[-1]:>6.2f} {c[best_i]:>6.2f} {epochs[best_i]:>9d} "
                  f"{c[ep10_i]:>6.2f} {np.nanmean(c):>6.2f}")
            for i, ep in enumerate(epochs):
                rows.append((seed, v, ep, c[i]))

    # Cross-seed mean curves + figure
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4), squeeze=False)
    for ax, (v, _) in zip(axes[0], variants.items()):
        stack = np.vstack(curves[v]) if curves[v] else np.zeros((1, 1))
        for c, seed in zip(curves[v], args.seeds):
            ax.plot(epochs_ref, c, alpha=0.45, label=f"seed {seed}")
        ax.plot(epochs_ref, stack.mean(0), color="k", lw=2.2, label="mean")
        ax.axhline(0.8, ls="--", c="green", lw=0.8); ax.axhline(0.6, ls="--", c="orange", lw=0.8)
        ax.set_title(v, fontsize=10); ax.set_xlabel("epoch"); ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("P1 frac_match (top-3)"); ax.legend(fontsize=7)
    fig.suptitle("P1 over training — by rare-cell-type exclusion (dashed: GREEN 0.8 / AMBER 0.6)")
    fig.tight_layout()
    fig.savefig(out_dir / "p1_over_epochs.png", dpi=150)

    csv = out_dir / "p1_over_epochs.csv"
    with open(csv, "w") as f:
        f.write("seed,variant,epoch,frac_match\n")
        for seed, v, ep, val in rows:
            f.write(f"{seed},{v},{ep},{val:.4f}\n")

    # Cross-seed summary (mean over seeds of: final, best-over-epochs, mean-over-epochs)
    print("\n" + "="*78)
    print("CROSS-SEED MEAN (over seeds):  final-epoch   best-epoch   mean-over-epochs")
    for v in variants:
        stack = np.vstack(curves[v])
        finals = stack[:, -1].mean()
        bests = np.nanmax(stack, axis=1).mean()
        means = np.nanmean(stack, axis=1).mean()
        print(f"  {v:<20} {finals:>10.2f} {bests:>12.2f} {means:>16.2f}")
    print("="*78)
    print(f"\nSaved: {out_dir/'p1_over_epochs.png'}\n       {csv}")


if __name__ == "__main__":
    main()
