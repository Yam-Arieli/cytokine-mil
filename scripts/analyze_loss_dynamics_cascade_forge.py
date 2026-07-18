"""
Recompute source_potency-style shape metrics on LOSS (-log(p_correct)) instead of raw
accuracy, on the cascade_forge dynamics.pkl already on disk -- no retraining needed, since
p_correct_trajectory (the model's softmax probability on the true class) IS the per-tube
cross-entropy loss up to the -log(.) transform.

Motivation: accuracy/p_correct is bounded and saturating -- every cascade_forge label
already reaches p_correct ~0.995-0.998 within the first ~30-50 of 400 epochs
(reports/cascade_forge_potency/label_trajectories.png shows no separation by cascade
depth). Loss does not saturate the same way: distinguishing p=0.995 from p=0.9999 is a
~20x difference in loss but a ~0.5-point difference in accuracy. If a weak secondary
program keeps sharpening confidence after argmax/near-1 accuracy is already reached, loss
has resolution to show it where accuracy structurally cannot. This script tests that,
without presupposing the answer.

Ground truth and validation calls mirror scripts/validate_source_potency_cascade_forge.py
exactly (same CascadeGraph, same validate_against_degree / _perm_test_greater, reused
unchanged) -- only the per-tube trajectory transform and its shape-metric definitions
are new (added here, not to cytokine_mil/analysis/source_potency.py, since this is still
an open exploratory question, not yet a validated default).

Usage:
    python scripts/analyze_loss_dynamics_cascade_forge.py \
        --seeds_dir results/cascade_forge_potency --seeds 42 123 7 \
        --out reports/cascade_forge_potency/LOSS_POTENCY_VALIDATION.md
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
from cytokine_mil.analysis.source_potency import validate_against_degree, _perm_test_greater, _zscore

# numpy>=2.0 renamed np.trapz -> np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

LARGE_CASCADES = {
    "A": {"B": (0.75, 2.0)}, "B": {"C": (0.65, 2.0)}, "C": {"D": (0.55, 2.0)},
    "E": {"F": (0.70, 1.5)}, "F": {"G": (0.60, 1.5)},
    "H": {"I": (0.70, 1.0), "J": (0.60, 2.0), "K": (0.50, 3.0)},
    "L": {"N": (0.65, 1.0)}, "M": {"N": (0.60, 1.0)},
    "O": {"P": (0.60, 1.0)}, "P": {"O": (0.45, 1.0)},
}
ISOLATED_LABELS = ("Q", "R", "S", "T")


# ---------------------------------------------------------------------------
# Loss-shape metrics (mirror source_potency.py's accuracy-shape metrics, but for a
# DECREASING-toward-0 quantity; polarity is flipped and flagged explicitly below).
# ---------------------------------------------------------------------------

def loss_trajectory_from_p_correct(traj, eps=1e-7):
    a = np.clip(np.asarray(traj, dtype=np.float64), eps, 1.0)
    return -np.log(a)


def loss_floor(traj):
    """L_min: lowest loss reached (analog of P_max ceiling, but a FLOOR near 0)."""
    a = np.asarray(traj, dtype=np.float64)
    return float(a.min()) if a.size else float("nan")


def normalized_loss_persistence(traj):
    """AUC of (loss - L_min)/(L_start - L_min), trapz/(n-1), in [0,1].
    HIGH = loss stays elevated longer relative to its own start/floor = drops LATE = deep.
    (Opposite polarity from normalized_trajectory_auc, where HIGH=shallow -- this is loss,
    not accuracy; flagged here per CLAUDE.md's precise-output-labels convention.)"""
    a = np.asarray(traj, dtype=np.float64)
    if a.size < 2:
        return 0.0
    l_min, l_start = a.min(), a[0]
    denom = l_start - l_min
    if denom <= 0:
        return 0.0
    return float(_trapz((a - l_min) / denom) / (a.size - 1))


def late_phase_drop(traj, last_frac=1.0 / 3.0):
    """Drop in loss over the final `last_frac` of training. >0 = still improving late."""
    a = np.asarray(traj, dtype=np.float64)
    if a.size < 2:
        return 0.0
    start = int(np.clip(round((1.0 - last_frac) * (a.size - 1)), 0, a.size - 1))
    if start >= a.size - 1:
        return 0.0
    tail = a[start:]
    return float(tail[0] - tail.min())


def per_cytokine_loss_metrics(records, exclude=None):
    exclude_set = set(exclude or [])
    recs2 = []
    for r in records:
        r2 = dict(r)
        r2["loss_trajectory"] = loss_trajectory_from_p_correct(r["p_correct_trajectory"])
        recs2.append(r2)
    donor_trajs = aggregate_to_donor_level(recs2, "loss_trajectory")
    out = {}
    for cyt, by_donor in donor_trajs.items():
        if cyt in exclude_set:
            continue
        arrs = [np.asarray(v, dtype=np.float64) for v in by_donor.values()]
        n = min(a.size for a in arrs)
        traj = np.mean(np.stack([a[:n] for a in arrs]), axis=0)
        out[cyt] = {
            "traj": traj,
            "L_min": loss_floor(traj),
            "L_start": float(traj[0]) if traj.size else float("nan"),
            "normalized_loss_persistence": normalized_loss_persistence(traj),
            "late_phase_drop": late_phase_drop(traj),
        }
    return out


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds_dir", default="results/cascade_forge_potency")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--exclude", nargs="+", default=["PBS"])
    p.add_argument("--learned_floor", type=float, default=1.0,
                    help="exclude labels whose L_min exceeds this (never learned; "
                         "analog of source_potency's ceiling_floor, flipped for loss)")
    p.add_argument("--out", default=str(REPO / "reports/cascade_forge_potency/LOSS_POTENCY_VALIDATION.md"))
    return p.parse_args()


def main():
    args = _parse_args()
    graph = CascadeGraph.from_dict(LARGE_CASCADES, isolated_labels=ISOLATED_LABELS)
    out_degree = {lab: len(graph.edges.get(lab, {})) for lab in graph.labels}
    reach_by_src = {}
    for a, b in graph.reachable:
        reach_by_src.setdefault(a, set()).add(b)
    cascade_size = {lab: len(reach_by_src.get(lab, set())) for lab in graph.labels}
    is_source = {lab: out_degree[lab] > 0 for lab in graph.labels}
    is_isolated = {lab: lab in ISOLATED_LABELS for lab in graph.labels}

    seed_metrics = []
    for s in args.seeds:
        dp = Path(args.seeds_dir) / f"seed_{s}" / "dynamics.pkl"
        if not dp.exists():
            print(f"skip (missing): {dp}"); continue
        with open(dp, "rb") as fh:
            d = pickle.load(fh)
        recs, epochs = d.get("records") or [], d.get("logged_epochs") or []
        if not recs or not epochs:
            print(f"skip (no records/epochs): {dp}"); continue
        seed_metrics.append(per_cytokine_loss_metrics(recs, exclude=args.exclude))
        print(f"loaded {dp}: {len(recs)} records, {len(epochs)} epochs")
    if not seed_metrics:
        sys.exit("No usable dynamics.pkl found.")

    keys = ("L_min", "L_start", "normalized_loss_persistence", "late_phase_drop")
    all_labels = sorted({c for m in seed_metrics for c in m})
    avg = {}
    seed_traj = {}
    for c in all_labels:
        vals = {k: [m[c][k] for m in seed_metrics if c in m] for k in keys}
        avg[c] = {k: float(np.nanmean(vals[k])) if vals[k] else float("nan") for k in keys}
        seed_traj[c] = np.mean(np.stack([m[c]["traj"] for m in seed_metrics if c in m]), axis=0)

    included = [c for c in all_labels if avg[c]["L_min"] <= args.learned_floor]
    persistence = _zscore(np.array([avg[c]["normalized_loss_persistence"] for c in included]))
    drop = _zscore(np.array([avg[c]["late_phase_drop"] for c in included]))
    loss_potency = {c: float(persistence[i] + drop[i]) for i, c in enumerate(included)}

    v_outdeg = validate_against_degree(loss_potency, out_degree)
    v_cascsize = validate_against_degree(loss_potency, cascade_size)
    source_labels = [c for c in loss_potency if is_source.get(c)]
    leaf_or_iso = [c for c in loss_potency if not is_source.get(c, False)]
    da = np.array([loss_potency[c] for c in source_labels])
    sa = np.array([loss_potency[c] for c in leaf_or_iso])
    v_group = _perm_test_greater(da, sa) if da.size >= 2 and sa.size >= 2 else None
    isolated_only = [c for c in loss_potency if is_isolated.get(c)]
    non_isolated = [c for c in loss_potency if not is_isolated.get(c, False)]
    da2 = np.array([loss_potency[c] for c in non_isolated])
    sa2 = np.array([loss_potency[c] for c in isolated_only])
    v_iso = _perm_test_greater(da2, sa2) if da2.size >= 2 and sa2.size >= 2 else None

    ranked = sorted(included, key=lambda c: loss_potency[c], reverse=True)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)

    # Loss-trajectory plot (log-y, where any late resolution would actually be visible)
    vmax = max(cascade_size.values())
    cmap = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    epochs = np.arange(1, len(next(iter(seed_traj.values()))) + 1)
    for ax, yscale, title in [(axes[0], "linear", "loss, linear scale"),
                              (axes[1], "log", "loss, log scale (late resolution lives here)")]:
        for c in all_labels:
            color = cmap(norm(cascade_size.get(c, 0)))
            ax.plot(epochs, seed_traj[c], color=color, lw=1.4, alpha=0.9)
        ax.set_yscale(yscale)
        ax.set_xlabel("epoch"); ax.set_title(title)
    axes[0].set_ylabel("loss = -log(p_correct)  (3-seed mean, donor-aggregated)")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label("cascade_size (downstream reach, incl. transitive)")
    fig.suptitle("cascade_forge: per-label LOSS dynamics vs cascade depth")
    fig.savefig(outp.parent / "loss_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    L = []
    L.append("# loss_potency (loss-based shape metric) vs the exact cascade_forge ground truth\n\n")
    L.append(f"Seeds: {len(seed_metrics)} · labels scored (L_min <= {args.learned_floor}): "
             f"{len(included)} of {len(all_labels)}\n\n")
    L.append("loss = -log(p_correct) per tube per epoch (exact cross-entropy on the true "
             "class -- no retraining, recomputed from the existing p_correct_trajectory). "
             "normalized_loss_persistence: HIGH = loss stays elevated longer relative to "
             "its own start/floor = drops LATE = deep prediction. late_phase_drop: >0 = "
             "loss kept falling in the final third = still learning late. "
             "loss_potency = z(normalized_loss_persistence) + z(late_phase_drop).\n\n")
    L.append("## Validation\n")
    L.append(f"- **out_degree**: Spearman rho = **{v_outdeg['rho']:.3f}** (n={v_outdeg['n']})\n")
    L.append(f"- **cascade_size**: Spearman rho = **{v_cascsize['rho']:.3f}** (n={v_cascsize['n']})\n")
    if v_group:
        L.append(f"- **source (out_degree>0) > leaf-or-isolated**: mean Delta = "
                 f"{v_group['obs_diff']:.3f}, one-sided p = **{v_group['p']:.4f}** "
                 f"(source n={v_group['n_a']}, leaf/isolated n={v_group['n_b']})\n")
    if v_iso:
        L.append(f"- **non-isolated > isolated negatives**: mean Delta = "
                 f"{v_iso['obs_diff']:.3f}, one-sided p = **{v_iso['p']:.4f}** "
                 f"(non-isolated n={v_iso['n_a']}, isolated n={v_iso['n_b']})\n")
    L.append("\n## Full ranked table\n\n")
    L.append("| rank | label | loss_potency | L_min | L_start | norm_loss_persist | "
             "late_drop | out_deg | cascade_size | role |\n"
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for i, c in enumerate(ranked):
        role = "isolated" if is_isolated.get(c) else ("source" if is_source.get(c) else "leaf")
        r = avg[c]
        L.append(f"| {i+1} | {c} | {loss_potency[c]:+.2f} | {r['L_min']:.5f} | "
                 f"{r['L_start']:.4f} | {r['normalized_loss_persistence']:.3f} | "
                 f"{r['late_phase_drop']:.5f} | {out_degree.get(c,0)} | "
                 f"{cascade_size.get(c,0)} | {role} |\n")
    L.append("\n(interpretation added by hand after inspecting the numbers above)\n")
    outp.write_text("".join(L))

    print(f"\nSaved: {outp}")
    print(f"out_degree rho={v_outdeg['rho']:.3f} (n={v_outdeg['n']})  "
          f"cascade_size rho={v_cascsize['rho']:.3f} (n={v_cascsize['n']})")
    if v_group:
        print(f"source>leaf/isolated p={v_group['p']:.4f} (Delta={v_group['obs_diff']:.3f})")
    if v_iso:
        print(f"non-isolated>isolated p={v_iso['p']:.4f} (Delta={v_iso['obs_diff']:.3f})")


if __name__ == "__main__":
    main()
