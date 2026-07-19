"""
Cheap follow-up (no new training): does an EARLY-epoch rise-rate feature, extracted from
the already-collected binary (label-vs-PBS) dynamics.pkl trajectories, carry cascade-role
information beyond what source_potency/loss_potency already capture -- and does COMBINING
it with them separate true sources from isolated negatives better than either alone?

Motivation: reports/cascade_forge_potency_binary/label_trajectories_zoom1_50.png showed
visible spread across labels in the first ~50 epochs (unlike the near-total flatline over
the full 4000-epoch run). This quantifies that spread with a continuous metric -- mean
p_correct over an early window -- instead of eyeballing curve colors, then tests (a)
whether it alone correlates with cascade_size/out_degree, and (b) whether combining it
(as a simple summed z-score) with source_potency/loss_potency improves the existing
source-vs-leaf-or-isolated and non-isolated-vs-isolated group separations.

This is purely a synthetic-ground-truth calibration step -- it does NOT touch cross_asym/
coupling (the real-data pipeline). Whether an early-rise feature is worth porting over to
combine with cross_asym on real cytokines is a separate decision made AFTER this result.

Usage:
    python scripts/analyze_early_rise_cascade_forge.py \
        --seeds_dir results/cascade_forge_potency_binary --seeds 42 123 7 \
        --early_window 50 \
        --out reports/cascade_forge_potency_binary/EARLY_RISE_ANALYSIS.md
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
from cytokine_mil.analysis.source_potency import (
    per_cytokine_metrics, source_potency_table, validate_against_degree,
    _perm_test_greater, _zscore,
)

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
    p.add_argument("--seeds_dir", default="results/cascade_forge_potency_binary")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--early_window", type=int, default=50,
                    help="number of leading epochs to average p_correct over")
    p.add_argument("--ceiling_floor", type=float, default=0.1)
    p.add_argument("--exclude", nargs="+", default=["PBS"])
    p.add_argument("--out", default=str(REPO / "reports/cascade_forge_potency_binary/EARLY_RISE_ANALYSIS.md"))
    return p.parse_args()


def early_rise_mean(traj: np.ndarray, window: int) -> float:
    """Mean p_correct over the first `window` epochs. HIGH = fast riser, LOW = slow riser."""
    a = np.asarray(traj, dtype=np.float64)
    n = min(window, a.size)
    return float(a[:n].mean()) if n else float("nan")


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
    seed_early = []
    for s in args.seeds:
        dp = Path(args.seeds_dir) / f"seed_{s}" / "dynamics.pkl"
        if not dp.exists():
            print(f"skip (missing): {dp}"); continue
        with open(dp, "rb") as fh:
            d = pickle.load(fh)
        recs, epochs = d.get("records") or [], d.get("logged_epochs") or []
        if not recs or not epochs:
            print(f"skip (no records/epochs): {dp}"); continue
        seed_metrics.append(per_cytokine_metrics(recs, epochs, exclude=args.exclude))

        donor_trajs = aggregate_to_donor_level(recs, "p_correct_trajectory")
        early = {}
        for cyt, by_donor in donor_trajs.items():
            if cyt in args.exclude:
                continue
            arrs = [np.asarray(v, dtype=np.float64) for v in by_donor.values()]
            n = min(a.size for a in arrs)
            traj = np.mean(np.stack([a[:n] for a in arrs]), axis=0)
            early[cyt] = early_rise_mean(traj, args.early_window)
        seed_early.append(early)
        print(f"loaded {dp}: {len(recs)} records, {len(epochs)} epochs")
    if not seed_metrics:
        sys.exit("No usable dynamics.pkl found.")

    keys = ("P_max", "normalized_auc", "plateau_epoch", "late_gain")
    all_labels = sorted({c for m in seed_metrics for c in m})
    avg = {}
    for c in all_labels:
        vals = {k: [m[c][k] for m in seed_metrics if c in m] for k in keys}
        avg[c] = {k: float(np.nanmean(vals[k])) if vals[k] else float("nan") for k in keys}
    table = source_potency_table(avg, ceiling_floor=args.ceiling_floor)
    source_potency = {c: table[c]["source_potency"] for c in table}
    included = [c for c in all_labels if table[c]["included"]]

    early_mean = {c: float(np.mean([e[c] for e in seed_early if c in e])) for c in all_labels}

    # ------------------------------------------------------------------
    # (a) early_rise_mean ALONE vs ground truth
    # ------------------------------------------------------------------
    v_out_alone = validate_against_degree(early_mean, out_degree)
    v_cs_alone = validate_against_degree(early_mean, cascade_size)
    src_a = np.array([early_mean[c] for c in included if is_source.get(c)])
    leaf_a = np.array([early_mean[c] for c in included if not is_source.get(c, False)])
    v_group_alone = _perm_test_greater(src_a, leaf_a) if src_a.size >= 2 and leaf_a.size >= 2 else None
    noniso_a = np.array([early_mean[c] for c in included if not is_isolated.get(c, False)])
    iso_a = np.array([early_mean[c] for c in included if is_isolated.get(c)])
    v_iso_alone = _perm_test_greater(noniso_a, iso_a) if noniso_a.size >= 2 and iso_a.size >= 2 else None

    # ------------------------------------------------------------------
    # (b) combined = z(source_potency) + z(early_rise_mean), among included labels
    # ------------------------------------------------------------------
    sp_z = _zscore(np.array([source_potency[c] for c in included]))
    em_z = _zscore(np.array([early_mean[c] for c in included]))
    combined = {c: float(sp_z[i] + em_z[i]) for i, c in enumerate(included)}

    v_out_comb = validate_against_degree(combined, out_degree)
    v_cs_comb = validate_against_degree(combined, cascade_size)
    src_c = np.array([combined[c] for c in included if is_source.get(c)])
    leaf_c = np.array([combined[c] for c in included if not is_source.get(c, False)])
    v_group_comb = _perm_test_greater(src_c, leaf_c) if src_c.size >= 2 and leaf_c.size >= 2 else None
    noniso_c = np.array([combined[c] for c in included if not is_isolated.get(c, False)])
    iso_c = np.array([combined[c] for c in included if is_isolated.get(c)])
    v_iso_comb = _perm_test_greater(noniso_c, iso_c) if noniso_c.size >= 2 and iso_c.size >= 2 else None

    v_out_sp = validate_against_degree(source_potency, out_degree)
    v_cs_sp = validate_against_degree(source_potency, cascade_size)

    # ------------------------------------------------------------------
    # Scatter: early_rise_mean vs source_potency, colored by role
    # ------------------------------------------------------------------
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for grp, color, marker in [("isolated (no cascade)", "tab:orange", "s"),
                                ("leaf (out_degree=0, in cascade)", "0.6", "o"),
                                ("source (out_degree>0)", "tab:red", "^")]:
        members = ([c for c in included if is_isolated.get(c)] if grp.startswith("isolated") else
                   [c for c in included if not is_isolated.get(c) and not is_source.get(c)] if grp.startswith("leaf") else
                   [c for c in included if is_source.get(c)])
        xs = [early_mean[c] for c in members]
        ys = [source_potency[c] for c in members]
        ax.scatter(xs, ys, c=color, marker=marker, s=70, edgecolors="k", linewidths=0.4,
                   label=grp, alpha=0.9)
        for c in members:
            ax.annotate(c, (early_mean[c], source_potency[c]), fontsize=8,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"early_rise_mean = mean(p_correct) over first {args.early_window} epochs")
    ax.set_ylabel("source_potency (late-plateau composite)")
    ax.set_title("cascade_forge binary: early-rise vs late-plateau, by role")
    ax.legend(); fig.tight_layout()
    fig.savefig(outp.parent / "early_rise_vs_source_potency.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    L = []
    L.append("# Early-rise feature: alone vs combined with source_potency\n\n")
    L.append(f"Seeds: {len(seed_metrics)} · early_window: first {args.early_window} epochs · "
             f"labels scored: {len(included)} of {len(all_labels)}\n\n")
    L.append("early_rise_mean = mean(p_correct) over the first `early_window` epochs of the "
             "SAME binary dynamics.pkl already used for source_potency -- no new training. "
             "combined = z(source_potency) + z(early_rise_mean), z-scored over included labels.\n\n")

    L.append("## (0) source_potency alone (reference, already reported)\n")
    L.append(f"- out_degree rho = **{v_out_sp['rho']:.3f}** (n={v_out_sp['n']})\n")
    L.append(f"- cascade_size rho = **{v_cs_sp['rho']:.3f}** (n={v_cs_sp['n']})\n\n")

    L.append("## (a) early_rise_mean ALONE\n")
    L.append(f"- out_degree rho = **{v_out_alone['rho']:.3f}** (n={v_out_alone['n']})\n")
    L.append(f"- cascade_size rho = **{v_cs_alone['rho']:.3f}** (n={v_cs_alone['n']})\n")
    if v_group_alone:
        L.append(f"- source > leaf-or-isolated: Delta={v_group_alone['obs_diff']:.4f}, "
                 f"p=**{v_group_alone['p']:.4f}** (n_a={v_group_alone['n_a']}, n_b={v_group_alone['n_b']})\n")
    if v_iso_alone:
        L.append(f"- non-isolated > isolated: Delta={v_iso_alone['obs_diff']:.4f}, "
                 f"p=**{v_iso_alone['p']:.4f}** (n_a={v_iso_alone['n_a']}, n_b={v_iso_alone['n_b']})\n")

    L.append("\n## (b) combined = z(source_potency) + z(early_rise_mean)\n")
    L.append(f"- out_degree rho = **{v_out_comb['rho']:.3f}** (n={v_out_comb['n']})\n")
    L.append(f"- cascade_size rho = **{v_cs_comb['rho']:.3f}** (n={v_cs_comb['n']})\n")
    if v_group_comb:
        L.append(f"- source > leaf-or-isolated: Delta={v_group_comb['obs_diff']:.4f}, "
                 f"p=**{v_group_comb['p']:.4f}** (n_a={v_group_comb['n_a']}, n_b={v_group_comb['n_b']})\n")
    if v_iso_comb:
        L.append(f"- non-isolated > isolated: Delta={v_iso_comb['obs_diff']:.4f}, "
                 f"p=**{v_iso_comb['p']:.4f}** (n_a={v_iso_comb['n_a']}, n_b={v_iso_comb['n_b']})\n")

    L.append("\n## Full per-label table\n\n")
    L.append("| label | early_rise_mean | source_potency | combined | out_deg | cascade_size | role |\n"
             "|---|---:|---:|---:|---:|---:|---|\n")
    for c in sorted(included, key=lambda c: combined[c], reverse=True):
        role = "isolated" if is_isolated.get(c) else ("source" if is_source.get(c) else "leaf")
        L.append(f"| {c} | {early_mean[c]:.4f} | {source_potency[c]:+.2f} | {combined[c]:+.2f} | "
                 f"{out_degree.get(c,0)} | {cascade_size.get(c,0)} | {role} |\n")
    L.append("\n(interpretation added by hand after inspecting the numbers above)\n")
    outp.write_text("".join(L))

    print(f"\nSaved: {outp}")
    print(f"source_potency alone: out_degree rho={v_out_sp['rho']:.3f} cascade_size rho={v_cs_sp['rho']:.3f}")
    print(f"early_rise_mean alone: out_degree rho={v_out_alone['rho']:.3f} cascade_size rho={v_cs_alone['rho']:.3f}")
    if v_group_alone:
        print(f"  source>leaf/iso p={v_group_alone['p']:.4f}")
    if v_iso_alone:
        print(f"  non-iso>iso p={v_iso_alone['p']:.4f}")
    print(f"combined: out_degree rho={v_out_comb['rho']:.3f} cascade_size rho={v_cs_comb['rho']:.3f}")
    if v_group_comb:
        print(f"  source>leaf/iso p={v_group_comb['p']:.4f}")
    if v_iso_comb:
        print(f"  non-iso>iso p={v_iso_comb['p']:.4f}")


if __name__ == "__main__":
    main()
