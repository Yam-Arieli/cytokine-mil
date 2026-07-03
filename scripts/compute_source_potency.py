"""
Compute the per-cytokine source-potency ranking from training dynamics + validate.

Reads existing multiclass dynamics.pkl (Oesinghaus 91-class, one per seed), computes
the curve-shape source-potency score (cytokine_mil.analysis.source_potency), and
validates it against internal ground truth:
  1. directed SOURCE out-degree in the audited coupling graph (headline; dynamics != IG),
  2. undirected coupling degree (width),
  3. pre-registered DEEP vs SHALLOW pools (one-sided permutation),
  4. literature master-regulators (descriptive).

Writes the ranked table + the shape×ceiling scatter (reproduces the early figure with
donor/seed discipline) + the verdict.

Usage:
    python scripts/compute_source_potency.py --base_dir results/attention_dynamics --seeds 42 123 7
"""

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from cytokine_mil.analysis.source_potency import (
    DEEP_POOL, MASTER_REGULATORS, SHALLOW_POOL, graph_coupling_degree, graph_out_degree,
    per_cytokine_metrics, source_potency_table, validate_against_degree,
    validate_deep_vs_shallow,
)

AXES_CSV = REPO / "reports/cascade_pairs/cytokine_axes.csv"
AUDITED_CSV = REPO / "reports/cascade_pairs/cytokine_axes_audited.csv"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default=None, help="dir with seed_<s>/dynamics.pkl")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--dynamics", nargs="+", default=None,
                   help="explicit dynamics.pkl paths (overrides --base_dir/--seeds)")
    p.add_argument("--axes_csv", default=str(AXES_CSV))
    p.add_argument("--audited_csv", default=str(AUDITED_CSV))
    p.add_argument("--ceiling_floor", type=float, default=0.1)
    p.add_argument("--plateau_frac", type=float, default=0.9)
    p.add_argument("--exclude", nargs="+", default=["PBS"])
    p.add_argument("--out", default=str(REPO / "reports/source_potency/SOURCE_POTENCY_RESULTS.md"))
    return p.parse_args()


def _dyn_paths(args):
    if args.dynamics:
        return [Path(p) for p in args.dynamics]
    if args.base_dir:
        return [Path(args.base_dir) / f"seed_{s}" / "dynamics.pkl" for s in args.seeds]
    sys.exit("Provide --base_dir or --dynamics")


def _read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = _parse_args()

    # --- per-seed metrics, then average across seeds ---
    seed_metrics = []
    for dp in _dyn_paths(args):
        if not dp.exists():
            print(f"skip (missing): {dp}"); continue
        with open(dp, "rb") as f:
            d = pickle.load(f)
        recs, epochs = d.get("records") or [], d.get("logged_epochs") or []
        if not recs or not epochs:
            print(f"skip (no records/epochs): {dp}"); continue
        seed_metrics.append(per_cytokine_metrics(
            recs, epochs, exclude=args.exclude, plateau_frac=args.plateau_frac))
        print(f"loaded {dp}: {len(recs)} records, {len(epochs)} epochs")
    if not seed_metrics:
        sys.exit("No usable dynamics.pkl found.")

    keys = ("P_max", "normalized_auc", "plateau_epoch", "late_gain")
    all_cyts = sorted({c for m in seed_metrics for c in m})
    avg = {}
    for c in all_cyts:
        vals = {k: [m[c][k] for m in seed_metrics if c in m] for k in keys}
        avg[c] = {k: float(np.nanmean(vals[k])) if vals[k] else float("nan") for k in keys}
    table = source_potency_table(avg, ceiling_floor=args.ceiling_floor)

    # --- ground-truth degrees ---
    coupling_deg = graph_coupling_degree(_read_csv_rows(args.axes_csv))
    out_deg = graph_out_degree(_read_csv_rows(args.audited_csv))
    potency = {c: table[c]["source_potency"] for c in table}

    # --- validation ---
    v_out = validate_against_degree(potency, out_deg)
    v_coup = validate_against_degree(potency, coupling_deg)
    v_ds = validate_deep_vs_shallow(potency)

    # --- figure: shape (normalized_auc) x ceiling (P_max), colored by pool ---
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    _scatter(table, outp.parent / "shape_vs_ceiling.png")
    _bar(table, outp.parent / "source_potency_ranking.png")

    # --- report ---
    ranked = sorted([c for c in table if table[c]["included"]],
                    key=lambda c: table[c]["source_potency"], reverse=True)
    L = ["# Source-potency ranking from training dynamics\n\n",
         f"Seeds: {len(seed_metrics)} · cytokines scored (P_max ≥ {args.ceiling_floor}): "
         f"{len(ranked)} of {len(all_cyts)} · plateau_frac={args.plateau_frac}\n\n",
         "Score: `source_potency = z(1 − normalized_traj_auc) + z(late_phase_gain)` "
         "(higher = plateaus later / keeps learning late = richer cascade source). "
         "Read only among cytokines above the ceiling floor (unlearnable-late confound).\n\n",
         "## Validation\n",
         f"- **Directed SOURCE out-degree** (headline, dynamics≠IG): "
         f"Spearman ρ = **{v_out['rho']:.3f}** (n={v_out['n']})\n",
         f"- **Coupling degree (width)**: Spearman ρ = **{v_coup['rho']:.3f}** (n={v_coup['n']})\n",
         f"- **DEEP vs SHALLOW** (pre-registered): mean Δ = {v_ds['obs_diff']:.3f}, "
         f"one-sided p = **{v_ds['p']:.4f}** (deep n={v_ds['n_a']}, shallow n={v_ds['n_b']})\n\n",
         "## Master-regulator ranks (literature sanity)\n"]
    for mr in MASTER_REGULATORS:
        if mr in ranked:
            L.append(f"- {mr}: rank {ranked.index(mr)+1}/{len(ranked)} "
                     f"(potency={table[mr]['source_potency']:+.2f})\n")
        else:
            L.append(f"- {mr}: not scored (below ceiling floor or absent)\n")
    L.append("\n## Ranked source-potency (included cytokines)\n")
    L.append("| rank | cytokine | source_potency | P_max | norm_auc | plateau_ep | late_gain | "
             "out_deg | coup_deg | pool |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for i, c in enumerate(ranked):
        r = table[c]
        pool = "DEEP" if c in DEEP_POOL else ("SHALLOW" if c in SHALLOW_POOL else "")
        L.append(f"| {i+1} | {c} | {r['source_potency']:+.2f} | {r['P_max']:.3f} | "
                 f"{r['normalized_auc']:.3f} | {r['plateau_epoch']:.0f} | {r['late_gain']:.3f} | "
                 f"{out_deg.get(c,0)} | {coupling_deg.get(c,0)} | {pool} |\n")
    L.append("\n> Node *magnitude* claim (source-potency), NOT direction/existence/causation. "
             "Complements cascadir's directed edges. Seed-averaged; dynamics pipeline is seed-noisy.\n")
    outp.write_text("".join(L))

    print(f"\nSaved: {outp}")
    print(f"  out-degree ρ={v_out['rho']:.3f} (n={v_out['n']}) | "
          f"coupling ρ={v_coup['rho']:.3f} | deep>shallow p={v_ds['p']:.4f}")
    print(f"  top-5 sources: {ranked[:5]}")


def _scatter(table, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for pool, color, marker in [("other", "0.7", "o"), ("SHALLOW", "tab:blue", "o"),
                                ("DEEP", "tab:red", "^")]:
        members = (SHALLOW_POOL if pool == "SHALLOW" else DEEP_POOL if pool == "DEEP"
                   else [c for c in table if c not in SHALLOW_POOL and c not in DEEP_POOL])
        xs = [table[c]["normalized_auc"] for c in members if c in table]
        ys = [table[c]["P_max"] for c in members if c in table]
        ax.scatter(xs, ys, c=color, marker=marker, s=60 if pool != "other" else 25,
                   label=pool, edgecolors="k" if pool != "other" else "none", linewidths=0.4,
                   alpha=0.9 if pool != "other" else 0.5, zorder=3 if pool != "other" else 1)
        if pool != "other":
            for c in members:
                if c in table:
                    ax.annotate(c, (table[c]["normalized_auc"], table[c]["P_max"]),
                                fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Normalized trajectory AUC  (HIGH = plateaus early = shallow)")
    ax.set_ylabel("P_max ceiling")
    ax.set_title("Source-potency: learning-curve shape × ceiling (deep vs shallow)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _bar(table, path):
    inc = sorted([c for c in table if table[c]["included"]],
                 key=lambda c: table[c]["source_potency"], reverse=True)
    top = inc[:15] + inc[-10:] if len(inc) > 25 else inc
    vals = [table[c]["source_potency"] for c in top]
    colors = ["tab:red" if c in DEEP_POOL else "tab:blue" if c in SHALLOW_POOL else "0.6"
              for c in top]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top)), vals, color=colors)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7)
    ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("source_potency (z-composite)")
    ax.set_title("Cytokine source-potency (red=DEEP, blue=SHALLOW pools)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
