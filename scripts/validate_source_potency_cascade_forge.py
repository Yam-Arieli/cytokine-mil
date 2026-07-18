"""
Validate source_potency (training-dynamics plateau shape) against the EXACT authored
cascade_forge graph -- no proxy pools, no partial hand-audit; the ground truth is the
literal LARGE_CASCADES dict used to generate the data.

Reuses cytokine_mil.analysis.source_potency unchanged (per_cytokine_metrics,
source_potency_table, validate_against_degree). Ground truth per label, derived directly
from cascade_forge.graph.CascadeGraph:
  - out_degree: number of DIRECT downstream targets (mirrors graph_out_degree on real data)
  - cascade_size: number of labels reachable downstream (transitive closure)
  - is_source: out_degree > 0 (has a real downstream cascade -- A,B,C,E,F,H,L,M,O,P)
  - is_isolated: in ISOLATED_LABELS (Q,R,S,T -- zero cascade edges, the clean negative)
The P3-equivalent test: source_potency(is_source) vs source_potency(leaf-or-isolated,
i.e. out_degree==0), one-sided permutation -- same test as validate_deep_vs_shallow but
with the exact ground-truth partition instead of hand-picked real-data pools.

Usage:
    python scripts/validate_source_potency_cascade_forge.py \
        --seeds_dir results/cascade_forge_potency --seeds 42 123 7 \
        --out reports/cascade_forge_potency/SOURCE_POTENCY_VALIDATION.md
"""

import argparse
import json
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
from cytokine_mil.analysis.source_potency import (
    per_cytokine_metrics, source_potency_table, validate_against_degree, _perm_test_greater,
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
    p.add_argument("--seeds_dir", default="results/cascade_forge_potency")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--ceiling_floor", type=float, default=0.1)
    p.add_argument("--exclude", nargs="+", default=["PBS"])
    p.add_argument("--out", default=str(REPO / "reports/cascade_forge_potency/SOURCE_POTENCY_VALIDATION.md"))
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
    print(f"Ground truth: {len(graph.labels)} labels, out_degree={out_degree}")

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
        seed_metrics.append(per_cytokine_metrics(recs, epochs, exclude=args.exclude))
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
    potency = {c: table[c]["source_potency"] for c in table}

    v_outdeg = validate_against_degree(potency, out_degree)
    v_cascsize = validate_against_degree(potency, cascade_size)

    source_labels = [c for c in potency if is_source.get(c) and np.isfinite(potency[c])]
    leaf_or_iso = [c for c in potency if not is_source.get(c, False) and np.isfinite(potency[c])]
    da = np.array([potency[c] for c in source_labels])
    sa = np.array([potency[c] for c in leaf_or_iso])
    v_group = _perm_test_greater(da, sa) if da.size >= 2 and sa.size >= 2 else None

    isolated_only = [c for c in potency if is_isolated.get(c) and np.isfinite(potency[c])]
    non_isolated = [c for c in potency if not is_isolated.get(c, False) and np.isfinite(potency[c])]
    da2 = np.array([potency[c] for c in non_isolated])
    sa2 = np.array([potency[c] for c in isolated_only])
    v_iso = _perm_test_greater(da2, sa2) if da2.size >= 2 and sa2.size >= 2 else None

    ranked = sorted([c for c in table if table[c]["included"]],
                     key=lambda c: table[c]["source_potency"], reverse=True)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for grp, color, marker in [("isolated (no cascade)", "tab:orange", "s"),
                                ("leaf (out_degree=0, in cascade)", "0.6", "o"),
                                ("source (out_degree>0)", "tab:red", "^")]:
        members = ([c for c in all_labels if is_isolated.get(c)] if grp.startswith("isolated") else
                   [c for c in all_labels if not is_isolated.get(c) and not is_source.get(c)] if grp.startswith("leaf") else
                   [c for c in all_labels if is_source.get(c)])
        xs = [table[c]["normalized_auc"] for c in members if c in table]
        ys = [table[c]["P_max"] for c in members if c in table]
        ax.scatter(xs, ys, c=color, marker=marker, s=70, edgecolors="k", linewidths=0.4,
                   label=grp, alpha=0.9)
        for c in members:
            if c in table:
                ax.annotate(c, (table[c]["normalized_auc"], table[c]["P_max"]),
                            fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Normalized trajectory AUC (HIGH = plateaus early = shallow)")
    ax.set_ylabel("P_max ceiling")
    ax.set_title("cascade_forge: source_potency shape x ceiling vs KNOWN ground truth")
    ax.legend(); fig.tight_layout()
    fig.savefig(outp.parent / "cascade_forge_shape_vs_ceiling.png", dpi=150)
    plt.close(fig)

    L = []
    L.append("# source_potency validated against the exact cascade_forge ground truth\n\n")
    L.append(f"Seeds: {len(seed_metrics)} · labels scored (P_max >= {args.ceiling_floor}): "
             f"{len(ranked)} of {len(all_labels)}\n\n")
    L.append("Ground truth (from LARGE_CASCADES, no proxy/pool/audit): "
             "out_degree = direct downstream targets; cascade_size = transitive-closure "
             "reach; is_source = out_degree>0 (A,B,C,E,F,H,L,M,O,P); "
             "isolated = Q,R,S,T (zero cascade edges, true negative controls); "
             "leaf = in-cascade but out_degree=0 (D,G,I,J,K,N).\n\n")
    L.append("## Validation\n")
    L.append(f"- **out_degree** (P1-equivalent): Spearman rho = **{v_outdeg['rho']:.3f}** "
             f"(n={v_outdeg['n']})\n")
    L.append(f"- **cascade_size** (transitive reach): Spearman rho = **{v_cascsize['rho']:.3f}** "
             f"(n={v_cascsize['n']})\n")
    if v_group:
        L.append(f"- **source (out_degree>0) > leaf-or-isolated (out_degree=0)** "
                 f"(P3-equivalent): mean Delta = {v_group['obs_diff']:.3f}, "
                 f"one-sided p = **{v_group['p']:.4f}** (source n={v_group['n_a']}, "
                 f"leaf/isolated n={v_group['n_b']})\n")
    if v_iso:
        L.append(f"- **non-isolated (any cascade role) > isolated negatives**: "
                 f"mean Delta = {v_iso['obs_diff']:.3f}, one-sided p = **{v_iso['p']:.4f}** "
                 f"(non-isolated n={v_iso['n_a']}, isolated n={v_iso['n_b']})\n")
    L.append("\n## Full ranked table\n\n")
    L.append("| rank | label | source_potency | P_max | norm_auc | plateau_ep | late_gain | "
             "out_deg | cascade_size | role |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for i, c in enumerate(ranked):
        r = table[c]
        role = "isolated" if is_isolated.get(c) else ("source" if is_source.get(c) else "leaf")
        L.append(f"| {i+1} | {c} | {r['source_potency']:+.2f} | {r['P_max']:.3f} | "
                 f"{r['normalized_auc']:.3f} | {r['plateau_epoch']:.0f} | {r['late_gain']:.3f} | "
                 f"{out_degree.get(c,0)} | {cascade_size.get(c,0)} | {role} |\n")
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
