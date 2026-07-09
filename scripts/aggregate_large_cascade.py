#!/usr/bin/env python
"""Aggregate large-cascade benchmark outputs into results/cascade_forge_large/RESULTS.md.

Collects every ``*.metrics.json`` (one per forged snapshot/config) plus the per-edge
``*.direction.csv`` and builds the headline tables: direction accuracy vs the symmetric
control per config, the effect_size floor curve, all-vs-receptor, t=3 vs t=6, coupling
false-positive rate on the isolated negatives, and direction accuracy by cascade depth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forge_large_cascade import ISOLATED_LABELS, LARGE_CASCADES  # noqa: E402


def _node_depths():
    """Longest-path depth of each node over the DAG (feedback edges dropped)."""
    import cascade_forge as cf
    g = cf.CascadeGraph.from_dict(LARGE_CASCADES, isolated_labels=ISOLATED_LABELS)
    bidir = {frozenset(p) for p in g.bidirectional}
    dag = [(u, v) for (u, v) in g.direct if frozenset((u, v)) not in bidir]
    depth = {n: 0 for n in g.labels}
    for _ in range(len(g.labels)):
        for u, v in dag:
            depth[v] = max(depth[v], depth[u] + 1)
    return depth, g


def _fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.0%}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default="results/cascade_forge_large")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    root = Path(args.results_dir)
    metric_files = sorted(root.rglob("*.metrics.json"))
    if not metric_files:
        print(f"[aggregate] no *.metrics.json under {root}", file=sys.stderr)
        return 1
    rows = [json.load(open(f)) for f in metric_files]
    df = pd.DataFrame(rows)
    df["effect_size"] = df["effect_size"].astype(float)
    df["snapshot_time"] = df["snapshot_time"].astype(float)
    df = df.sort_values(["responder_mode", "effect_size", "snapshot_time"]).reset_index(drop=True)

    depth, g = _node_depths()

    lines: list[str] = []
    L = lines.append
    L("# Large cascade_forge experiment — results\n")
    L(f"Authored graph: {len(g.labels)} labels ({len(ISOLATED_LABELS)} isolated negatives "
      f"Q,R,S,T), {len(g.direct)} direct edges, feedback pair {g.bidirectional}.")
    L(f"Configs benchmarked: {len(df)} snapshots.\n")

    # 1. Per-config summary
    L("## Per-config summary\n")
    L("| mode | effect | t | cells | cross_acc | dirscore(ctrl) | coupling recall | coupling FP | isolated FP |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        L(f"| {r['responder_mode']} | {r['effect_size']:.2f} | {r['snapshot_time']:.0f} | "
          f"{int(r['n_cells']):,} | {_fmt_pct(r['cross_accuracy'])} | "
          f"{_fmt_pct(r['dirscore_accuracy'])} | {_fmt_pct(r['coupling_recall'])} | "
          f"{_fmt_pct(r['coupling_false_positive_rate'])} | "
          f"{int(r['isolated_false_positives'])}/{int(r['n_isolated_pairs'])} |")
    L("")

    # 2. Effect-size floor (mode=all, latest snapshot)
    tmax = df["snapshot_time"].max()
    floor = df[(df["responder_mode"] == "all") & (df["snapshot_time"] == tmax)]
    if len(floor):
        L(f"## Effect-size floor (mode=all, t={tmax:.0f})\n")
        L("| effect_size | cross_acc | dirscore(ctrl) |")
        L("|---:|---:|---:|")
        for _, r in floor.sort_values("effect_size").iterrows():
            L(f"| {r['effect_size']:.2f} | {_fmt_pct(r['cross_accuracy'])} | "
              f"{_fmt_pct(r['dirscore_accuracy'])} |")
        L("")

    # 3. all vs receptor (eff 0.30)
    cmp = df[df["effect_size"] == 0.30]
    if cmp["responder_mode"].nunique() > 1:
        L("## responder_mode: all vs receptor (effect_size=0.30)\n")
        L("| t | mode | cross_acc | coupling recall | coupling FP |")
        L("|---:|---|---:|---:|---:|")
        for _, r in cmp.sort_values(["snapshot_time", "responder_mode"]).iterrows():
            L(f"| {r['snapshot_time']:.0f} | {r['responder_mode']} | "
              f"{_fmt_pct(r['cross_accuracy'])} | {_fmt_pct(r['coupling_recall'])} | "
              f"{_fmt_pct(r['coupling_false_positive_rate'])} |")
        L("")

    # 4. Direction accuracy by cascade depth (pool per-edge direction.csv over all configs)
    dir_files = sorted(root.rglob("*.direction.csv"))
    if dir_files:
        per_edge = []
        for f in dir_files:
            d = pd.read_csv(f)
            for _, r in d.iterrows():
                up = str(r.get("expected_upstream", ""))
                per_edge.append({"src_depth": depth.get(up, -1),
                                 "correct": bool(r.get("cross_correct", False))})
        ed = pd.DataFrame(per_edge)
        if len(ed):
            by = ed.groupby("src_depth")["correct"].agg(["mean", "count"]).reset_index()
            L("## Direction accuracy by cascade depth (pooled over configs)\n")
            L("| src depth (hops from root) | accuracy | n edge-observations |")
            L("|---:|---:|---:|")
            for _, r in by.iterrows():
                L(f"| {int(r['src_depth'])} | {r['mean']:.0%} | {int(r['count'])} |")
            L("")

    L("_See per-config `*.direction.csv` (per-pair calls) and `*.coupling.csv` "
      "(existence + false positives)._")

    out = Path(args.out) if args.out else (root / "RESULTS.md")
    out.write_text("\n".join(lines))
    print(f"[aggregate] wrote {out} ({len(df)} configs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
