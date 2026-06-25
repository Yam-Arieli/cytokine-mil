#!/usr/bin/env python
"""Synthetic apparatus — TREE cascade (a 4th scenario, comparable to the §30/§32 ladder).

Adds a non-linear (partial-order) cascade to the three ladder scenarios of
`apparatus_cross_asym_ladder.py`, with the SAME hyperparameters (so the four are
directly comparable). The tree:

    p0 -> p1,  p1 -> p2,  p0 -> p3        (p0 root; p1,p3 children; p2 under p1)

and each node carries its OWN block at magnitude m PLUS half of its parent's FULL
(accumulated) program — a pure RETENTION structure (the descendant carries the
ancestor's program), the regime that inverts cross_asym (CLAUDE.md §32):

        B0     B1     B2     B3            (shown at m=1; the run uses m=1.5)
  p0:   1      .      .      .
  p1:   0.5    1      .      .
  p2:   0.25   0.5    1      .
  p3:   0.5    .      .      1

cross_asym is scored on the COMPARABLE (ancestor->descendant) pairs
{p0->p1, p1->p2, p0->p2 (transitive), p0->p3}; the two SIBLING pairs
{p1,p3},{p2,p3} are incomparable (reported separately, expected ~ambiguous).

The 3 ladder scenarios are imported verbatim from `apparatus_cross_asym_ladder` so
the comparison is apples-to-apples (same m, block, n_genes, cells, noise, seed).

Usage (cluster, CPU):
  python scripts/apparatus_tree_cascade.py --output_dir results/vaccine_progression/apparatus_tree
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import apparatus_cross_asym_ladder as APP  # noqa: E402  (same hyperparams + helpers)
from cascadir.progression import accuracy_vs_oracle, kendall_tau, recover_order  # noqa: E402

CONTROL = APP.CONTROL
BLUE, GREY, RED = APP.BLUE, APP.GREY, APP.RED

# tree topology (position -> parent position); 0 is the root
PARENT = {1: 0, 2: 1, 3: 0}
# scramble labels so ALPHABETICAL order != cascade order (same trick as the ladder),
# keeping the symmetric directional_score a genuine ~chance control.
LABEL = {0: "c2", 1: "c4", 2: "c1", 3: "c3"}


def _program_vec(p: int, m: float) -> List[float]:
    """Own block at m + half the parent's full (recursive) program vector."""
    v = [0.0, 0.0, 0.0, 0.0]
    v[p] += m
    if p in PARENT:
        pv = _program_vec(PARENT[p], m)
        for k in range(4):
            v[k] += 0.5 * pv[k]
    return v


def _generate_tree(
    n_cell_types: int = 3, n_genes: int = 320, block: int = 20, m: float = 1.5,
    cells_per: int = 200, noise: float = 0.35, seed: int = 0,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], List[str],
           List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Return (cells_by_pair, conditions[topo order], comparable_oracle, sibling_pairs)."""
    rng = np.random.default_rng(seed)
    blocks = {i: np.arange(i * block, (i + 1) * block) for i in range(4)}
    cell_types = [f"t{i + 1}" for i in range(n_cell_types)]
    cbp: Dict[Tuple[str, str], np.ndarray] = {}
    for p in range(4):
        vec = _program_vec(p, m)
        for t in cell_types:
            X = (noise * rng.standard_normal((cells_per, n_genes))).astype(np.float64)
            for k in range(4):
                if vec[k]:
                    X[:, blocks[k]] += vec[k]
            np.clip(X, 0.0, None, out=X)
            cbp[(LABEL[p], t)] = X
    for t in cell_types:  # control = pure noise
        X = (noise * rng.standard_normal((cells_per, n_genes))).astype(np.float64)
        np.clip(X, 0.0, None, out=X)
        cbp[(CONTROL, t)] = X
    conditions = [LABEL[p] for p in range(4)]                # topo order p0..p3
    edges = [(0, 1), (1, 2), (0, 2), (0, 3)]                 # ancestor->descendant (incl transitive)
    oracle = [(LABEL[u], LABEL[d]) for u, d in edges]        # (upstream, downstream)
    siblings = [(LABEL[1], LABEL[3]), (LABEL[2], LABEL[3])]  # incomparable
    return cbp, conditions, oracle, siblings


def _run_tree(top_k: int, seed: int) -> dict:
    cbp, conds, oracle, siblings = _generate_tree(m=1.5, block=20, seed=seed)
    sigs = APP._discover_signatures(cbp, conds, top_k=top_k)
    stats = APP._pair_stats(cbp, sigs, conds)               # all 6 pairs
    cross_by_pair = {(r.condition_a, r.condition_b): r.cross_asym_median
                     for r in stats.itertuples()}
    ds_by_pair = {(r.condition_a, r.condition_b): r.directional_score_median
                  for r in stats.itertuples()}
    rec = recover_order(cross_by_pair, conds)
    inv = {v: k for k, v in LABEL.items()}
    edge_rows = [{"pair": f"{a}->{b}", "from": inv[a], "to": inv[b],
                  "cross_asym": cross_by_pair.get(tuple(sorted((a, b))), float("nan"))}
                 for (a, b) in oracle]
    sib_rows = [{"pair": f"{a} vs {b}", "cross_asym": cross_by_pair.get(tuple(sorted((a, b))), float("nan"))}
                for (a, b) in siblings]
    return {
        "scenario": "tree", "mode": "tree", "seed_strength": 0.5, "top_k": top_k,
        "cross_accuracy": accuracy_vs_oracle(cross_by_pair, oracle),
        "dirscore_accuracy": accuracy_vs_oracle(ds_by_pair, oracle),
        "recovered_order": rec, "true_order": conds,
        "kendall_tau": kendall_tau(rec, conds),
        "edges": edge_rows, "siblings": sib_rows, "stats": stats,
    }


def _plot(results: List[dict], save_path: Path):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    names = [r["scenario"] for r in results]
    cross = [r["cross_accuracy"] for r in results]
    ds = [r["dirscore_accuracy"] for r in results]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.7 * len(names) + 2.5, 4.2))
    ax.bar(x - w / 2, cross, w, label="cross_asym (direction)", color=BLUE)
    ax.bar(x + w / 2, ds, w, label="directional_score (symmetric control)", color=GREY)
    ax.axhline(0.5, ls="--", c=RED, lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("accuracy vs planted (directed) pairs"); ax.set_ylim(0, 1.05)
    ax.set_title("Apparatus + tree: cross_asym recovers planted direction?")
    ax.legend(fontsize=8, loc="upper right")
    for xi, c in zip(x, cross):
        ax.text(xi - w / 2, c + 0.02, f"{c:.0%}", ha="center", fontsize=8)
    fig.savefig(save_path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="results/vaccine_progression/apparatus_tree")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # 3 ladder scenarios — identical to apparatus_cross_asym_ladder.py
    ladders = [
        APP._run_scenario("distinct", "distinct", seed_strength=0.5, top_k=args.block, seed=args.seed),
        APP._run_scenario("monotone_noseed", "monotone", seed_strength=0.0, top_k=2 * args.block, seed=args.seed),
        APP._run_scenario("monotone_seeded", "monotone", seed_strength=0.5, top_k=2 * args.block, seed=args.seed),
    ]
    tree = _run_tree(top_k=args.block, seed=args.seed)
    scenarios = ladders + [tree]

    _plot(scenarios, out / "apparatus_tree_accuracy.pdf")
    for r in scenarios:
        r["stats"].to_csv(out / f"stats_{r['scenario']}.csv", index=False)

    verdict = {"scenarios": [{k: v for k, v in r.items() if k != "stats"} for r in scenarios]}
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=list))

    # report
    lines = [
        "# Apparatus + TREE cascade — cross_asym on a retention-structured tree (§32 follow-up)",
        "",
        "Tree `p0->p1, p1->p2, p0->p3`; each node = own block (m) + **half its parent's full",
        "program** (a RETENTION structure: the descendant carries the ancestor). Same",
        "hyperparameters as the 3 ladder scenarios (m=1.5, block=20, 320 genes, 3 cell types,",
        "200 cells/group, noise=0.35, seed=0), so the four are directly comparable.",
        "",
        "| scenario | cross_asym acc | dirscore acc (control) | recovered order | Kendall τ |",
        "|---|---|---|---|---|",
    ]
    for r in scenarios:
        lines.append(
            f"| {r['scenario']} | {r['cross_accuracy']:.0%} | {r['dirscore_accuracy']:.0%} | "
            f"{'→'.join(r['recovered_order'])} | {r['kendall_tau']:+.2f} |")
    lines += ["", "## Tree — per-edge cross_asym (comparable ancestor→descendant pairs)",
              "Correct = POSITIVE (upstream=alphabetically-first only by chance; sign vs the",
              "directed edge is what `cross_accuracy` scores).", "",
              "| edge (true upstream→downstream) | cross_asym |", "|---|---|"]
    for e in tree["edges"]:
        lines.append(f"| p{e['from']}→p{e['to']}  ({e['pair']}) | {e['cross_asym']:+.3f} |")
    lines += ["", "## Tree — sibling (incomparable) pairs, expected ~ambiguous (≈0)", "",
              "| pair | cross_asym |", "|---|---|"]
    for s in tree["siblings"]:
        lines.append(f"| {s['pair']} | {s['cross_asym']:+.3f} |")
    lines += ["",
              "**Reading:** this tree is a pure *retention* structure (descendant carries",
              "ancestor), so — consistent with the §32 state-axis finding — `cross_asym` is",
              "expected to mis-sign the ancestor→descendant edges while leaving siblings",
              "ambiguous. See the table for the actual signs.", "",
              f"Figure: `apparatus_tree_accuracy.pdf`. Generated by `scripts/apparatus_tree_cascade.py` (seed={args.seed})."]
    rp = Path("reports/vaccine_progression/APPARATUS_TREE_RESULTS.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(lines) + "\n")

    print(json.dumps(verdict, indent=2, default=list))
    print(f"\nReport: {rp}  Figure: {out / 'apparatus_tree_accuracy.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
