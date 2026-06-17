#!/usr/bin/env python
"""Synthetic GO/NO-GO apparatus for the COVID-progression experiment (CLAUDE.md §30).

Tests whether `cross_asym` recovers a planted PROGRESSION ORDER from a single
cross-sectional snapshot, on cells×genes data, and characterises the MAGNITUDE
CONFOUND that the real COVID severity axis is exposed to.

Three planted scenarios (all: control + ordered conditions g1<g2<...; cells×genes;
several cell types; an upstream condition's cells carry a downstream "seed"):

  1. distinct        — disjoint programs per grade + downstream seed. Discovered
                       signatures are grade-specific. cross_asym MUST recover the
                       order  → HARD GATE (the method works at all).
  2. monotone_noseed — NESTED programs (severe = mild's genes + more, equal
                       magnitude), no seed. Discovered signatures overlap → the
                       magnitude/nesting confound. Characterisation (often flips).
  3. monotone_seeded — nested programs + a real downstream seed in upstream cells.
                       Does the seed rescue direction despite nesting?

For each scenario we report cross_asym accuracy vs the planted order AND the
SYMMETRIC `directional_score` control accuracy (expected ~chance). The gate passes
iff scenario 1 reaches cross_accuracy == 1.0 while its directional_score control is
clearly below. The monotone scenarios calibrate how to read the real COVID result.

Usage:
    python scripts/apparatus_cross_asym_ladder.py --output_dir results/covid_progression/apparatus
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cytokine_mil.analysis.pathway_audit import directional_asymmetry_test
from cascadir.progression import accuracy_vs_oracle, kendall_tau, recover_order

CONTROL = "ctrl"
GREEN, RED, BLUE, GREY = "#2a9d4a", "#c0392b", "#2c6fbb", "#888888"


# ---------------------------------------------------------------------------
# Planted ladder generator (in-memory cells_by_pair)
# ---------------------------------------------------------------------------

def _generate_ladder(
    mode: str,
    n_conditions: int = 4,
    n_cell_types: int = 3,
    n_genes: int = 320,
    block: int = 20,
    m: float = 1.5,
    seed_strength: float = 0.5,
    cells_per: int = 200,
    noise: float = 0.35,
    seed: int = 0,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], List[str], List[Tuple[str, str]]]:
    """Return (cells_by_pair, conditions(ordered upstream→downstream), oracle pairs).

    mode: 'distinct' (disjoint program blocks) or 'monotone' (nested blocks).
    seed_strength: fraction of m of each downstream-exclusive block added to an
        upstream condition's cells (the carried "seed"). 0 = no seed.
    """
    rng = np.random.default_rng(seed)
    # Scramble names so ALPHABETICAL order != planted progression order. This makes
    # the symmetric `directional_score` control a genuine ~chance control (its sign is
    # fixed per canonical pair, so it only aligns with direction when upstream happens
    # to be the alphabetically-first label). The real COVID grades have the same
    # property — alphabetical {Asymptomatic,Critical,Mild,Moderate,Severe} != severity.
    _perms = {3: [1, 2, 0], 4: [1, 3, 0, 2], 5: [2, 4, 0, 3, 1]}
    base = [f"c{i + 1}" for i in range(n_conditions)]
    perm = _perms.get(n_conditions, list(range(n_conditions)))
    conditions = [base[p] for p in perm]  # upstream -> downstream (planted order)
    cell_types = [f"t{i + 1}" for i in range(n_cell_types)]
    pos = {c: i for i, c in enumerate(conditions)}
    # per-position gene block
    blocks = {i: np.arange(i * block, (i + 1) * block) for i in range(n_conditions)}

    def program_genes(c) -> List[int]:
        if mode == "distinct":
            return list(blocks[pos[c]])
        # monotone: nested — all blocks up to and including this position
        return [g for i in range(pos[c] + 1) for g in blocks[i]]

    cells_by_pair: Dict[Tuple[str, str], np.ndarray] = {}
    for c in conditions + [CONTROL]:
        for t in cell_types:
            X = (noise * rng.standard_normal((cells_per, n_genes))).astype(np.float64)
            if c != CONTROL:
                X[:, program_genes(c)] += m
                if seed_strength > 0:
                    # carry downstream-EXCLUSIVE program genes (a real "seed")
                    for d in conditions:
                        if pos[d] <= pos[c]:
                            continue
                        excl = [g for g in program_genes(d) if g not in program_genes(c)]
                        if excl:
                            X[:, excl] += seed_strength * m
            np.clip(X, 0.0, None, out=X)
            cells_by_pair[(c, t)] = X
    oracle = [(a, b) for a, b in combinations(conditions, 2)]  # a is always upstream
    return cells_by_pair, conditions, oracle


def _discover_signatures(
    cells_by_pair, conditions, top_k: int
) -> Dict[str, np.ndarray]:
    """Top-k genes by mean(condition) − mean(control), pooled across cell types.

    Mirrors a 'naive vs control' discovered signature (the realistic, and
    confound-exposed, signature for a monotone-intensity axis)."""
    cts = sorted({t for (_, t) in cells_by_pair})
    ctrl = np.concatenate([cells_by_pair[(CONTROL, t)] for t in cts], axis=0)
    ctrl_mean = ctrl.mean(axis=0)
    sigs = {}
    for c in conditions:
        cc = np.concatenate([cells_by_pair[(c, t)] for t in cts], axis=0)
        eff = cc.mean(axis=0) - ctrl_mean
        sigs[c] = np.argsort(eff)[::-1][:top_k].astype(np.int64)
    return sigs


# ---------------------------------------------------------------------------
# Per-pair cross_asym + symmetric directional_score control
# ---------------------------------------------------------------------------

def _pair_stats(cells_by_pair, sig_idx, conditions) -> pd.DataFrame:
    rows = []
    for a, b in combinations(sorted(conditions), 2):  # canonical a<b (alphabetical)
        df = directional_asymmetry_test(cells_by_pair, sig_idx, A=a, B=b,
                                        P_A=a, P_B=b, pbs_label=CONTROL, min_cells=10)
        if df.empty:
            continue
        cross = float((df["sA_PB_norm"] - df["sB_PA_norm"]).median())
        dscore = float(df["directional_score"].median())
        rows.append({"condition_a": a, "condition_b": b,
                     "cross_asym_median": cross, "directional_score_median": dscore})
    return pd.DataFrame(rows)


def _accuracy(stats: pd.DataFrame, col: str, oracle) -> float:
    by_pair = {(r.condition_a, r.condition_b): getattr(r, col)
               for r in stats.itertuples()}
    return accuracy_vs_oracle(by_pair, oracle)


def _run_scenario(name, mode, seed_strength, top_k, seed) -> dict:
    cbp, conds, oracle = _generate_ladder(mode=mode, seed_strength=seed_strength, seed=seed)
    sigs = _discover_signatures(cbp, conds, top_k=top_k)
    stats = _pair_stats(cbp, sigs, conds)
    cross_by_pair = {(r.condition_a, r.condition_b): r.cross_asym_median
                     for r in stats.itertuples()}
    rec = recover_order(cross_by_pair, conds)
    return {
        "scenario": name, "mode": mode, "seed_strength": seed_strength, "top_k": top_k,
        "cross_accuracy": _accuracy(stats, "cross_asym_median", oracle),
        "dirscore_accuracy": _accuracy(stats, "directional_score_median", oracle),
        "recovered_order": rec,
        "true_order": conds,
        "kendall_tau": kendall_tau(rec, conds),
        "stats": stats,
    }


def _plot(results: List[dict], save_path: Path):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    names = [r["scenario"] for r in results]
    cross = [r["cross_accuracy"] for r in results]
    ds = [r["dirscore_accuracy"] for r in results]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2.5, 4.2))
    ax.bar(x - w / 2, cross, w, label="cross_asym (direction)", color=BLUE)
    ax.bar(x + w / 2, ds, w, label="directional_score (symmetric control)", color=GREY)
    ax.axhline(0.5, ls="--", c=RED, lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("accuracy vs planted order")
    ax.set_ylim(0, 1.05)
    ax.set_title("Apparatus: cross_asym recovers planted order (vs symmetric control)")
    ax.legend(fontsize=8, loc="lower right")
    for xi, c in zip(x, cross):
        ax.text(xi - w / 2, c + 0.02, f"{c:.0%}", ha="center", fontsize=8)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="results/covid_progression/apparatus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenarios = [
        _run_scenario("distinct", "distinct", seed_strength=0.5, top_k=args.block, seed=args.seed),
        _run_scenario("monotone_noseed", "monotone", seed_strength=0.0, top_k=2 * args.block, seed=args.seed),
        _run_scenario("monotone_seeded", "monotone", seed_strength=0.5, top_k=2 * args.block, seed=args.seed),
    ]

    _plot(scenarios, out / "apparatus_accuracy.pdf")
    for r in scenarios:
        r["stats"].to_csv(out / f"stats_{r['scenario']}.csv", index=False)

    gate = scenarios[0]
    passed = bool(gate["cross_accuracy"] == 1.0 and gate["dirscore_accuracy"] <= 0.6)
    verdict = {
        "gate_scenario": "distinct",
        "gate_passed": passed,
        "scenarios": [{k: v for k, v in r.items() if k != "stats"} for r in scenarios],
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=list))

    # human-readable report
    lines = [
        "# Apparatus gate — cross_asym recovers planted progression order (§30)",
        "",
        f"**GATE (distinct-program ladder): {'PASS ✅' if passed else 'FAIL ❌'}**",
        "",
        "The hard gate is that on a distinct-program cascade ladder, `cross_asym`",
        "recovers the planted order (cross_accuracy = 100%) while the symmetric",
        "`directional_score` control does not — proving the direction signal comes",
        "from the antisymmetry, not magnitude. The monotone scenarios characterise the",
        "MAGNITUDE CONFOUND the real COVID severity axis is exposed to.",
        "",
        "| scenario | cross_asym acc | dirscore acc (control) | recovered order | Kendall τ |",
        "|---|---|---|---|---|",
    ]
    for r in scenarios:
        lines.append(
            f"| {r['scenario']} | {r['cross_accuracy']:.0%} | {r['dirscore_accuracy']:.0%} | "
            f"{'→'.join(r['recovered_order'])} | {r['kendall_tau']:+.2f} |")
    lines += [
        "",
        "**Reading for the COVID test:** if the real severity grades behave like",
        "`monotone_noseed`, direction is magnitude-confounded (cross_asym unreliable);",
        "if like `monotone_seeded`/`distinct`, the snapshot carries a genuine",
        "progression seed and `cross_asym` is informative. The real run's cross-vs-",
        "dirscore gap + per-cell-type sign consensus decide which regime holds.",
        "",
        f"Figure: `apparatus_accuracy.pdf`. Generated by `scripts/apparatus_cross_asym_ladder.py` (seed={args.seed}).",
    ]
    report_path = Path("reports/covid_progression/APPARATUS_GATE_RESULTS.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")

    print(json.dumps(verdict, indent=2, default=list))
    print(f"\nGATE {'PASSED' if passed else 'FAILED'}. "
          f"Report: {report_path}  Figure: {out / 'apparatus_accuracy.pdf'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
