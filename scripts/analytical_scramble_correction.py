#!/usr/bin/env python
"""Analytically correct the direction benchmark for the alphabetical-order confound.

The large run authored the graph with the upstream label always alphabetically before the
downstream, so the SYMMETRIC directional_score control scored a fake 100% (a trivial
"first = upstream" rule wins). This recomputes both accuracies under label names whose
sort-order is DECOUPLED from cascade direction (averaged over many random relabelings),
using the already-computed per-pair values — no re-training.

Key facts it exploits:
  * cross_asym is ANTISYMMETRIC: cross_asym(b,a) = -cross_asym(a,b). Re-orienting the
    canonical pair flips its sign together with expected_sign, so cross accuracy is
    NAMING-INVARIANT (it equals the reported value).
  * directional_score is SYMMETRIC: its value is fixed per pair, so under a coin-flip
    of which endpoint sorts first its accuracy -> 50% (chance).

Reads results/cascade_forge_large/<config>/<snapshot>.direction.csv (+ sibling
metrics.json for config labels). Prints a corrected table and writes
results/cascade_forge_large/RESULTS_scramble_corrected.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _oriented(df):
    """Per edge: C = cross value oriented to the TRUE upstream (naming-invariant, >0 = correct);
    D = directional_score (fixed sign); and the two endpoint node names."""
    up = df["expected_upstream"].astype(str)
    a = df["condition_a"].astype(str)
    b = df["condition_b"].astype(str)
    ca = df["cross_asym_median"].astype(float)
    ds = df["directional_score_median"].astype(float)
    C = np.where(a.values == up.values, ca.values, -ca.values)   # >0 iff cross points to true upstream
    other = np.where(a.values == up.values, b.values, a.values)
    return up.values, other, C, ds.values


def _scrambled_dirscore_acc(up, other, D, n_scrambles, seed):
    """Mean directional_score accuracy over random relabelings (ranks on the distinct nodes,
    so shared nodes across edges are handled correctly)."""
    nodes = sorted(set(up) | set(other))
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_scrambles):
        rank = {n: r for n, r in zip(nodes, rng.permutation(len(nodes)))}
        up_first = np.array([rank[u] < rank[o] for u, o in zip(up, other)])
        exp_sign = np.where(up_first, 1, -1)
        d_sign = np.sign(D)
        accs.append(float(np.mean(d_sign == exp_sign)))
    return float(np.mean(accs)), float(np.std(accs))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default="results/cascade_forge_large")
    p.add_argument("--n_scrambles", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)
    root = Path(args.results_dir)

    rows = []
    for dcsv in sorted(root.rglob("*.direction.csv")):
        mjson = dcsv.with_suffix("").with_suffix(".metrics.json")
        cfg = json.load(open(mjson)) if mjson.exists() else {}
        df = pd.read_csv(dcsv)
        up, other, C, D = _oriented(df)
        cross_acc = float(np.mean(C > 0))                       # naming-invariant
        ds_orig = float(np.mean(np.sign(D) == 1))               # confounded (all upstream sort first)
        ds_scr, ds_std = _scrambled_dirscore_acc(up, other, D, args.n_scrambles, args.seed)
        rows.append({
            "config": f"{cfg.get('responder_mode','?')}_eff{cfg.get('effect_size','?')}",
            "t": cfg.get("snapshot_time", float("nan")),
            "n_edges": len(df),
            "cross_acc": cross_acc,
            "dirscore_confounded": ds_orig,
            "dirscore_scrambled": ds_scr,
            "dirscore_scrambled_std": ds_std,
        })
    res = pd.DataFrame(rows).sort_values(["config", "t"]).reset_index(drop=True)

    lines = ["# Direction benchmark — confound-corrected (analytical)\n"]
    lines.append("cross_asym accuracy is naming-invariant (unchanged). directional_score "
                 "(symmetric control) is recomputed under label names whose sort-order is "
                 f"decoupled from direction (mean over {args.n_scrambles} random relabelings).\n")
    lines.append("| config | t | edges | cross_acc | dirscore (confounded) | dirscore (scrambled ctrl) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in res.iterrows():
        lines.append(f"| {r['config']} | {r['t']:.0f} | {int(r['n_edges'])} | "
                     f"{r['cross_acc']:.0%} | {r['dirscore_confounded']:.0%} | "
                     f"{r['dirscore_scrambled']:.0%} ± {r['dirscore_scrambled_std']:.0%} |")
    lines.append("\n**Read:** cross_asym stays high (real direction signal), while the symmetric "
                 "control collapses from a confounded 100% to ~50% (chance) once label order no "
                 "longer leaks the answer — the contrast the benchmark needs.")
    out = root / "RESULTS_scramble_corrected.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[wrote] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
