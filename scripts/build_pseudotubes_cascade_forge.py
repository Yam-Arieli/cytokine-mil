"""
Adapter: forge a cascade_forge snapshot and write it as pseudo-tubes in the
cytokine_mil manifest convention, so the existing Stage1/Stage2 AB-MIL training +
dynamics pipeline can run on it unchanged.

Motivation: the source_potency training-dynamics score (cytokine_mil.analysis.
source_potency) needs a dataset with KNOWN cascade depth/out-degree ground truth to
validate against, instead of noisy real data with only a hand-audited partial graph.
cascade_forge (reports/cascade_forge_simulation_1M_cells_2026-07-09/report.pdf) already
gives exactly that: an authored, locked cascade graph (a depth-3 chain, a shorter chain,
a fan-out, a fan-in, a feedback loop, and 4 ISOLATED negative-control labels with zero
cascade edges -- the cleanest possible "shallow" ground truth). This script reuses the
exact LARGE_CASCADES graph from scripts/forge_large_cascade.py (same ground truth as the
published report) and writes one pseudo-tube per (donor, condition) -- cascade_forge
already produces exactly one tube per (donor, condition) with "PBS" as the control label,
so no stratified resampling or PBS-pooling is needed (unlike the Sheu/ID adapters).

Usage:
    python scripts/build_pseudotubes_cascade_forge.py \
        --out_dir results/cascade_forge_potency/pseudotubes \
        --n_cells_per_tube 2000 --n_donors 10 --effect_size 0.30 --snapshot_time 6 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cascade_forge as cf

# Same locked ground truth as scripts/forge_large_cascade.py (the published 1M-cell report).
LARGE_CASCADES = {
    "A": {"B": (0.75, 2.0)}, "B": {"C": (0.65, 2.0)}, "C": {"D": (0.55, 2.0)},  # deep chain
    "E": {"F": (0.70, 1.5)}, "F": {"G": (0.60, 1.5)},                            # chain
    "H": {"I": (0.70, 1.0), "J": (0.60, 2.0), "K": (0.50, 3.0)},                 # fan-out
    "L": {"N": (0.65, 1.0)}, "M": {"N": (0.60, 1.0)},                            # fan-in
    "O": {"P": (0.60, 1.0)}, "P": {"O": (0.45, 1.0)},                            # feedback
}
ISOLATED_LABELS = ("Q", "R", "S", "T")   # negative controls (no cascade)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", required=True, help="output dir for pseudo-tube h5ads + manifest")
    p.add_argument("--n_cell_types", type=int, default=8)
    p.add_argument("--n_cells_per_tube", type=int, default=2000)
    p.add_argument("--n_donors", type=int, default=10)
    p.add_argument("--effect_size", type=float, default=0.30)
    p.add_argument("--responder_mode", choices=["all", "receptor"], default="all")
    p.add_argument("--snapshot_time", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    tubes_dir = out_dir / "tubes"
    tubes_dir.mkdir(parents=True, exist_ok=True)

    sim = cf.CascadeSimulator(
        LARGE_CASCADES,
        isolated_labels=ISOLATED_LABELS,
        n_cell_types=args.n_cell_types,
        n_cells_per_tube=args.n_cells_per_tube,
        n_donors=args.n_donors,
        effect_size=args.effect_size,
        responder_mode=args.responder_mode,
        output="lognorm",   # already log1p -- matches the "preprocessed" pseudo-tube contract
        sparse=True,        # memory safety; PseudoTubeDataset/CellDataset handle sparse X
        seed=args.seed,
    )
    print(f"[build] {len(sim.graph.labels)} labels ({len(ISOLATED_LABELS)} isolated), "
          f"{len(sim.graph.direct)} direct edges, {sim.model.n_genes} genes | "
          f"mode={args.responder_mode} eff={args.effect_size} t={args.snapshot_time}",
          flush=True)

    result = sim.simulate(snapshot_times=[args.snapshot_time])
    adata = result.adata

    manifest = []
    for donor in sorted(adata.obs["donor"].unique()):
        d_mask = adata.obs["donor"] == donor
        for cond in sorted(adata.obs["condition"].unique()):
            mask = d_mask & (adata.obs["condition"] == cond)
            n = int(mask.sum())
            if n == 0:
                continue
            sub = adata[mask].copy()
            path = tubes_dir / f"{donor}__{cond}.h5ad"
            sub.write_h5ad(path)
            manifest.append({
                "path": str(path),
                "donor": str(donor),
                "cytokine": str(cond),   # field name kept for PseudoTubeDataset/CytokineLabel compat
                "n_cells": n,
                "cell_types_included": sorted(sub.obs["cell_type"].unique().tolist()),
                "tube_idx": 0,
            })
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh)
    print(f"[build] wrote {len(manifest)} pseudo-tubes -> {manifest_path}", flush=True)

    gene_list_path = out_dir / "gene_list.json"
    with open(gene_list_path, "w") as fh:
        json.dump(list(adata.var_names), fh)
    print(f"[build] wrote {len(adata.var_names)} genes (no HVG selection -- full curated "
          f"panel, same convention as the Sheu 500-gene adapter) -> {gene_list_path}", flush=True)

    gt = dict(sim.graph.to_ground_truth())
    gt["isolated_labels"] = list(ISOLATED_LABELS)
    gt["config"] = result.config
    with open(out_dir / "ground_truth.json", "w") as fh:
        json.dump(gt, fh, indent=2)
    print(f"[build] wrote ground_truth.json "
          f"({len(gt['direct_edges'])} direct edges, {len(gt['isolated_labels'])} isolated)",
          flush=True)


if __name__ == "__main__":
    main()
