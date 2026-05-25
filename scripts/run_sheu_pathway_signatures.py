#!/usr/bin/env python3
"""
Pathway-signature cascade analysis on Sheu 2024 phase-1 pseudotubes (§23).

Replaces empirical top-DE signatures (which collapse on a diagonal in the
500-gene targeted panel) with literature-curated, adaptor-specific gene sets.
Computes cascade penetration per (pathway, A-stimulus, cell-type) triple.

Outputs (under --out_dir):
    resolved_pathways.json          — which curated genes were found / missing
    penetration_long.parquet        — per (pathway, primary, A, cell_type)
    ifnar_binary_summary.csv        — pre-registered IFNAR binary test per cell type
    magnitude_lps_tnf.csv           — LPS > TNF > PBS on NF-κB test per cell type
    plots/
        penetration_heatmap.pdf
        pathway_strip_<pathway>.pdf
        ifnar_binary_summary.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.eda_pair_benchmark import load_phase1_cells
from cytokine_mil.analysis.pathway_signatures import (
    PATHWAY_SIGNATURES,
    STIMULUS_PRIMARY_PATHWAYS,
    IFNAR_POSITIVE_STIMULI,
    IFNAR_NEGATIVE_STIMULI,
    PREREG_CASCADE_TESTS,
    resolve_all_pathways,
    pick_control_genes,
    compute_all_penetrations,
    ifnar_binary_test,
    magnitude_cascade_test,
    run_preregistered_battery,
)
from cytokine_mil.analysis.pathway_plots import (
    plot_penetration_heatmap,
    plot_pathway_score_strip,
    plot_ifnar_binary_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json",
    )
    parser.add_argument(
        "--out_dir",
        default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_pathway",
    )
    parser.add_argument("--min_cells", type=int, default=10)
    parser.add_argument("--n_ctrl", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] manifest={args.manifest}", flush=True)
    cells_by_pair, gene_names = load_phase1_cells(args.manifest)
    print(f"[load] {len(cells_by_pair)} (cytokine, cell_type) buckets, "
          f"{len(gene_names)} genes", flush=True)

    # 1) Resolve pathways against the panel
    print("\n[resolve] curated pathway genes vs Sheu panel:", flush=True)
    resolved = resolve_all_pathways(gene_names)
    resolved_summary = {}
    for p, r in resolved.items():
        ok = r["ok"]
        msg = f"  {p:18s} ok={ok}  found={len(r['found'])}/" \
              f"{len(PATHWAY_SIGNATURES[p]['up'])}  " \
              f"missing={r['missing']}"
        print(msg, flush=True)
        resolved_summary[p] = {
            "ok": bool(ok),
            "found": list(r["found"]),
            "missing": list(r["missing"]),
            "primary_for": PATHWAY_SIGNATURES[p]["primary_for"],
            "cascade_from": PATHWAY_SIGNATURES[p]["cascade_from"],
        }
    with open(out_dir / "resolved_pathways.json", "w") as f:
        json.dump(resolved_summary, f, indent=2)

    # 2) Pre-compute pathway index map (used by control gene selection)
    rng = np.random.default_rng(args.seed)
    pathway_idx_map = {p: r["idx"] for p, r in resolved.items() if r["ok"]}
    if not pathway_idx_map:
        print("[fail] No pathways resolved with sufficient gene hits — aborting.")
        return
    control_idx = pick_control_genes(args.n_ctrl, pathway_idx_map, len(gene_names), rng=rng)
    print(f"\n[ctrl] picked {len(control_idx)} non-pathway control genes", flush=True)

    # 3) Compute all penetrations
    print("\n[compute] cascade penetration across pathways x cell types ...", flush=True)
    penetration_df, _ = compute_all_penetrations(
        cells_by_pair, gene_names,
        n_ctrl=args.n_ctrl, min_cells=args.min_cells, seed=args.seed,
    )
    penetration_df.to_parquet(out_dir / "penetration_long.parquet", index=False)
    print(f"[compute] {len(penetration_df)} rows -> penetration_long.parquet", flush=True)

    # 4) Pre-registered IFNAR binary test
    print("\n[test] IFNAR_induced binary: PIC,LPS,LPSlo,IFNb vs P3CSK,CpG,TNF", flush=True)
    binary = ifnar_binary_test(penetration_df)
    binary.to_csv(out_dir / "ifnar_binary_summary.csv", index=False)
    if not binary.empty:
        print(binary.to_string(index=False))
    else:
        print("  (no rows — IFNAR_induced pathway not resolved or no cell types)")

    # 5) Magnitude test for LPS -> TNF on NF-κB (back-compat per-cell-type table)
    print("\n[test] LPS > TNF > PBS on NFkB_canonical (cascade magnitude check)", flush=True)
    mag = magnitude_cascade_test(penetration_df, "NFkB_canonical", "LPS", "TNF")
    mag.to_csv(out_dir / "magnitude_lps_tnf.csv", index=False)
    if not mag.empty:
        print(mag.to_string(index=False))
    else:
        print("  (NFkB_canonical pathway not resolved)")

    # 5b) Pre-registered battery (5 tests: 1 binary IFNAR + 4 magnitude NFkB)
    print("\n[battery] running pre-registered cascade test battery "
          f"({len(PREREG_CASCADE_TESTS)} tests, Bonferroni α={0.05/len(PREREG_CASCADE_TESTS):.4f})",
          flush=True)
    battery = run_preregistered_battery(
        cells_by_pair, resolved, penetration_df,
        alpha=0.05, min_cells=args.min_cells,
    )
    battery["summary"].to_csv(out_dir / "preregistered_battery_summary.csv", index=False)
    if not battery["magnitude_per_test"].empty:
        battery["magnitude_per_test"].to_csv(out_dir / "magnitude_per_test.csv", index=False)
    print("\n[battery] summary:")
    print(battery["summary"].to_string(index=False))
    n_pass_alpha = int(battery["summary"]["pass_alpha"].sum())
    n_pass_bonf = int(battery["summary"]["pass_bonferroni"].sum())
    print(f"\n[battery] {n_pass_alpha}/{len(battery['summary'])} tests pass α=0.05")
    print(f"[battery] {n_pass_bonf}/{len(battery['summary'])} tests pass Bonferroni α=0.05/{len(PREREG_CASCADE_TESTS)}")

    # 6) Plots
    print("\n[plot] penetration heatmap, per-pathway strip plots, binary summary", flush=True)
    if not penetration_df.empty:
        plot_penetration_heatmap(
            penetration_df,
            save_path=str(plots_dir / "penetration_heatmap.pdf"),
        )
    for pathway, r in resolved.items():
        if not r["ok"]:
            continue
        plot_pathway_score_strip(
            cells_by_pair, pathway, r["idx"], control_idx,
            save_path=str(plots_dir / f"pathway_strip_{pathway}.pdf"),
        )
    if not binary.empty:
        plot_ifnar_binary_summary(
            binary, penetration_df,
            save_path=str(plots_dir / "ifnar_binary_summary.pdf"),
        )

    print(f"\n[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
