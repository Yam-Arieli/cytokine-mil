#!/usr/bin/env python3
"""
Pair-level EDA benchmark on Sheu 2024 phase-1 pseudotubes.

Loads pooled cells per (cytokine, cell_type), computes a battery of
asymmetric and symmetric statistics for every ordered cytokine pair,
runs a permutation null over labeled-pair labels, and generates the
diagnostic plots.

Outputs (under --out_dir):
    bucket_counts.csv               cell counts per (cytokine, cell_type)
    statistics_long.parquet         raw stats per (A, B, cell_type, statistic)
    statistics_summary.parquet      aggregated per (unordered_pair, statistic)
    auc_per_statistic.csv           per-statistic discrimination AUC
    permutation_null.parquet        permutation null distribution
    plots/
        statistic_heatmap.pdf       labeled-pair × statistic z-score heatmap
        auc_bars.pdf                AUC with permutation-null overlay
        signature_scatter/<A>__<B>.pdf
        projection_density/<A>__<B>.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.eda_pair_benchmark import (
    POSITIVE_PAIRS,
    NEGATIVE_PAIRS,
    load_phase1_cells,
    compute_all_pairs,
    aggregate_to_unordered,
    compute_auc_per_statistic,
    permutation_null_auc,
    labeled_pair_status,
)
from cytokine_mil.analysis.eda_pair_plots import (
    plot_statistic_heatmap,
    plot_auc_bars,
    plot_signature_scatter,
    plot_projection_density,
)


SHEU_STIMULI = ["PBS", "LPS", "LPSlo", "P3CSK", "PIC", "TNF", "CpG", "IFNb"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json",
    )
    parser.add_argument(
        "--out_dir",
        default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_eda",
    )
    parser.add_argument("--time_filter", default="",
                        help="Optional. If set (e.g. '3hr'), filter stim cells "
                             "by adata.obs['time_point']. PBS tubes are loaded "
                             "in full regardless. Phase-1 manifest is already "
                             "filtered to {0hr, 3hr} — leaving this empty is fine.")
    parser.add_argument("--min_cells", type=int, default=10)
    parser.add_argument("--n_permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    (plots_dir / "signature_scatter").mkdir(parents=True, exist_ok=True)
    (plots_dir / "projection_density").mkdir(parents=True, exist_ok=True)

    print(f"[load] manifest={args.manifest}", flush=True)
    cells_by_pair, gene_names = load_phase1_cells(
        args.manifest,
        time_filter=args.time_filter if args.time_filter else None,
    )
    print(
        f"[load] {len(cells_by_pair)} (cytokine, cell_type) buckets, "
        f"{len(gene_names)} genes",
        flush=True,
    )

    # Bucket counts (for sanity)
    counts = pd.DataFrame(
        [
            {"cytokine": cyt, "cell_type": ct, "n_cells": int(v.shape[0])}
            for (cyt, ct), v in cells_by_pair.items()
        ]
    ).sort_values(["cytokine", "cell_type"])
    counts.to_csv(out_dir / "bucket_counts.csv", index=False)
    print(f"[load] wrote bucket_counts.csv", flush=True)

    print("[compute] running pair statistics ...", flush=True)
    stats_df = compute_all_pairs(
        cells_by_pair,
        stimuli=SHEU_STIMULI,
        pbs_label="PBS",
        min_cells=args.min_cells,
        seed=args.seed,
    )
    stats_df.to_parquet(out_dir / "statistics_long.parquet", index=False)
    print(f"[compute] {len(stats_df)} rows -> statistics_long.parquet", flush=True)

    summary_df = aggregate_to_unordered(stats_df, agg="mean")
    summary_df.to_parquet(out_dir / "statistics_summary.parquet", index=False)
    print(f"[compute] {len(summary_df)} rows -> statistics_summary.parquet", flush=True)

    print("[auc] per-statistic AUC of labeled positives vs negatives ...", flush=True)
    auc_df = compute_auc_per_statistic(summary_df, statistic_value_col="max")
    auc_df.to_csv(out_dir / "auc_per_statistic.csv", index=False)
    if len(auc_df) > 0:
        print(auc_df.to_string(index=False))
    else:
        print("  (no labeled rows — check pair_label assignment)")

    print(f"[null] permutation null x{args.n_permutations} ...", flush=True)
    null_df = permutation_null_auc(
        summary_df,
        statistic_value_col="max",
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    null_df.to_parquet(out_dir / "permutation_null.parquet", index=False)
    print(f"[null] wrote permutation_null.parquet ({len(null_df)} rows)", flush=True)

    print("[plot] heatmap, AUC bars ...", flush=True)
    plot_statistic_heatmap(
        summary_df,
        save_path=str(plots_dir / "statistic_heatmap.pdf"),
        statistic_value_col="max",
    )
    plot_auc_bars(
        auc_df, null_df,
        save_path=str(plots_dir / "auc_bars.pdf"),
        p_threshold=0.05,
    )

    print("[plot] per-pair signature scatters + projection densities ...", flush=True)
    all_labeled = POSITIVE_PAIRS + NEGATIVE_PAIRS
    for A, B in all_labeled:
        label = labeled_pair_status(A, B)
        slug = f"{A}__{B}"
        plot_signature_scatter(
            cells_by_pair, A, B,
            save_path=str(plots_dir / "signature_scatter" / f"{slug}.pdf"),
            pair_label=label,
        )
        plot_projection_density(
            cells_by_pair, A, B,
            save_path=str(plots_dir / "projection_density" / f"{slug}.pdf"),
            pair_label=label,
        )

    print(f"[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
