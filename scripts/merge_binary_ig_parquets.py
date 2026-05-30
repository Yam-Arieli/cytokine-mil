"""
Concatenate per-cytokine binary IG parquets into a single 24-cytokine
parquet for the Path A → Bridge → Path B pipeline.

Each input parquet has the long-format schema produced by
``scripts/run_binary_ig_probe.py``:
    cytokine | gene | ig | mean_expression | rank_ig

`rank_ig` is per-cytokine (0 = highest IG within that cytokine), so simple
row-concat is the correct merge — no re-ranking, no de-duping by gene.

Usage:
    python scripts/merge_binary_ig_parquets.py INPUT1.parquet INPUT2.parquet ... OUT.parquet

Sanity checks before writing OUT:
    * every input has the required schema
    * cytokine sets are disjoint across inputs (otherwise we'd double-count)
    * gene-name sets agree across inputs (otherwise downstream HVG lookup
      becomes ambiguous)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLS = {"cytokine", "gene", "ig", "mean_expression", "rank_ig"}


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python scripts/merge_binary_ig_parquets.py "
            "INPUT1.parquet INPUT2.parquet ... OUT.parquet",
            flush=True,
        )
        sys.exit(2)

    inputs = sys.argv[1:-1]
    out = sys.argv[-1]

    dfs = []
    seen_cyts: set = set()
    seen_genes: set | None = None

    for p in inputs:
        p = Path(p)
        if not p.exists():
            print(f"FATAL: input does not exist: {p}", flush=True)
            sys.exit(2)
        df = pd.read_parquet(p)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            print(f"FATAL: {p} missing columns: {missing}", flush=True)
            sys.exit(2)
        cyts = set(df["cytokine"].unique())
        overlap = cyts & seen_cyts
        if overlap:
            print(
                f"FATAL: cytokine overlap between inputs ({overlap}); "
                f"refusing to concat (would double-count).",
                flush=True,
            )
            sys.exit(2)
        genes = set(df["gene"].unique())
        if seen_genes is None:
            seen_genes = genes
        else:
            diff = seen_genes ^ genes
            if diff:
                print(
                    f"FATAL: gene-set mismatch between inputs (symmetric "
                    f"difference size = {len(diff)}); inputs must share the "
                    f"same HVG universe.",
                    flush=True,
                )
                sys.exit(2)
        print(
            f"  {p}: {len(df)} rows, "
            f"{len(cyts)} cytokines, {len(genes)} genes",
            flush=True,
        )
        seen_cyts.update(cyts)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    print(
        f"Merged: {len(merged)} rows, {merged['cytokine'].nunique()} cytokines "
        f"-> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
