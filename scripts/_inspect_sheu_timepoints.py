#!/usr/bin/env python3
"""One-off inspector: list Sheu raw h5ads and report unique time_point values
and per-(time_point × pseudo_donor) cell counts. Helps decide which time
points are worth building extra pseudotubes for."""
import sys
from collections import Counter
from pathlib import Path

import scanpy as sc

RAW_DIR = Path("/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw")
PSEUDO_DIR = Path("/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes")

# 1. List raw h5ads
print(">>> Candidate raw h5ads under", RAW_DIR)
candidates = sorted(RAW_DIR.rglob("*.h5ad"))
for c in candidates[:8]:
    print(f"  {c}  ({c.stat().st_size/1e6:.1f} MB)")
if not candidates:
    print("  (none — raw is probably .csv/.csv.gz, not h5ad)")

# 2. Try the manifest path's raw — look at one of the pseudotubes' provenance
print("\n>>> Existing Sheu pseudotube directories:")
for p in Path("/cs/labs/mornitzan/yam.arieli/datasets").glob("Sheu*"):
    print(f"  {p}")

# 3. If a "processed" master h5ad exists, scan it for time points
master_candidates = [
    RAW_DIR / "Sheu2024_processed.h5ad",
    RAW_DIR / "Sheu2024_master.h5ad",
] + candidates
for cand in master_candidates:
    if cand.exists() and cand.stat().st_size > 1e6:
        print(f"\n>>> Loading {cand} ...")
        try:
            adata = sc.read_h5ad(str(cand))
            print(f"  shape={adata.shape}")
            print(f"  obs cols: {list(adata.obs.columns)}")
            if "time_point" in adata.obs.columns:
                tp = adata.obs["time_point"].astype(str)
                print(f"  time_point unique: {sorted(tp.unique())}")
                if "pseudo_donor" in adata.obs.columns:
                    cross = adata.obs.groupby(["time_point", "pseudo_donor"]).size().unstack(fill_value=0)
                    print(f"\n  cells per (time_point × pseudo_donor):")
                    print(cross.to_string())
                if "cytokine" in adata.obs.columns:
                    cross2 = adata.obs.groupby(["time_point", "cytokine"]).size().unstack(fill_value=0)
                    print(f"\n  cells per (time_point × cytokine):")
                    print(cross2.to_string())
        except Exception as e:
            print(f"  ERROR loading: {e}")
        break
