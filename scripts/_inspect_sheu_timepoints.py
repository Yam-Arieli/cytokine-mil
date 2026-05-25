#!/usr/bin/env python3
"""Inspect the Sheu samptag metadata file for available time points and
cells-per-(time_point × pseudo_donor) before building extra pseudotubes."""
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

META = Path("/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw"
            "/GSE224518_samptag.all_cellannotations_metadata.txt.gz")

print(f">>> Reading {META} ...")
df = pd.read_csv(str(META), sep="\t", compression="gzip")
print(f"  shape={df.shape}")
print(f"  columns: {list(df.columns)}")

for c in ("timept", "time_point", "type", "cytokine", "pseudo_donor", "batch"):
    if c in df.columns:
        u = sorted(df[c].dropna().astype(str).unique())
        print(f"  unique {c} ({len(u)}): {u[:30]}")

# Build a "pseudo_donor" column the same way the adapter does (type × replicate)
# so the cross-tab matches the actual training distribution.
type_col = "type" if "type" in df.columns else None
rep_col = next((c for c in df.columns if c.lower() in ("replicate", "rep")), None)
if type_col and rep_col:
    df["pseudo_donor"] = df[type_col].astype(str) + "_" + df[rep_col].astype(str)

# Identify time-point column
tp_col = "timept" if "timept" in df.columns else ("time_point" if "time_point" in df.columns else None)
cyt_col = "stim" if "stim" in df.columns else ("cytokine" if "cytokine" in df.columns else None)

if tp_col and cyt_col:
    print(f"\n>>> Cells per ({tp_col} × {cyt_col}):")
    print(df.groupby([tp_col, cyt_col]).size().unstack(fill_value=0).to_string())

if tp_col and "pseudo_donor" in df.columns:
    print(f"\n>>> Cells per ({tp_col} × pseudo_donor):")
    print(df.groupby([tp_col, "pseudo_donor"]).size().unstack(fill_value=0).to_string())
