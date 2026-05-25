#!/usr/bin/env python3
"""Inspect Oesinghaus manifest + first pseudotube for cytokine names,
cell types, and gene-symbol convention (case, prefix)."""
import json
import sys
from pathlib import Path

MANIFEST = Path("/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json")

m = json.load(open(MANIFEST))
cyts = sorted({e["cytokine"] for e in m})
donors = sorted({e["donor"] for e in m})
print(f"n_entries     = {len(m)}")
print(f"n_cytokines   = {len(cyts)}")
print(f"n_donors      = {len(donors)}")
print(f"\nDonors: {donors}")
print(f"\nCytokines (sorted): {cyts}")

# IFN-related candidates
print(f"\nIFN-related (case-insensitive substring 'ifn' or 'interferon'):")
for c in cyts:
    if "ifn" in c.lower() or "interferon" in c.lower():
        print(f"  '{c}'")

# Sample one pseudotube to see gene-name convention
sample = m[0]["path"]
print(f"\nSample pseudotube: {sample}")
import scanpy as sc
ad = sc.read_h5ad(sample)
print(f"  shape={ad.shape}")
print(f"  obs cols: {list(ad.obs.columns)}")
gene_sample = list(ad.var_names[:30])
print(f"  first 30 genes: {gene_sample}")
n_upper = sum(1 for g in ad.var_names if g.isupper())
print(f"  uppercase genes: {n_upper}/{len(ad.var_names)}")

# Check for canonical ISGs
canonical_isgs = ["ISG15", "MX1", "MX2", "IFIT1", "IFIT2", "IFIT3", "RSAD2",
                  "STAT1", "IRF7", "USP18", "OAS1", "OAS2", "OAS3", "OASL",
                  "CCL5", "CXCL10", "IFNB1"]
nfkb_genes = ["TNF", "IL1B", "IL6", "NFKBIA", "NFKBID", "TNFAIP3", "CXCL1",
              "CXCL2", "CCL3", "CCL4", "BIRC3", "NFKBIE", "NFKBIZ"]
present_isgs = [g for g in canonical_isgs if g in ad.var_names]
missing_isgs = [g for g in canonical_isgs if g not in ad.var_names]
present_nfkb = [g for g in nfkb_genes if g in ad.var_names]
missing_nfkb = [g for g in nfkb_genes if g not in ad.var_names]
print(f"\nCanonical ISGs present ({len(present_isgs)}/{len(canonical_isgs)}): {present_isgs}")
print(f"Canonical ISGs missing: {missing_isgs}")
print(f"\nNF-κB genes present ({len(present_nfkb)}/{len(nfkb_genes)}): {present_nfkb}")
print(f"NF-κB genes missing: {missing_nfkb}")

# Cell types
if "cell_type" in ad.obs.columns:
    cts = sorted(ad.obs["cell_type"].astype(str).unique())
    print(f"\nCell types in this tube: {cts}")
