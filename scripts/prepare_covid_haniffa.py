#!/usr/bin/env python
"""Prepare the COVID-Haniffa atlas for the §30 cascade-direction run.

Loads `haniffa21.processed.h5ad`, AUTO-DETECTS the severity / patient / cell-type
obs fields (the atlas field names are documented but we validate at runtime),
subsets to `Healthy` + the 5 ordered COVID severity grades (dropping the LPS and
Non_covid arms), standardizes obs to `severity` / `patient_id` / `cell_type`, and
writes a slim prepared h5ad + an `obs_summary.json` (grade × donor × cell-type
counts) used by the pre-registration and the analysis.

`--inspect_only` reads obs in backed mode (no full load) and just reports the
detected fields + value_counts — run this first on the cluster to confirm the
field mapping cheaply before the heavy subset/write.

Usage:
  python scripts/prepare_covid_haniffa.py --inspect_only
  python scripts/prepare_covid_haniffa.py \
      --raw   /cs/labs/mornitzan/yam.arieli/datasets/COVID_Haniffa/raw/haniffa21.processed.h5ad \
      --out   /cs/labs/mornitzan/yam.arieli/datasets/COVID_Haniffa/prepared/covid_haniffa_prepared.h5ad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np

RAW_DEFAULT = "/cs/labs/mornitzan/yam.arieli/datasets/COVID_Haniffa/raw/haniffa21.processed.h5ad"
OUT_DEFAULT = "/cs/labs/mornitzan/yam.arieli/datasets/COVID_Haniffa/prepared/covid_haniffa_prepared.h5ad"

# Canonical ordered severity grades (upstream -> downstream) + the control.
GRADES = ["Asymptomatic", "Mild", "Moderate", "Severe", "Critical"]
CONTROL = "Healthy"
GRADE_VOCAB = {g.lower(): g for g in GRADES + [CONTROL]}

SEVERITY_CANDIDATES = [
    "Status_on_day_collection_summary", "Status_on_day_collection",
    "Status", "severity", "disease_severity", "Severity", "condition",
]
DONOR_CANDIDATES = [
    "patient_id", "sample_id", "donor_id", "donor", "PatientID", "patient", "sample",
]
CELLTYPE_CANDIDATES = [
    "initial_clustering", "full_clustering", "cell_type", "celltype",
    "majority_voting", "Annotation", "cell_type_annotation",
]


def _detect_severity(obs) -> str:
    best, best_hits = None, 0
    for c in SEVERITY_CANDIDATES + list(obs.columns):
        if c not in obs.columns:
            continue
        vals = {str(v).lower() for v in obs[c].unique()}
        hits = len(vals & set(GRADE_VOCAB))
        if hits > best_hits:
            best, best_hits = c, hits
    if best is None or best_hits < 3:
        raise SystemExit(
            f"Could not detect a severity column (need >=3 of {sorted(GRADE_VOCAB)}). "
            f"Columns available: {list(obs.columns)}")
    return best


def _detect_first(obs, candidates, what) -> str:
    for c in candidates:
        if c in obs.columns:
            return c
    raise SystemExit(f"Could not detect a {what} column among {candidates}. "
                     f"Columns available: {list(obs.columns)}")


def _canonical_grade(v: str):
    """Map a raw severity value to a canonical grade / CONTROL, or None to drop."""
    return GRADE_VOCAB.get(str(v).strip().lower(), None)


def _xmin(X):
    return float(X.min()) if X is not None else None


def _expr_report(adata):
    print(f"\n[expr] X min={_xmin(adata.X):.3g} max={float(adata.X.max()):.3g}")
    has_raw = adata.raw is not None
    print(f"[expr] raw present: {has_raw}"
          + (f"  (raw X min={_xmin(adata.raw.X):.3g}, n_vars={adata.raw.n_vars})" if has_raw else ""))
    print(f"[expr] layers: {list(adata.layers.keys())}")


def _resolve_nonneg(sub):
    """Return an AnnData whose X is non-negative (raw counts or log-norm), carrying
    sub.obs. Prefers X if already non-negative, else adata.raw, else a layer."""
    import anndata as _ad
    if _xmin(sub.X) is not None and _xmin(sub.X) >= 0:
        return sub
    if sub.raw is not None and _xmin(sub.raw.X) is not None and _xmin(sub.raw.X) >= 0:
        print("[expr] using adata.raw (X was scaled/negative)")
        return _ad.AnnData(X=sub.raw.X.copy(), obs=sub.obs.copy(), var=sub.raw.var.copy())
    for key in ("raw", "counts", "raw_counts", "lognorm", "log1p", "data", "normalized"):
        if key in sub.layers and _xmin(sub.layers[key]) >= 0:
            print(f"[expr] using layer {key!r} (X was scaled/negative)")
            return _ad.AnnData(X=sub.layers[key].copy(), obs=sub.obs.copy(), var=sub.var.copy())
    raise SystemExit(
        f"No non-negative expression source. X min={_xmin(sub.X)}; "
        f"raw={'present' if sub.raw is not None else 'absent'}; layers={list(sub.layers)}. "
        "cascadir needs raw counts or log-normalized values.")


def _report(obs, sev_col, donor_col, ct_col):
    print(f"[detect] severity = {sev_col!r}")
    print(f"[detect] donor    = {donor_col!r}")
    print(f"[detect] celltype = {ct_col!r}")
    print(f"\n[severity value_counts in {sev_col!r}]")
    print(obs[sev_col].value_counts().to_string())
    mapped = obs[sev_col].map(_canonical_grade)
    keep = mapped.notna()
    print(f"\n[mapped → canonical] kept {int(keep.sum())}/{len(obs)} cells "
          f"({obs[sev_col].nunique()} raw values → {mapped[keep].nunique()} canonical)")
    print(mapped[keep].value_counts().to_string())
    print(f"\n[donors per canonical grade] ({donor_col})")
    tmp = obs.loc[keep, [donor_col]].copy()
    tmp["g"] = mapped[keep].values
    print(tmp.groupby("g")[donor_col].nunique().to_string())
    print(f"\n[cell types in {ct_col!r}] n={obs[ct_col].nunique()}")
    print(obs[ct_col].value_counts().head(40).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--celltype_field", default=None,
                    help="force a cell-type column (default: auto-detect, prefers initial_clustering)")
    ap.add_argument("--inspect_only", action="store_true")
    args = ap.parse_args()

    backed = "r" if args.inspect_only else None
    print(f"[load] {args.raw}  (backed={backed})")
    adata = ad.read_h5ad(args.raw, backed=backed)
    obs = adata.obs
    print(f"[load] {adata.n_obs} cells × {adata.n_vars} genes; obs columns: {list(obs.columns)}")

    sev_col = _detect_severity(obs)
    donor_col = _detect_first(obs, DONOR_CANDIDATES, "donor/patient")
    ct_col = args.celltype_field or _detect_first(obs, CELLTYPE_CANDIDATES, "cell-type")
    _report(obs, sev_col, donor_col, ct_col)
    _expr_report(adata)

    if args.inspect_only:
        print("\n[inspect_only] done — no file written.")
        return 0

    # Subset to mappable cells (Healthy + 5 grades), standardize obs.
    canon = adata.obs[sev_col].map(_canonical_grade)
    keep = canon.notna().to_numpy()
    sub = adata[keep].to_memory() if adata.isbacked else adata[keep].copy()
    sub.obs["severity"] = canon[keep].astype(str).values
    sub.obs["patient_id"] = adata.obs[donor_col][keep].astype(str).values
    sub.obs["cell_type"] = adata.obs[ct_col][keep].astype(str).values

    # The Haniffa "processed" X is scaled/z-scored (negative) — cascadir needs raw or
    # log-norm. Resolve a non-negative source (raw / layer) before writing.
    sub = _resolve_nonneg(sub)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(args.out)
    print(f"[write] {args.out}  ({sub.n_obs} cells × {sub.n_vars} genes; "
          f"X min={_xmin(sub.X):.3g} max={float(sub.X.max()):.3g})")

    summary = {
        "raw": args.raw, "out": args.out,
        "detected": {"severity": sev_col, "donor": donor_col, "cell_type": ct_col},
        "n_cells": int(sub.n_obs), "n_genes": int(sub.n_vars),
        "control_label": CONTROL, "grades_ordered": GRADES,
        "cells_per_grade": sub.obs["severity"].value_counts().to_dict(),
        "donors_per_grade": sub.obs.groupby("severity")["patient_id"].nunique().to_dict(),
        "cell_types": sorted(sub.obs["cell_type"].unique().tolist()),
        "n_cell_types": int(sub.obs["cell_type"].nunique()),
    }
    summ_path = Path(args.out).parent / "obs_summary.json"
    summ_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[write] {summ_path}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
