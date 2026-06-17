#!/usr/bin/env python
"""Run cascadir end-to-end on the prepared COVID-Haniffa atlas (§30 fit stage).

Fits `CascadeDirection` (encoder → per-grade binary models → IG signatures →
cross_asym) from the prepared h5ad, then PERSISTS every artifact the downstream
(CPU) analysis job needs — so analysis never reloads the 7 GB atlas or the GPU model:

  direction_table.csv          all grade pairs: cross_asym_median,
                               directional_score_median, classification, null_p, ...
  signature_coupling.csv       cell-level coupling (EXPLORATORY; donor_level is N/A
                               under nested donors — see §30)
  signatures.json              {grade: [genes]} discovered signatures
  donor_signature_scores.parquet  donor × cell_type × condition mean score on each
                               grade signature + n_cells  → the nested-donor bootstrap cache
  per_celltype_cross_asym.csv  per (pair, cell_type): sA_PB_norm, sB_PA_norm,
                               cross_asym, directional_score  (consensus figure)
  cell_scores_subsample.parquet  subsampled per-cell scores on every grade signature
                               (condition, cell_type)  → signature-scatter figure

Usage (cluster, GPU):
  python scripts/run_covid_cascadir.py \
      --prepared /cs/.../COVID_Haniffa/prepared/covid_haniffa_prepared.h5ad \
      --output_dir results/covid_progression/fit --device cuda --seed 42
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import cascadir as cd
from cytokine_mil.analysis.pathway_audit import directional_asymmetry_test
from cascadir.progression import SIG_PREFIX

CONTROL = "Healthy"
GRADES = ["Asymptomatic", "Mild", "Moderate", "Severe", "Critical"]


def _sig_idx(est) -> dict:
    """{grade: np.ndarray of column indices into tube_set.gene_names}."""
    gidx = est.tube_set.gene_index()
    out = {}
    for cond, sig in est.signatures.items():
        idx = np.array([gidx[g] for g in sig.genes if g in gidx], dtype=np.int64)
        out[cond] = idx
    return out


def _donor_score_cache(est, sig_idx) -> pd.DataFrame:
    """donor × cell_type × condition mean score on each grade signature + n_cells."""
    grades = sorted(sig_idx.keys())
    rows = []
    for d in est.tube_set.donors:
        cbp = est.tube_set.cells_by_pair(donors=[d])  # {(condition, cell_type): (N,G)}
        for (cond, ct), X in cbp.items():
            row = {"donor": d, "condition": cond, "cell_type": ct, "n_cells": int(len(X))}
            for g in grades:
                idx = sig_idx[g]
                row[f"{SIG_PREFIX}{g}"] = float(X[:, idx].mean()) if idx.size else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _per_celltype_cross_asym(cells_by_pair, sig_idx, conditions) -> pd.DataFrame:
    frames = []
    for a, b in combinations(sorted(conditions), 2):
        if a not in sig_idx or b not in sig_idx:
            continue
        df = directional_asymmetry_test(cells_by_pair, sig_idx, A=a, B=b,
                                        P_A=a, P_B=b, pbs_label=CONTROL, min_cells=10)
        if df.empty:
            continue
        df["cross_asym"] = df["sA_PB_norm"] - df["sB_PA_norm"]
        df["condition_a"], df["condition_b"] = a, b
        frames.append(df[["condition_a", "condition_b", "cell_type",
                          "sA_PB_norm", "sB_PA_norm", "cross_asym", "directional_score"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cell_subsample(cells_by_pair, sig_idx, conditions, per_group=1500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grades = sorted(sig_idx.keys())
    rows = []
    for (cond, ct), X in cells_by_pair.items():
        if cond != CONTROL and cond not in conditions:
            continue
        n = len(X)
        take = X if n <= per_group else X[rng.choice(n, per_group, replace=False)]
        rec = {"condition": cond, "cell_type": ct}
        scores = {f"{SIG_PREFIX}{g}": take[:, sig_idx[g]].mean(axis=1) for g in grades}
        for i in range(len(take)):
            rows.append({**rec, **{k: float(v[i]) for k, v in scores.items()}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--output_dir", default="results/covid_progression/fit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--assume", default="auto", choices=["auto", "raw", "lognorm"])
    ap.add_argument("--celltype_col", default="cell_type")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.prepared}")
    adata = ad.read_h5ad(args.prepared)
    print(f"[load] {adata.n_obs} cells × {adata.n_vars} genes; "
          f"grades present: {sorted(set(adata.obs['severity']) - {CONTROL})}")

    est = cd.CascadeDirection(
        condition_col="severity", donor_col="patient_id",
        celltype_col=args.celltype_col, control_label=CONTROL,
        device=args.device, seed=args.seed,
    ).fit(adata, assume=args.assume)
    print("[fit] done. conditions:", est.tube_set.stimulus_conditions)

    conditions = list(est.tube_set.stimulus_conditions)  # the grades present

    # 1. direction table (all pairs)
    dt = est.direction_table()
    dt.to_csv(out / "direction_table.csv", index=False)
    print(f"[write] direction_table.csv  ({len(dt)} pairs)")

    # 2. signature coupling (exploratory; cell-level — donor_level N/A under nesting)
    try:
        sc = est.signature_coupling(donor_level=False)
        sc.to_csv(out / "signature_coupling.csv", index=False)
        print(f"[write] signature_coupling.csv  ({len(sc)} pairs)")
    except Exception as e:  # noqa: BLE001 — exploratory, never block the run
        print(f"[warn] signature_coupling failed (exploratory, skipping): {e}")

    # 3. signatures
    sigs = {c: list(s.genes) for c, s in est.signatures.items()}
    (out / "signatures.json").write_text(json.dumps(sigs, indent=2))
    print(f"[write] signatures.json  ({len(sigs)} grades)")

    sig_idx = _sig_idx(est)
    cells_by_pair = est.tube_set.cells_by_pair()  # pooled

    # 4. donor score cache (the nested-donor bootstrap input)
    cache = _donor_score_cache(est, sig_idx)
    cache.to_parquet(out / "donor_signature_scores.parquet", index=False)
    print(f"[write] donor_signature_scores.parquet  ({len(cache)} donor×celltype rows)")

    # 5. per-cell-type cross_asym (consensus figure)
    pct = _per_celltype_cross_asym(cells_by_pair, sig_idx, conditions)
    pct.to_csv(out / "per_celltype_cross_asym.csv", index=False)
    print(f"[write] per_celltype_cross_asym.csv  ({len(pct)} rows)")

    # 6. subsampled per-cell scores (scatter figure)
    sub = _cell_subsample(cells_by_pair, sig_idx, conditions, seed=args.seed)
    sub.to_parquet(out / "cell_scores_subsample.parquet", index=False)
    print(f"[write] cell_scores_subsample.parquet  ({len(sub)} cells)")

    print("[done] fit artifacts in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
