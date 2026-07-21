#!/usr/bin/env python
"""Run cascadir end-to-end on the prepared Sheu 2024 single-timepoint AnnData.

Fits `CascadeDirection` (Stage-1 encoder -> per-stimulus binary AB-MIL -> IG
signatures -> cross_asym / coupling) from the assembled h5ad
(`scripts/prepare_sheu_cascadir.py`), then persists:

  direction_table.csv     all 21 pairs: cross_asym_median, directional_score_median,
                          classification, direction, upstream, null_p, ...
  coupling_cell.csv       cell-level, degree-corrected coupling (Sheu has only ~4
                          pseudo-donors, so donor_level=False per cascadir's own
                          rule -- pipeline.py:409-413, see cascadir-values SKILL.md)

Usage (cluster, GPU):
  python scripts/run_sheu_cascadir.py \\
      --prepared /cs/.../Sheu2024_pseudotubes/prepared/sheu_5hr_prepared.h5ad \\
      --output_dir results/sheu_cascadir_native/5hr --device cuda --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

import cascadir as cd
from cascadir.config import PreprocessConfig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--output_dir", default="results/sheu_cascadir_native/5hr")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--assume", default="lognorm", choices=["auto", "raw", "lognorm"])
    ap.add_argument("--control_label", default="PBS")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.prepared}")
    adata = ad.read_h5ad(args.prepared)
    print(f"[load] {adata.n_obs} cells x {adata.n_vars} genes; "
          f"stimuli present: {sorted(set(adata.obs['cytokine']) - {args.control_label})}")

    # Sheu tubes are already normalize_total+log1p'd with no raw `counts` layer
    # (CLAUDE.md §3), and the 500-gene targeted panel makes HVG selection a no-op
    # either way -- flavor="seurat" computes HVGs directly on log-normalized values
    # instead of requiring raw counts (cascadir's own documented fallback, see
    # PreprocessConfig / preprocess()'s NotPreprocessedError message).
    est = cd.CascadeDirection(
        condition_col="cytokine", donor_col="donor", celltype_col="cell_type",
        control_label=args.control_label, device=args.device, seed=args.seed,
        preprocess_config=PreprocessConfig(flavor="seurat"),
    ).fit(adata, assume=args.assume)
    print("[fit] done. conditions:", est.tube_set.stimulus_conditions)

    dt = est.direction_table()
    dt.to_csv(out / "direction_table.csv", index=False)
    print(f"[write] direction_table.csv  ({len(dt)} pairs)")

    # Sheu has ~4 pseudo-donors -- donor_level coupling is underpowered (cascadir's
    # own rule, pipeline.py:409-413); use the cell-level degree-corrected fallback.
    coupling = est.signature_coupling(donor_level=False, degree_correct=True)
    coupling.to_csv(out / "coupling_cell.csv", index=False)
    print(f"[write] coupling_cell.csv  ({len(coupling)} pairs)")

    print("[done] fit artifacts in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
