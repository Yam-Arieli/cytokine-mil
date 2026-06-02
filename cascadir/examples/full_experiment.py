"""Full experiment on a brand-new dataset: preprocess -> Path A -> Path B -> analysis.

Run::

    python examples/full_experiment.py

Demonstrates everything cascadir does, in the order the method runs:
  1. fit            : validate + preprocess + pseudo-tubes + encoder + binary models + signatures
  2. discover_axes  : Path A — *which* pairs are coupled (direction-agnostic)
  3. direction_table: Path B — *who is upstream* for each pair (cross_asym)
  4. benchmark      : score the directions against known (upstream, downstream) labels
"""

from __future__ import annotations

import logging

from synthetic_data import make_synthetic_anndata

import cascadir as cd


def main() -> None:
    logging.basicConfig(level=logging.WARNING)  # quiet the training logs

    adata = make_synthetic_anndata(seed=0)
    print(f"[data] {adata.n_obs} cells x {adata.n_vars} genes; "
          f"conditions={sorted(adata.obs['cytokine'].unique())}\n")

    est = cd.CascadeDirection(
        condition_col="cytokine",
        donor_col="donor",
        celltype_col="cell_type",
        control_label="PBS",
        tube_config=cd.TubeConfig(n_tubes=5, n_per_cell_type=20, min_cells=8),
        train_config=cd.TrainConfig(encoder_epochs=5, binary_epochs=40),
        cross_asym_config=cd.CrossAsymConfig(top_n=10, min_cells=8, n_null_perms=30),
        device="cpu",
        seed=42,
    ).fit(adata, assume="raw")

    # 1) validation report
    print("[validate]")
    print(est.validation_report.summary(), "\n")

    # 2) Path A — coupling discovery (existence)
    axes = est.discover_axes()
    print("[Path A] coupling axes (existence; direction-agnostic)")
    print(axes.summary())
    print(axes.axes.to_string(index=False), "\n")

    # 3) Path B — direction (who is upstream)
    print("[Path B] direction table (cross_asym)")
    print(est.direction_table().to_string(index=False), "\n")

    # 4) Analysis — score against known direction labels
    bench = est.benchmark([("CytA", "CytB")])  # ground truth: CytA upstream of CytB
    print("[analysis] benchmark vs known labels")
    print(bench.summary())


if __name__ == "__main__":
    main()
