"""End-to-end cascadir quickstart on synthetic data (CPU, < 1 minute).

Run::

    python examples/quickstart.py

Builds a tiny dataset with a planted ``CytA -> CytB`` cascade, fits the full
pipeline on CPU, and prints the direction table. The planted upstream (``CytA``)
should be called upstream.
"""

from __future__ import annotations

import logging

from synthetic_data import make_synthetic_anndata

import cascadir as cd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    adata = make_synthetic_anndata(seed=0)
    print(f"data: {adata.n_obs} cells x {adata.n_vars} genes; "
          f"conditions={sorted(adata.obs['cytokine'].unique())}")

    # Small configs so the demo runs fast; defaults reproduce the study.
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

    print("\nDiscovered signatures (top 5 genes each):")
    for cond, sig in est.signatures.items():
        print(f"  {cond}: {list(sig.genes[:5])}")

    print("\nDirection table:")
    print(est.direction_table().to_string(index=False))

    call = est.direction("CytA", "CytB")
    print(f"\nCytA vs CytB -> {call.classification}, direction={call.direction}, "
          f"upstream={call.upstream}, cross_asym_median={call.cross_asym_median:+.4f}, "
          f"null_p={call.null_p}")


if __name__ == "__main__":
    main()
