"""End-to-end: cascadir must recover the planted direction from forged data.

This is the whole point of the package. It is skipped unless ``cascadir`` (and torch)
are installed; it trains small MIL models so it is the slowest test (a couple of
minutes on CPU). Uses reduced epochs for speed.
"""

from __future__ import annotations

import pytest

cd = pytest.importorskip("cascadir")

import cascade_forge as cf


@pytest.mark.slow
def test_cascadir_recovers_planted_direction():
    cascades = {
        "AlphaKine": {"BetaKine": (0.75, 2.0)},
        "BetaKine": {"GammaKine": (0.65, 2.0)},
        "DeltaKine": {"EpsilonKine": (0.7, 1.0)},
    }
    sim = cf.CascadeSimulator(
        cascades, n_cell_types=3, n_cells_per_tube=250, n_donors=5,
        effect_size=0.35, output="raw", seed=0,
    )
    adata = sim.simulate(snapshot_times=[4.0]).adata
    edges = sim.graph.direct

    cd.validate_anndata(
        adata, condition_col="condition", donor_col="donor",
        celltype_col="cell_type", control_label="PBS",
    )
    est = cd.CascadeDirection(
        condition_col="condition", donor_col="donor", celltype_col="cell_type",
        control_label="PBS",
        train_config=cd.TrainConfig(encoder_epochs=20, binary_epochs=60),
    ).fit(adata, assume="raw")

    bench = est.benchmark(edges)
    assert bench.n_found == len(edges)
    # Direction should be recovered well above chance on this clean planted cascade.
    assert bench.cross_accuracy_all >= 0.66
