"""End-to-end test for cascadir.CascadeDirection on the planted-cascade data."""

from __future__ import annotations

import pytest

import cascadir as cd
from cascadir.exceptions import NotFittedError


def _fast_estimator():
    return cd.CascadeDirection(
        condition_col="cytokine",
        donor_col="donor",
        celltype_col="cell_type",
        control_label="PBS",
        tube_config=cd.TubeConfig(n_tubes=5, n_per_cell_type=20, min_cells=8),
        train_config=cd.TrainConfig(encoder_epochs=5, binary_epochs=40),
        cross_asym_config=cd.CrossAsymConfig(top_n=10, min_cells=8, n_null_perms=20),
        device="cpu",
        seed=42,
    )


def test_not_fitted_raises():
    est = _fast_estimator()
    with pytest.raises(NotFittedError):
        est.direction("CytA", "CytB")
    with pytest.raises(NotFittedError):
        est.direction_table()


def test_end_to_end_recovers_planted_direction(synthetic_adata):
    est = _fast_estimator().fit(synthetic_adata, assume="raw")

    # fitted state is populated
    assert est.tube_set is not None
    assert set(est.signatures.keys()) == {"CytA", "CytB"}
    assert est.validation_report.ok

    table = est.direction_table()
    assert not table.empty
    for col in (
        "condition_a",
        "condition_b",
        "cross_asym_median",
        "classification",
        "direction",
        "upstream",
        "null_p",
    ):
        assert col in table.columns

    call = est.direction("CytA", "CytB")
    # planted CytA -> CytB: upstream must be CytA (positive cross_asym)
    assert call.cross_asym_median > 0
    assert call.upstream == "CytA"
    assert call.direction == "a_to_b"
    assert call.classification in ("STRONG", "WEAK")
    # directional_score (symmetric control) is surfaced
    assert "directional_score_median" in table.columns
    # discovered signatures should be enriched for the planted program genes
    a_sig = set(est.signatures["CytA"].genes)
    planted_up = set(synthetic_adata.uns["planted_cascade"]["upstream_program_genes"])
    assert len(a_sig & planted_up) >= 1


def test_full_runner_path_a_and_benchmark(synthetic_adata):
    """The same fitted estimator runs Path A (discover_axes) and analysis (benchmark)."""
    est = _fast_estimator().fit(synthetic_adata, assume="raw")

    # Path A: coupling discovery (existence)
    axes = est.discover_axes()
    assert axes.n_donors == 3
    assert axes.underpowered is True  # 3 donors -> Wilcoxon underpowered (advisory)
    pairs = set(zip(axes.axes["axis_a"], axes.axes["axis_b"]))
    assert ("CytA", "CytB") in pairs

    # Analysis: score the known direction CytA -> CytB
    bench = est.benchmark([("CytA", "CytB")])
    assert bench.n_found == 1
    assert bench.n_scored == 1
    assert bench.cross_accuracy == 1.0     # planted direction recovered
    assert "cross_asym accuracy" in bench.summary()
