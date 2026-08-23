"""`CascadeDirection.from_artifacts` must reproduce a live `fit()` exactly.

The staged/DAG workflow trains the encoder, the binary models and the signatures in
separate processes, then rebuilds an estimator from the persisted artifacts. If that
rebuild differed from `fit()`'s state in any way the downstream coupling/direction
numbers would silently diverge, so the round-trip is asserted element-wise.
"""

from __future__ import annotations

import pandas as pd
import pytest

import cascadir as cd
from cascadir.exceptions import DataValidationError


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


@pytest.fixture(scope="module")
def fitted(request):
    from synthetic_data import make_synthetic_anndata

    return _fast_estimator().fit(make_synthetic_anndata(seed=0), assume="raw")


def _rehydrate(fitted):
    return cd.CascadeDirection.from_artifacts(
        fitted.tube_set,
        fitted.signatures,
        condition_col="cytokine",
        donor_col="donor",
        celltype_col="cell_type",
        control_label="PBS",
        cross_asym_config=fitted.cross_asym_config,
        device="cpu",
        seed=42,
    )


def test_direction_table_identical(fitted):
    pd.testing.assert_frame_equal(
        fitted.direction_table(), _rehydrate(fitted).direction_table()
    )


def test_signature_coupling_identical(fitted):
    for donor_level in (False, True):
        pd.testing.assert_frame_equal(
            fitted.signature_coupling(donor_level=donor_level),
            _rehydrate(fitted).signature_coupling(donor_level=donor_level),
        )


def test_benchmark_identical(fitted):
    labels = [("CytA", "CytB")]
    assert (
        fitted.benchmark(labels).cross_accuracy
        == _rehydrate(fitted).benchmark(labels).cross_accuracy
    )


def test_cells_by_pair_matches(fitted):
    rehydrated = _rehydrate(fitted)
    assert set(rehydrated._cells_by_pair) == set(fitted._cells_by_pair)
    for key, arr in fitted._cells_by_pair.items():
        assert (rehydrated._cells_by_pair[key] == arr).all()


def test_rejects_empty_signatures(fitted):
    with pytest.raises(DataValidationError, match="empty"):
        cd.CascadeDirection.from_artifacts(
            fitted.tube_set, {}, condition_col="cytokine",
            donor_col="donor", celltype_col="cell_type",
        )


def test_rejects_unknown_condition(fitted):
    from cascadir.types import Signature

    bad = dict(fitted.signatures)
    bad["NotAStimulus"] = Signature(
        condition="NotAStimulus", genes=("g0",), ig_scores=(1.0,), top_n=1
    )
    with pytest.raises(DataValidationError, match="absent from the tube"):
        cd.CascadeDirection.from_artifacts(
            fitted.tube_set, bad, condition_col="cytokine",
            donor_col="donor", celltype_col="cell_type",
        )


def test_rejects_foreign_gene_space(fitted):
    from cascadir.types import Signature

    cond = next(iter(fitted.signatures))
    bad = dict(fitted.signatures)
    bad[cond] = Signature(
        condition=cond, genes=("__not_a_gene__",), ig_scores=(1.0,), top_n=1
    )
    with pytest.raises(DataValidationError, match="different gene space"):
        cd.CascadeDirection.from_artifacts(
            fitted.tube_set, bad, condition_col="cytokine",
            donor_col="donor", celltype_col="cell_type",
        )


def test_rejects_control_mismatch(fitted):
    with pytest.raises(DataValidationError, match="does not match"):
        cd.CascadeDirection.from_artifacts(
            fitted.tube_set, fitted.signatures, condition_col="cytokine",
            donor_col="donor", celltype_col="cell_type", control_label="Untreated",
        )
