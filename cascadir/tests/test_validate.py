"""Tests for cascadir.validate — strict suitability checking."""

from __future__ import annotations

import numpy as np
import pytest

from cascadir import validate_anndata
from cascadir.exceptions import DataValidationError


def _kw():
    return dict(
        condition_col="cytokine",
        donor_col="donor",
        celltype_col="cell_type",
        control_label="PBS",
    )


def test_valid_data_passes(synthetic_adata):
    report = validate_anndata(synthetic_adata, **_kw())
    assert report.ok
    assert report.control_present
    assert report.n_donors == 3
    assert report.x_state == "raw_counts"
    assert report.issues == ()


def test_missing_control_raises(synthetic_adata):
    a = synthetic_adata
    a.obs["cytokine"] = a.obs["cytokine"].replace({"PBS": "rest"})
    with pytest.raises(DataValidationError) as exc:
        validate_anndata(a, **_kw())
    assert "control_label" in str(exc.value)


def test_too_few_donors_raises(synthetic_adata):
    a = synthetic_adata
    a.obs["donor"] = "donor1"  # collapse to one donor
    with pytest.raises(DataValidationError) as exc:
        validate_anndata(a, **_kw())
    assert "donor" in str(exc.value)


def test_negative_values_raise(synthetic_adata):
    a = synthetic_adata
    X = a.X.copy()
    X[0, 0] = -5.0
    a.X = X
    with pytest.raises(DataValidationError) as exc:
        validate_anndata(a, **_kw())
    assert "negative" in str(exc.value).lower()


def test_nan_values_raise(synthetic_adata):
    a = synthetic_adata
    X = a.X.copy().astype(np.float32)
    X[1, 1] = np.nan
    a.X = X
    with pytest.raises(DataValidationError) as exc:
        validate_anndata(a, **_kw())
    assert "nan" in str(exc.value).lower()


def test_missing_obs_column_raises(synthetic_adata):
    a = synthetic_adata
    del a.obs["cell_type"]
    with pytest.raises(DataValidationError) as exc:
        validate_anndata(a, **_kw())
    assert "cell_type" in str(exc.value)


def test_nonstrict_returns_report_with_issues(synthetic_adata):
    a = synthetic_adata
    a.obs["donor"] = "donor1"
    report = validate_anndata(a, **_kw(), strict=False)
    assert not report.ok
    assert any("donor" in m for m in report.issues)
    # summary is human-readable and mentions NOT OK
    assert "NOT OK" in report.summary()
