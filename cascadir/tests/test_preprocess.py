"""Tests for cascadir.preprocess — state detection + the normalized-or-not branches."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cascadir import is_lognormalized, is_raw_counts, preprocess
from cascadir.exceptions import NotPreprocessedError


def test_detect_raw_counts(synthetic_adata):
    assert is_raw_counts(synthetic_adata)
    assert not is_lognormalized(synthetic_adata)


def test_preprocess_raw_to_lognorm(synthetic_adata):
    out = preprocess(synthetic_adata, assume="raw")
    # small panel (<= n_hvgs) keeps all genes
    assert out.n_vars == synthetic_adata.n_vars
    assert is_lognormalized(out)
    assert not is_raw_counts(out)
    assert "counts" in out.layers


def test_preprocess_idempotent_on_lognorm(synthetic_adata):
    once = preprocess(synthetic_adata, assume="raw")
    before = once.X.copy()
    twice = preprocess(once, assume="auto")  # should detect lognorm and skip re-log
    assert np.allclose(before, twice.X, atol=1e-5)


def test_auto_detect_ambiguous_raises():
    # fractional values with a large max -> neither raw nor lognorm
    rng = np.random.default_rng(0)
    X = (rng.random((60, 20)) * 60.0 + 20.0).astype(np.float32)  # max ~80, fractional
    obs = pd.DataFrame(
        {
            "cytokine": (["PBS", "CytA", "CytB"] * 20),
            "donor": (["d1", "d2", "d3"] * 20),
            "cell_type": (["Mono", "NK"] * 30),
        }
    )
    adata = ad.AnnData(X=X, obs=obs)
    with pytest.raises(NotPreprocessedError) as exc:
        preprocess(adata, assume="auto")
    assert "assume='raw'" in str(exc.value)


def _lognorm_adata_no_counts(n_genes=20):
    rng = np.random.default_rng(1)
    X = (rng.random((60, n_genes)) * 3.0).astype(np.float32)  # fractional, max ~3
    obs = pd.DataFrame(
        {
            "cytokine": (["PBS", "CytA", "CytB"] * 20),
            "donor": (["d1", "d2", "d3"] * 20),
            "cell_type": (["Mono", "NK"] * 30),
        }
    )
    return ad.AnnData(X=X, obs=obs)


def test_lognorm_seurat_v3_without_counts_raises():
    adata = _lognorm_adata_no_counts()
    with pytest.raises(NotPreprocessedError) as exc:
        preprocess(adata, assume="lognorm", flavor="seurat_v3")
    assert "seurat_v3" in str(exc.value)


def test_lognorm_seurat_flavor_ok():
    adata = _lognorm_adata_no_counts()
    out = preprocess(adata, assume="lognorm", flavor="seurat")
    assert out.n_vars == adata.n_vars  # small panel -> all genes kept
    # values unchanged (no re-normalization on lognorm input)
    assert np.allclose(out.X, adata.X, atol=1e-6)
