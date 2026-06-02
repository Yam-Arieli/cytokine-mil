"""Shared pytest fixtures for the cascadir test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the planted-cascade synthetic-data helper importable from examples/.
_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from synthetic_data import make_synthetic_anndata  # noqa: E402


@pytest.fixture
def synthetic_adata():
    """A valid raw-count AnnData with a planted CytA -> CytB cascade."""
    return make_synthetic_anndata(seed=0)
