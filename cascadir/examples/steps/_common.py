"""Shared helpers for the per-step example scripts.

Each numbered script imports from here so it can focus on the *one* step it
demonstrates. Everything runs on the small synthetic planted-cascade dataset
(``examples/synthetic_data.py``) so the scripts are fast and self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

# make examples/synthetic_data.py importable
_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from synthetic_data import make_synthetic_anndata, make_hub_anndata  # noqa: E402

import cascadir as cd  # noqa: E402

# obs column names + control label used throughout the synthetic data
COLS = dict(condition_col="cytokine", donor_col="donor", celltype_col="cell_type")
CONTROL = "PBS"


def raw_anndata(seed: int = 0):
    """A fresh raw-count AnnData with a planted CytA -> CytB cascade."""
    return make_synthetic_anndata(seed=seed)


def hub_anndata(seed: int = 0):
    """Raw-count AnnData with >= 3 conditions and a planted HUB (CytH) — for the
    degree-correction lesson (step 08). CytA<->CytB specifically coupled, CytC
    independent, CytH coupled-to-everyone in the raw gate."""
    return make_hub_anndata(seed=seed)


def preprocessed(seed: int = 0):
    """Log-normalized, HVG-subset AnnData (the method-ready state)."""
    return cd.preprocess(raw_anndata(seed), assume="raw")


def tube_set(seed: int = 0):
    """In-memory pseudo-tubes from the preprocessed data."""
    return cd.build_pseudotubes(
        preprocessed(seed),
        control_label=CONTROL,
        n_per_cell_type=20,
        min_cells=8,
        n_tubes=4,
        seed=seed,
        **COLS,
    )


def trained_encoder(proc, seed: int = 0):
    """A quick Stage-1 encoder (few epochs — this is a demo)."""
    return cd.train_encoder(
        proc, celltype_col=COLS["celltype_col"], epochs=5, device="cpu", seed=seed
    )


def banner(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)
