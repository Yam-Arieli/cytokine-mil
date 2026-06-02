"""Tests for cascadir.pseudotubes — in-memory tube construction + dataset."""

from __future__ import annotations

import torch

from cascadir import build_pseudotubes, preprocess
from cascadir.pseudotubes import InMemoryTubeDataset
from cascadir.types import BinaryLabel


def _build(synthetic_adata):
    proc = preprocess(synthetic_adata, assume="raw")
    return build_pseudotubes(
        proc,
        condition_col="cytokine",
        donor_col="donor",
        celltype_col="cell_type",
        control_label="PBS",
        n_per_cell_type=20,
        min_cells=8,
        n_tubes=4,
        seed=0,
    )


def test_build_pseudotubes_shapes(synthetic_adata):
    ts = _build(synthetic_adata)
    assert len(ts.tubes) > 0
    assert "PBS" in ts.conditions
    assert set(ts.stimulus_conditions) == {"CytA", "CytB"}
    assert len(ts.donors) == 3
    # every tube's cell_types align with its rows
    for t in ts.tubes:
        assert len(t.cell_types) == t.n_cells
        assert t.X.shape[1] == len(ts.gene_names)
        assert set(t.cell_types_included).issubset(set(ts.cell_types))


def test_cells_by_pair_grouping(synthetic_adata):
    ts = _build(synthetic_adata)
    cbp = ts.cells_by_pair()
    # keys are (condition, cell_type)
    conds = {c for (c, _t) in cbp}
    assert {"PBS", "CytA", "CytB"}.issubset(conds)
    for arr in cbp.values():
        assert arr.ndim == 2
        assert arr.shape[1] == len(ts.gene_names)


def test_inmemory_dataset_roundtrip(synthetic_adata):
    ts = _build(synthetic_adata)
    sub = ts.filter(conditions=["CytA", "PBS"])
    le = BinaryLabel(positive="CytA", negative="PBS")
    ds = InMemoryTubeDataset(sub, le)
    assert len(ds) == len(sub.tubes)
    X, label, donor, cond = ds[0]
    assert isinstance(X, torch.Tensor)
    assert X.shape[1] == len(ts.gene_names)
    assert label in (0, 1)
    assert cond in ("CytA", "PBS")
    assert isinstance(donor, str)
    # entries expose the trainer-facing schema
    entries = ds.get_entries()
    assert entries and "condition" in entries[0] and "cytokine" in entries[0]
