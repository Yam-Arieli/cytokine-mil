"""Round-trip test for the full-90 pseudo-tube shards.

The staged DAG persists the tube set once and every chunk task reads it back, so a
lossy round-trip would silently give different chunks different tubes — the failure mode
CLAUDE.md §27.6 documents. This asserts the reconstruction is exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

from cascadir.pseudotubes import build_pseudotubes  # noqa: E402

from cytokine_mil.analysis.full90_tube_io import (  # noqa: E402
    load_tube_set,
    read_meta,
    save_tube_shards,
)


@pytest.fixture
def tube_set():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "examples"))
    from synthetic_data import make_synthetic_anndata

    adata = make_synthetic_anndata(seed=0)
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return build_pseudotubes(
        adata,
        condition_col="cytokine",
        donor_col="donor",
        celltype_col="cell_type",
        control_label="PBS",
        n_per_cell_type=20,
        min_cells=8,
        n_tubes=3,
        seed=0,
    )


def _key(t):
    return (t.condition, t.donor, t.tube_idx)


def test_roundtrip_is_exact(tube_set, tmp_path):
    save_tube_shards(tube_set, tmp_path)
    back = load_tube_set(tmp_path, verify_sha=True)

    assert back.gene_names == tube_set.gene_names
    assert back.control_label == tube_set.control_label
    assert len(back.tubes) == len(tube_set.tubes)

    orig = {_key(t): t for t in tube_set.tubes}
    for t in back.tubes:
        o = orig[_key(t)]
        np.testing.assert_array_equal(t.X, o.X)
        assert t.cell_types == o.cell_types
        assert t.cell_types_included == o.cell_types_included


def test_subset_load_keeps_control(tube_set, tmp_path):
    save_tube_shards(tube_set, tmp_path)
    stim = [c for c in tube_set.stimulus_conditions][:1]
    sub = load_tube_set(tmp_path, conditions=stim)
    assert set(sub.conditions) == set(stim) | {"PBS"}


def test_donor_subset(tube_set, tmp_path):
    save_tube_shards(tube_set, tmp_path)
    d0 = tube_set.donors[0]
    sub = load_tube_set(tmp_path, donors=[d0])
    assert set(sub.donors) == {d0}


def test_meta_records_shard_digest(tube_set, tmp_path):
    meta = save_tube_shards(tube_set, tmp_path)
    assert meta["shards_sha256"] == read_meta(tmp_path)["shards_sha256"]
    assert meta["n_tubes"] == len(tube_set.tubes)


def test_no_match_raises(tube_set, tmp_path):
    save_tube_shards(tube_set, tmp_path)
    with pytest.raises(ValueError, match="no shards matched"):
        load_tube_set(tmp_path, conditions=["NotAStimulus"], include_control=False)
