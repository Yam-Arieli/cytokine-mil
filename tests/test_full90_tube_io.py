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


# ---------------------------------------------------------------------------
# Fixed tube split (main / reserve) and the encoded-tube cache
# ---------------------------------------------------------------------------


def test_tube_indices_subset_is_exact_and_uniform(tube_set, tmp_path):
    """`tube_indices` keeps the same indices in every (donor, condition) group."""
    save_tube_shards(tube_set, tmp_path)
    n_idx = max(t.tube_idx for t in tube_set.tubes) + 1
    main = list(range(0, max(1, n_idx // 2)))
    sub = load_tube_set(tmp_path, tube_indices=main)

    assert {t.tube_idx for t in sub.tubes} == set(main)
    # every group that existed still exists, just with fewer tubes
    groups = {(t.donor, t.condition) for t in tube_set.tubes}
    assert {(t.donor, t.condition) for t in sub.tubes} == groups
    # and the surviving arrays are byte-identical to the originals
    by_key = {(t.donor, t.condition, t.tube_idx): t.X for t in tube_set.tubes}
    for t in sub.tubes:
        assert np.array_equal(t.X, by_key[(t.donor, t.condition, t.tube_idx)])


def test_main_and_reserve_are_disjoint(tube_set, tmp_path):
    save_tube_shards(tube_set, tmp_path)
    n_idx = max(t.tube_idx for t in tube_set.tubes) + 1
    if n_idx < 2:
        pytest.skip("fixture has a single tube per group")
    half = n_idx // 2
    main = load_tube_set(tmp_path, tube_indices=list(range(half)))
    reserve = load_tube_set(tmp_path, tube_indices=list(range(half, n_idx)))
    keys = lambda ts: {(t.donor, t.condition, t.tube_idx) for t in ts.tubes}
    assert keys(main).isdisjoint(keys(reserve))
    assert len(keys(main)) + len(keys(reserve)) == len(tube_set.tubes)


def test_unmatched_tube_indices_raises(tube_set, tmp_path):
    save_tube_shards(tube_set, tmp_path)
    with pytest.raises(ValueError, match="no shards matched"):
        load_tube_set(tmp_path, tube_indices=[999])


def test_embedding_cache_roundtrip(tube_set, tmp_path):
    """The encoded tubes reload in exactly the shape train_all_binary wants."""
    import torch

    from cytokine_mil.analysis.full90_tube_io import (
        load_embedding_cache,
        read_embedding_meta,
        save_embedding_cache,
    )

    embed_dim = 5
    rng = np.random.default_rng(0)
    cache = {
        (t.condition, t.donor, t.tube_idx): torch.from_numpy(
            rng.standard_normal((t.X.shape[0], embed_dim)).astype(np.float32)
        )
        for t in tube_set.tubes
    }
    meta = save_embedding_cache(cache, tmp_path / "emb")
    assert meta["n_tubes"] == len(cache)
    assert meta["embed_dim"] == embed_dim
    assert meta["shards_sha256"] == read_embedding_meta(tmp_path / "emb")["shards_sha256"]

    back = load_embedding_cache(tmp_path / "emb")
    assert set(back) == set(cache)
    for k, v in cache.items():
        assert torch.equal(back[k], v)


def test_embedding_cache_condition_subset_keeps_control(tube_set, tmp_path):
    import torch

    from cytokine_mil.analysis.full90_tube_io import (
        load_embedding_cache,
        save_embedding_cache,
    )

    cache = {
        (t.condition, t.donor, t.tube_idx): torch.zeros((t.X.shape[0], 3))
        for t in tube_set.tubes
    }
    save_embedding_cache(cache, tmp_path / "emb")
    stim = list(tube_set.stimulus_conditions)[:1]
    back = load_embedding_cache(tmp_path / "emb", conditions=stim, control_label="PBS")
    assert {k[0] for k in back} == set(stim) | {"PBS"}
