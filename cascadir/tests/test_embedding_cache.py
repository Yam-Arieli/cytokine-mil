"""Frozen-encoder embedding cache: a results-preserving Stage-2 speedup.

The cache pre-encodes each tube once and trains the attention/classifier head on the
cached embeddings (:meth:`AbMil.forward_from_H`) instead of re-running the frozen encoder
every mega-batch step. These tests lock in the guarantee that this is **bit-identical**
to the full per-step forward — the models, and therefore the discovered signatures, must
not change at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cascadir import (
    CascadeDirection,
    CrossAsymConfig,
    PreprocessConfig,
    TrainConfig,
    TubeConfig,
    build_frozen_embedding_cache,
    build_pseudotubes,
    preprocess,
    train_binary_mil,
    train_encoder,
)
from cascadir.signatures import derive_signature


def _build(synthetic_adata):
    """(preprocessed adata, tube_set, shared frozen encoder) with tiny fast configs."""
    proc = preprocess(synthetic_adata, assume="raw")
    ts = build_pseudotubes(
        proc,
        condition_col="cytokine",
        donor_col="donor",
        celltype_col="cell_type",
        control_label="PBS",
        n_per_cell_type=15,
        min_cells=8,
        n_tubes=2,
        seed=0,
    )
    enc = train_encoder(
        proc,
        celltype_col="cell_type",
        embed_dim=16,
        hidden_dims=(16, 16),
        epochs=2,
        device="cpu",
        seed=0,
    )
    return proc, ts, enc


def _heads_equal(m_a, m_b) -> bool:
    """True iff the two models' attention+classifier params are element-wise identical."""
    for sub in ("attention", "classifier"):
        sd_a = getattr(m_a, sub).state_dict()
        sd_b = getattr(m_b, sub).state_dict()
        assert sd_a.keys() == sd_b.keys()
        for k in sd_a:
            if not torch.equal(sd_a[k], sd_b[k]):
                return False
    return True


def test_cache_matches_encoder_forward(synthetic_adata):
    """build_frozen_embedding_cache stores exactly encoder(X), keyed by tube identity."""
    _, ts, enc = _build(synthetic_adata)
    cache = build_frozen_embedding_cache(enc, ts, device="cpu")
    assert set(cache.keys()) == {(t.condition, t.donor, t.tube_idx) for t in ts.tubes}
    enc = enc.eval()
    with torch.no_grad():
        for t in ts.tubes:
            X = torch.from_numpy(np.ascontiguousarray(t.X, dtype=np.float32))
            expected = enc(X)
            got = cache[(t.condition, t.donor, t.tube_idx)]
            assert torch.equal(got, expected)


def test_binary_training_bit_identical(synthetic_adata):
    """Cache ON vs OFF must produce the identical trained head AND signature."""
    _, ts, enc = _build(synthetic_adata)
    common = dict(
        control_label="PBS", epochs=15, seed=0, device="cpu", encoder_frozen=True
    )
    m_off = train_binary_mil(ts, "CytA", enc, use_embedding_cache=False, **common)
    m_on = train_binary_mil(ts, "CytA", enc, use_embedding_cache=True, **common)

    assert _heads_equal(m_on, m_off), "cached-embedding training changed the head weights"

    sig_off = derive_signature(m_off, ts, "CytA", top_n=20, n_steps=10, device="cpu")
    sig_on = derive_signature(m_on, ts, "CytA", top_n=20, n_steps=10, device="cpu")
    assert sig_on.genes == sig_off.genes
    assert sig_on.ig_scores == pytest.approx(sig_off.ig_scores)


def test_prebuilt_cache_matches_local_cache(synthetic_adata):
    """Passing a shared prebuilt cache == letting train_binary_mil build its own."""
    _, ts, enc = _build(synthetic_adata)
    shared = build_frozen_embedding_cache(enc, ts, device="cpu")
    common = dict(control_label="PBS", epochs=12, seed=0, device="cpu")
    m_local = train_binary_mil(ts, "CytA", enc, **common)  # builds its own cache
    m_shared = train_binary_mil(ts, "CytA", enc, embedding_cache=shared, **common)
    assert _heads_equal(m_shared, m_local)


def test_cache_bypassed_when_encoder_unfrozen(synthetic_adata):
    """With encoder_frozen=False the cache is stale, so it must be bypassed (no crash)."""
    _, ts, enc = _build(synthetic_adata)
    model = train_binary_mil(
        ts,
        "CytA",
        enc,
        control_label="PBS",
        epochs=5,
        seed=0,
        device="cpu",
        encoder_frozen=False,
        use_embedding_cache=True,  # ignored because the encoder is not frozen
    )
    assert model.encoder_frozen is False
    # re-freeze the shared encoder so this test does not corrupt others' expectations
    for p in enc.parameters():
        p.requires_grad = False


def test_pipeline_end_to_end_equivalence(synthetic_adata):
    """CascadeDirection.fit gives identical signatures/direction with the cache on vs off."""
    pre = PreprocessConfig(n_hvgs=4000)
    tube = TubeConfig(n_per_cell_type=15, min_cells=8, n_tubes=2, seed=0)
    ca = CrossAsymConfig(top_n=20, n_ig_steps=10, n_null_perms=0)
    base_train = dict(
        embed_dim=16,
        hidden_dims=(16, 16),
        attention_hidden_dim=8,
        encoder_epochs=2,
        binary_epochs=15,
    )

    def _fit(cache_flag: bool):
        est = CascadeDirection(
            condition_col="cytokine",
            donor_col="donor",
            celltype_col="cell_type",
            control_label="PBS",
            preprocess_config=pre,
            tube_config=tube,
            train_config=TrainConfig(**base_train, cache_frozen_embeddings=cache_flag),
            cross_asym_config=ca,
            device="cpu",
            seed=0,
        )
        return est.fit(synthetic_adata, assume="raw", validate=False)

    est_on = _fit(True)
    est_off = _fit(False)

    assert est_on.signatures.keys() == est_off.signatures.keys()
    for cond in est_on.signatures:
        assert est_on.signatures[cond].genes == est_off.signatures[cond].genes

    tbl_on = est_on.direction_table().sort_values(["condition_a", "condition_b"])
    tbl_off = est_off.direction_table().sort_values(["condition_a", "condition_b"])
    assert np.allclose(
        tbl_on["cross_asym_median"].to_numpy(),
        tbl_off["cross_asym_median"].to_numpy(),
        equal_nan=True,
    )
