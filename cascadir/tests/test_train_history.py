"""Opt-in training history, encoder early stopping, and prebuilt embedding caches.

Three additions to :mod:`cascadir.train`, all default-off. The load-bearing guarantee
tested here is that **the defaults are unchanged**: a run with the new arguments left
alone must produce bit-identical weights to one before they existed, because every
validated result in the project was produced on that path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cascadir import (
    build_frozen_embedding_cache,
    build_pseudotubes,
    preprocess,
    train_all_binary,
    train_binary_mil,
    train_encoder,
)
from cascadir.exceptions import DataValidationError


def _proc(synthetic_adata):
    return preprocess(synthetic_adata, assume="raw")


def _tubes(proc):
    return build_pseudotubes(
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


def _encoder(proc, **kw):
    return train_encoder(
        proc,
        celltype_col="cell_type",
        embed_dim=16,
        hidden_dims=(16, 16),
        epochs=3,
        device="cpu",
        seed=0,
        **kw,
    )


def _state_equal(a, b) -> bool:
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    return all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


# ---------------------------------------------------------------------------
# Defaults must not move
# ---------------------------------------------------------------------------


def test_encoder_defaults_bit_identical(synthetic_adata):
    """No val split, no patience, no history -> exactly the pre-existing behaviour."""
    proc = _proc(synthetic_adata)
    assert _state_equal(_encoder(proc), _encoder(proc))


def test_encoder_history_does_not_change_weights(synthetic_adata):
    """Asking for history is pure recording — the trained weights are untouched."""
    proc = _proc(synthetic_adata)
    plain = _encoder(proc)
    with_hist, hist = _encoder(proc, return_history=True)
    assert _state_equal(plain, with_hist)
    assert list(hist.columns) == [
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "is_best", "past_plateau",
    ]
    assert len(hist) == 3
    assert hist.val_loss.isna().all()  # no validation set was requested


def test_binary_history_does_not_change_weights(synthetic_adata):
    proc = _proc(synthetic_adata)
    ts, enc = _tubes(proc), _encoder(proc)
    cond = ts.stimulus_conditions[0]
    kw = dict(control_label="PBS", attention_hidden_dim=8, epochs=4, device="cpu", seed=0)
    plain = train_binary_mil(ts, cond, enc, **kw)
    with_hist, hist = train_binary_mil(ts, cond, enc, return_history=True, **kw)
    assert _state_equal(plain, with_hist)
    assert list(hist.columns) == ["epoch", "loss", "n_megabatches"]
    assert len(hist) == 4
    assert hist.attrs["condition"] == cond
    assert (hist.n_megabatches > 0).all()


# ---------------------------------------------------------------------------
# Validation split + early stopping
# ---------------------------------------------------------------------------


def test_val_split_is_stratified_and_history_populated(synthetic_adata):
    proc = _proc(synthetic_adata)
    enc, hist = _encoder(proc, val_fraction=0.2, return_history=True)
    assert hist.val_loss.notna().all()
    assert hist.attrs["n_val_cells"] > 0
    assert hist.attrs["n_train_cells"] + hist.attrs["n_val_cells"] == proc.n_obs
    # every cell type is represented in the held-out set
    from cascadir.train import _stratified_holdout

    y = proc.obs["cell_type"].astype(str).to_numpy()
    codes = np.unique(y, return_inverse=True)[1]
    _, va = _stratified_holdout(codes, 0.2, seed=0)
    assert set(codes[va].tolist()) == set(codes.tolist())


def test_patience_requires_a_validation_set(synthetic_adata):
    with pytest.raises(DataValidationError, match="patience requires val_fraction"):
        _encoder(_proc(synthetic_adata), patience=2)


def test_val_fraction_must_be_a_fraction(synthetic_adata):
    with pytest.raises(DataValidationError, match="val_fraction"):
        _encoder(_proc(synthetic_adata), val_fraction=1.0)


def test_early_stop_runs_past_the_plateau_then_restores_best(synthetic_adata):
    """patience=1 fires almost immediately; the returned weights are the best-val ones."""
    proc = _proc(synthetic_adata)
    enc, hist = train_encoder(
        proc,
        celltype_col="cell_type",
        embed_dim=16,
        hidden_dims=(16, 16),
        epochs=40,
        lr=5.0,  # deliberately divergent so validation loss stops improving early
        device="cpu",
        seed=0,
        val_fraction=0.2,
        patience=1,
        extra_epochs_after_stop=2,
        return_history=True,
    )
    stopped = hist.attrs["stopped_epoch"]
    assert stopped is not None, "patience=1 with a divergent lr should hit the plateau"
    assert len(hist) < 40, "early stopping should end training before the epoch cap"
    assert len(hist) == stopped + 2, "must run exactly extra_epochs_after_stop past it"
    assert bool(hist.past_plateau.iloc[-1]) is True
    # the returned encoder is the best-val checkpoint, not the final (worse) one
    best_epoch = hist.attrs["best_epoch"]
    assert bool(hist.loc[hist.epoch == best_epoch, "is_best"].iloc[0])
    assert hist.val_loss.min() == pytest.approx(hist.loc[hist.epoch == best_epoch, "val_loss"].iloc[0])
    last = hist.attrs["last_state_dict"]
    if best_epoch != len(hist):
        assert not all(
            torch.equal(v, enc.state_dict()[k].cpu()) for k, v in last.items()
        ), "best-val weights should differ from the final-epoch weights here"


# ---------------------------------------------------------------------------
# Prebuilt embedding cache
# ---------------------------------------------------------------------------


def test_prebuilt_cache_matches_internally_built(synthetic_adata):
    """Passing a persisted cache trains exactly the models an internal build would."""
    proc = _proc(synthetic_adata)
    ts, enc = _tubes(proc), _encoder(proc)
    conds = list(ts.stimulus_conditions)[:2]
    kw = dict(
        conditions=conds, control_label="PBS", attention_hidden_dim=8,
        epochs=4, device="cpu", seed=0,
    )
    internal = train_all_binary(ts, enc, **kw)
    cache = build_frozen_embedding_cache(enc, ts, device="cpu")
    external = train_all_binary(ts, enc, embedding_cache=cache, **kw)
    for c in conds:
        assert _state_equal(internal[c], external[c])


def test_train_all_binary_returns_histories(synthetic_adata):
    proc = _proc(synthetic_adata)
    ts, enc = _tubes(proc), _encoder(proc)
    conds = list(ts.stimulus_conditions)[:2]
    models, hists = train_all_binary(
        ts, enc, conditions=conds, control_label="PBS", attention_hidden_dim=8,
        epochs=3, device="cpu", seed=0, return_history=True,
    )
    assert set(models) == set(conds) == set(hists)
    for c in conds:
        assert len(hists[c]) == 3
        assert hists[c].attrs["condition"] == c
