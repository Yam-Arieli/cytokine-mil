"""Recurrent IG (opt-in): trajectory capture, consistency, and default no-op."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from cascadir import CascadeDirection, CrossAsymConfig, PreprocessConfig, TrainConfig, TubeConfig

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from synthetic_data import make_hub_anndata  # noqa: E402


# Small, fast configs (the science is exercised on the cluster, not in the unit test).
_PRE = PreprocessConfig(n_hvgs=4000)
_TUBE = TubeConfig(n_per_cell_type=15, min_cells=8, n_tubes=2, seed=0)
_TRAIN = TrainConfig(
    embed_dim=16,
    hidden_dims=(16, 16),
    attention_hidden_dim=8,
    encoder_epochs=2,
    binary_epochs=20,
)
_CA = CrossAsymConfig(top_n=20, n_null_perms=0)


def _make_estimator(**train_overrides):
    tc = TrainConfig(**{**_TRAIN.__dict__, **train_overrides})
    return CascadeDirection(
        condition_col="cytokine",
        donor_col="donor",
        celltype_col="cell_type",
        control_label="PBS",
        preprocess_config=_PRE,
        tube_config=_TUBE,
        train_config=tc,
        cross_asym_config=_CA,
        device="cpu",
        seed=0,
    )


@pytest.fixture(scope="module")
def fitted_recurrent():
    adata = make_hub_anndata(seed=0)
    est = _make_estimator(checkpoint_ig_every_n_epochs=5)
    est.fit(adata, assume="raw", validate=False)
    return est


def test_trajectory_has_expected_checkpoints(fitted_recurrent):
    est = fitted_recurrent
    assert est.signature_trajectories, "trajectories should be populated when opted in"
    expected_epochs = (5, 10, 15, 20)  # binary_epochs=20, every 5
    for cond, traj in est.signature_trajectories.items():
        assert traj.epochs == expected_epochs, f"{cond}: {traj.epochs}"
        assert traj.total_epochs == 20


def test_full_gene_ranking_stored(fitted_recurrent):
    est = fitted_recurrent
    n_genes = len(est.tube_set.gene_names)
    for traj in est.signature_trajectories.values():
        for ck in traj.checkpoints:
            # checkpoint_ig_top_n is None -> full ranking
            assert len(ck.genes) == n_genes
            assert len(ck.ig_scores) == n_genes


def test_trajectory_table_shape(fitted_recurrent):
    est = fitted_recurrent
    df = est.signature_trajectory_table()
    assert list(df.columns) == ["condition", "epoch", "gene", "ig", "rank_ig"]
    n_genes = len(est.tube_set.gene_names)
    n_cond = len(est.signature_trajectories)
    assert len(df) == n_cond * 4 * n_genes  # 4 checkpoints
    assert set(df["rank_ig"]) == set(range(n_genes))


def test_final_checkpoint_matches_static_signature(fitted_recurrent):
    """Epoch == binary_epochs is captured after the last step => equals the static S_X."""
    est = fitted_recurrent
    tn = est.cross_asym_config.top_n
    for cond, sig in est.signatures.items():
        final = est.signature_trajectories[cond].signature_at(20, top_n=tn)
        assert final.genes == sig.genes  # identical model, identical IG


def test_coupling_trajectory_final_matches_static(fitted_recurrent):
    """The per-epoch panel at the final epoch reproduces the static signature_coupling."""
    est = fitted_recurrent
    traj = est.coupling_trajectory(degree_correct=True)
    assert set(traj) == {5, 10, 15, 20}
    final = traj[20].set_index(["condition_a", "condition_b"])
    static = est.signature_coupling(degree_correct=True).set_index(
        ["condition_a", "condition_b"]
    )
    common = final.index.intersection(static.index)
    assert len(common) >= 1
    for col in ("cross_asym", "coupling"):
        a = final.loc[common, col].to_numpy()
        b = static.loc[common, col].to_numpy()
        assert np.allclose(a, b, atol=1e-8), col


def test_default_path_unchanged():
    """No ig_checkpoint_every => no trajectories, empty accessors, normal direction table."""
    adata = make_hub_anndata(seed=1)
    est = _make_estimator()  # checkpoint_ig_every_n_epochs defaults to None
    est.fit(adata, assume="raw", validate=False)
    assert est.signature_trajectories == {}
    assert est.signature_trajectory_table().empty
    assert est.coupling_trajectory() == {}
    tbl = est.direction_table()
    assert len(tbl) >= 1
