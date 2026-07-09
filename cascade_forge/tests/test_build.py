"""Tests for CascadeSimulator: contract, ground truth, save, and data-level direction."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from cascade_forge import CascadeSimulator, simulate


def test_simulate_contract_and_ground_truth(simple_cascades):
    sim = CascadeSimulator(
        simple_cascades, n_cell_types=3, n_cells_per_tube=90, n_donors=4,
        output="lognorm", seed=0,
    )
    res = sim.simulate(snapshot_times=[1.0, 3.0])
    assert set(res.adatas) == {1.0, 3.0}
    adata = res.adatas[3.0]
    # obs contract
    for col in ("condition", "donor", "cell_type"):
        assert col in adata.obs
    conds = set(adata.obs["condition"])
    assert "PBS" in conds and {"A", "B", "C"}.issubset(conds)
    assert adata.obs["donor"].nunique() == 4
    assert adata.var_names.is_unique
    assert np.all(np.asarray(adata.X) >= 0)
    # ground truth stashed in uns
    gt = adata.uns["cascade_forge"]
    assert ["A", "B"] in gt["direct_edges"]
    assert gt["snapshot_time"] == 3.0
    assert "program_genes_by_label" in gt and "activation_at_snapshot" in gt
    # convenience edges
    assert ("A", "B") in res.direct_edges
    assert ("A", "C") in res.reachable_edges
    assert ("G", "H") in res.bidirectional_pairs


def test_raw_output_is_integer_counts(simple_cascades):
    res = simulate(simple_cascades, snapshot_times=[2.0], n_cell_types=2,
                   n_cells_per_tube=60, n_donors=3, output="raw", seed=1)
    X = np.asarray(res.adata.X)
    assert np.allclose(X, np.round(X)) and np.all(X >= 0)


def test_control_label_collision_raises():
    with pytest.raises(ValueError):
        CascadeSimulator({"PBS": {"B": 0.5}}, control_label="PBS")


def test_few_donors_warns():
    with pytest.warns(RuntimeWarning):
        CascadeSimulator({"A": {"B": 0.5}}, n_donors=2)


def _mean_cross_asym(adata, a, b, program_genes, control="PBS"):
    """cross_asym(a,b) = (s(a,S_b)-s(PBS,S_b)) - (s(b,S_a)-s(PBS,S_a)), pooled over cells."""
    gnames = list(adata.var_names)
    X = np.asarray(adata.X)
    cond = adata.obs["condition"].to_numpy()
    sa = [gnames.index(g) for g in program_genes[a]]
    sb = [gnames.index(g) for g in program_genes[b]]
    s = lambda c, cols: X[cond == c][:, cols].mean()
    m_ab = s(a, sb) - s(control, sb)
    m_ba = s(b, sa) - s(control, sa)
    return m_ab - m_ba


def test_data_level_direction_recovers_planted_chain():
    # Independent of cascadir: the DATA itself must carry the correct cross_asym sign.
    cascades = {"A": {"B": (0.7, 1.0)}, "B": {"C": (0.6, 1.0)}, "D": {"E": (0.6, 1.0)}}
    sim = CascadeSimulator(cascades, n_cell_types=3, n_cells_per_tube=300, n_donors=4,
                           effect_size=0.4, output="lognorm", seed=0)
    res = sim.simulate(snapshot_times=[4.0])
    adata = res.adata
    pg = res.model.program_genes
    for (a, b) in res.direct_edges:
        assert _mean_cross_asym(adata, a, b, pg) > 0, f"wrong sign for {a}->{b}"


def test_save_writes_files(simple_cascades, tmp_path):
    res = simulate(simple_cascades, snapshot_times=[1.0, 2.0], n_cell_types=2,
                   n_cells_per_tube=40, n_donors=3, output="lognorm", seed=0)
    paths = res.save(str(tmp_path))
    assert len(paths) == 2 and all(os.path.exists(p) for p in paths)
    with open(tmp_path / "ground_truth.json") as f:
        gt = json.load(f)
    assert ["A", "B"] in gt["direct_edges"] and "config" in gt
