"""Tests for the gene model and per-cell planting."""

from __future__ import annotations

import numpy as np

from cascade_forge.expression import ExpressionConfig, build_gene_model, generate_tube
from cascade_forge.graph import normalize_cascades


def _model(cascades, **kw):
    labels, edges = normalize_cascades(cascades)
    rng = np.random.default_rng(0)
    return build_gene_model(labels, edges, n_donors=3, rng=rng, **kw), edges


def test_program_blocks_disjoint_and_markers_dominant():
    model, _ = _model({"A": {"B": 0.7}}, n_cell_types=3)
    # Programs are disjoint gene sets (cascadir requires S_a != S_b).
    ga, gb = set(model.program_genes["A"]), set(model.program_genes["B"])
    assert ga and gb and ga.isdisjoint(gb)
    # Cell-type marker gap dominates the program bump.
    marker_gap = model.cfg.marker_high_mu - model.cfg.housekeeping_mu
    assert model.cfg.effect_size < 0.5 * marker_gap


def test_responder_mode_all():
    model, _ = _model({"A": {"B": 0.7}}, n_cell_types=4, responder_mode="all")
    for lab in model.labels:
        assert np.allclose(model.mask_by_type[lab], 1.0)


def test_responder_mode_receptor_enforces_subset_constraint():
    # resp(src) subset resp(dst) for every edge, so cascade direction stays correct.
    model, edges = _model(
        {"A": {"B": 0.7}, "B": {"C": 0.6}}, n_cell_types=6, responder_mode="receptor",
    )
    for src, downstream in edges.items():
        for dst in downstream:
            assert set(model.responders[src]).issubset(set(model.responders[dst]))


def test_generate_tube_shapes_and_output_modes():
    model, _ = _model({"A": {"B": 0.7}}, n_cell_types=3)
    rng = np.random.default_rng(1)
    Xr, cts = generate_tube(model, 0, {"A": 1.0, "B": 0.5}, n_cells=60, output="raw", rng=rng)
    assert Xr.shape == (60, model.n_genes) and len(cts) == 60
    assert np.all(Xr >= 0) and np.allclose(Xr, np.round(Xr))    # integer counts
    Xl, _ = generate_tube(model, 0, None, n_cells=60, output="lognorm", rng=rng)
    assert Xl.shape == (60, model.n_genes) and np.all(Xl >= 0)


def test_autocrine_asymmetry_carries_direction():
    # The crux: s(A, S_B) > s(B, S_A). In an A-tube (A upstream of B), B's program is
    # partially on; in a B-tube, A's program is absent.
    labels, edges = normalize_cascades({"A": {"B": 0.7}})
    rng = np.random.default_rng(0)
    model = build_gene_model(labels, edges, n_cell_types=3, n_donors=3, rng=rng,
                             cfg=ExpressionConfig(effect_size=0.4))
    gnames = list(model.gene_names)
    sb = [gnames.index(g) for g in model.program_genes["B"]]
    sa = [gnames.index(g) for g in model.program_genes["A"]]

    gen = np.random.default_rng(7)
    # A applied: activation A=1.0 (own), B=0.7 (autocrine downstream).
    Xa, _ = generate_tube(model, 0, {"A": 1.0, "B": 0.7}, n_cells=400, output="lognorm", rng=gen)
    # B applied: activation B=1.0 (own), A absent.
    Xb, _ = generate_tube(model, 0, {"B": 1.0}, n_cells=400, output="lognorm", rng=gen)
    Xp, _ = generate_tube(model, 0, None, n_cells=400, output="lognorm", rng=gen)

    m_ab = Xa[:, sb].mean() - Xp[:, sb].mean()   # s(A, S_B) - s(PBS, S_B)
    m_ba = Xb[:, sa].mean() - Xp[:, sa].mean()   # s(B, S_A) - s(PBS, S_A)
    cross = m_ab - m_ba
    assert cross > 0                              # positive -> A upstream (correct)
