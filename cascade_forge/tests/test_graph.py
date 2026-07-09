"""Tests for cascade dict parsing / normalization."""

from __future__ import annotations

import pytest

from cascade_forge.graph import (
    CascadeGraph,
    bidirectional_pairs,
    direct_edges,
    normalize_cascades,
    reachable_edges,
)


def test_normalize_default_delta_and_scalar():
    labels, edges = normalize_cascades({"A": {"B": (0.7, 2.0), "C": (0.5,)}, "D": {"E": 0.4}})
    assert edges["A"]["B"] == (0.7, 2.0)
    assert edges["A"]["C"] == (0.5, 1.0)      # 1-tuple -> delta defaults to 1.0
    assert edges["D"]["E"] == (0.4, 1.0)      # scalar -> (0.4, 1.0)


def test_downstream_only_labels_exist():
    labels, edges = normalize_cascades({"A": {"B": 0.5}})
    assert labels == ["A", "B"]               # B exists as a label
    assert edges.get("B", {}) == {}           # ...with no outgoing cascade


def test_invalid_values_raise():
    with pytest.raises(ValueError):
        normalize_cascades({"A": {"A": 0.5}})          # self-edge
    with pytest.raises(ValueError):
        normalize_cascades({"A": {"B": 0.0}})          # non-positive strength
    with pytest.raises(ValueError):
        normalize_cascades({"A": {"B": (0.5, 0.0)}})   # non-positive delta
    with pytest.raises(ValueError):
        normalize_cascades({"A": {"B": (1, 2, 3)}})    # too many entries


def test_direct_and_reachable_edges():
    _, edges = normalize_cascades({"A": {"B": 0.7}, "B": {"C": 0.6}})
    assert direct_edges(edges) == [("A", "B"), ("B", "C")]
    reach = reachable_edges(["A", "B", "C"], edges)
    assert set(reach) == {("A", "B"), ("A", "C"), ("B", "C")}   # transitive A->C included


def test_bidirectional_pairs():
    labels, edges = normalize_cascades({"G": {"H": 0.7}, "H": {"G": 0.4}})
    reach = reachable_edges(labels, edges)
    assert bidirectional_pairs(reach) == [("G", "H")]


def test_cascade_graph_from_dict(simple_cascades):
    g = CascadeGraph.from_dict(simple_cascades)
    assert set(g.labels) == {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert ("A", "B") in g.direct and ("B", "C") in g.direct
    assert ("A", "C") in g.reachable          # transitive
    assert ("G", "H") in g.bidirectional
    gt = g.to_ground_truth()
    assert gt["direct_edges"] and isinstance(gt["direct_edges"][0], list)
