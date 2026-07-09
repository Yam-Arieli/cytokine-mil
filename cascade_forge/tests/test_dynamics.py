"""Tests for the pseudo-time activation propagation."""

from __future__ import annotations

import numpy as np
import pytest

from cascade_forge.dynamics import propagate
from cascade_forge.graph import normalize_cascades


def test_applied_label_clamped_and_others_zero_at_t0():
    labels, edges = normalize_cascades({"A": {"B": 0.7}})
    out = propagate("A", labels, edges, [0.0], dt_step=0.05)
    assert out[0.0]["A"] == 1.0
    assert out[0.0]["B"] == 0.0


def test_steady_state_is_path_product():
    # Chain A -> B -> C with strengths 0.7, 0.6. At large t: a_B -> 0.7, a_C -> 0.42.
    labels, edges = normalize_cascades({"A": {"B": (0.7, 1.0)}, "B": {"C": (0.6, 1.0)}})
    out = propagate("A", labels, edges, [200.0], dt_step=0.02)
    a = out[200.0]
    assert a["A"] == 1.0
    assert a["B"] == pytest.approx(0.7, abs=1e-2)
    assert a["C"] == pytest.approx(0.7 * 0.6, abs=1e-2)


def test_multihop_lag_ordering():
    # Deeper labels come on later: B reaches half its steady state before C does.
    labels, edges = normalize_cascades({"A": {"B": (0.8, 1.0)}, "B": {"C": (0.8, 1.0)}})
    ts = list(np.linspace(0.1, 20, 200))
    out = propagate("A", labels, edges, ts, dt_step=0.02)
    b_ss, c_ss = 0.8, 0.8 * 0.8

    def first_time_reaching(label, frac, ss):
        for t in ts:
            if out[t][label] >= frac * ss:
                return t
        return float("inf")

    t_half_b = first_time_reaching("B", 0.5, b_ss)
    t_half_c = first_time_reaching("C", 0.5, c_ss)
    assert t_half_b < t_half_c              # B leads C -> pseudo-time ordering


def test_downstream_only_label_has_no_effect_when_applied():
    labels, edges = normalize_cascades({"A": {"B": 0.7}})
    out = propagate("B", labels, edges, [50.0], dt_step=0.05)
    # B is downstream-only: applying it drives nothing, A stays 0.
    assert out[50.0]["B"] == 1.0
    assert out[50.0]["A"] == 0.0


def test_cycle_bounded_when_loop_gain_below_one():
    # a -> b, and a b<->c loop with gain 0.9*0.9 < 1: bounded, no warning.
    labels, edges = normalize_cascades({"a": {"b": (0.7, 1.0)}, "b": {"c": (0.9, 1.0)}, "c": {"b": (0.9, 1.0)}})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # any RuntimeWarning becomes an error
        out = propagate("a", labels, edges, [300.0], dt_step=0.01, activation_cap=100.0)
    b_ss = 0.7 / (1 - 0.81)                   # closed form for the driven loop
    assert out[300.0]["b"] == pytest.approx(b_ss, rel=0.05)
    assert np.isfinite(out[300.0]["c"])


def test_cycle_caps_and_warns_when_loop_gain_at_least_one():
    labels, edges = normalize_cascades({"a": {"b": (0.7, 1.0)}, "b": {"c": (1.0, 1.0)}, "c": {"b": (1.0, 1.0)}})
    with pytest.warns(RuntimeWarning):
        out = propagate("a", labels, edges, [200.0], dt_step=0.02, activation_cap=5.0)
    assert out[200.0]["b"] <= 5.0            # clamped, not infinite
