"""
Unit tests for cytokine_mil.analysis.source_potency (training-dynamics source score).
Pure numpy on hand-built trajectories / records / axes rows — no model, no IO.
"""

import numpy as np

from cytokine_mil.analysis.source_potency import (
    ceiling, graph_coupling_degree, graph_out_degree, late_phase_gain,
    normalized_trajectory_auc, per_cytokine_metrics, plateau_epoch,
    source_potency_table, validate_against_degree, validate_deep_vs_shallow,
)

EARLY = [0.1, 0.5, 0.9, 0.95, 0.95, 0.95]   # plateaus early
LATE = [0.1, 0.2, 0.3, 0.5, 0.7, 0.95]      # plateaus late
EP = [1, 2, 3, 4, 5, 6]


# --- curve-shape metrics ---------------------------------------------------

def test_ceiling_is_max():
    assert ceiling(LATE) == 0.95


def test_normalized_auc_early_gt_late():
    assert normalized_trajectory_auc(EARLY) > normalized_trajectory_auc(LATE)


def test_plateau_epoch_early_before_late():
    assert plateau_epoch(EARLY, EP, 0.9) < plateau_epoch(LATE, EP, 0.9)


def test_plateau_epoch_values():
    assert plateau_epoch(EARLY, EP, 0.9) == 3   # 0.9 >= 0.855 at idx2
    assert plateau_epoch(LATE, EP, 0.9) == 6


def test_plateau_epoch_none_on_zero():
    assert plateau_epoch([0.0, 0.0, 0.0], [1, 2, 3], 0.9) is None


def test_late_gain_flat_vs_rising():
    assert late_phase_gain([0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) == 0.0
    assert late_phase_gain(LATE) > 0.0


# --- per-cytokine metrics + table ------------------------------------------

def test_per_cytokine_metrics_excludes_pbs():
    recs = [
        {"cytokine": "A", "donor": "D1", "p_correct_trajectory": EARLY},
        {"cytokine": "A", "donor": "D2", "p_correct_trajectory": EARLY},
        {"cytokine": "B", "donor": "D1", "p_correct_trajectory": LATE},
        {"cytokine": "PBS", "donor": "D1", "p_correct_trajectory": EARLY},
    ]
    m = per_cytokine_metrics(recs, EP, exclude=["PBS"])
    assert set(m) == {"A", "B"}
    assert m["B"]["plateau_epoch"] > m["A"]["plateau_epoch"]


def test_source_potency_table_ceiling_floor():
    metrics = {
        "hi": {"P_max": 0.9, "normalized_auc": 0.4, "late_gain": 0.3, "plateau_epoch": 6.0},
        "low": {"P_max": 0.02, "normalized_auc": 0.9, "late_gain": 0.0, "plateau_epoch": 2.0},
        "mid": {"P_max": 0.5, "normalized_auc": 0.8, "late_gain": 0.05, "plateau_epoch": 3.0},
    }
    t = source_potency_table(metrics, ceiling_floor=0.1)
    assert t["low"]["included"] == 0.0 and np.isnan(t["low"]["source_potency"])
    assert t["hi"]["included"] == 1.0 and np.isfinite(t["hi"]["source_potency"])
    # 'hi' (later plateau + more late gain) should out-score 'mid'
    assert t["hi"]["source_potency"] > t["mid"]["source_potency"]


# --- graph degrees ---------------------------------------------------------

def test_coupling_degree():
    rows = [{"axis_a": "A", "axis_b": "B"}, {"axis_a": "A", "axis_b": "C"}]
    assert graph_coupling_degree(rows) == {"A": 2, "B": 1, "C": 1}


def test_out_degree_uses_sign_and_benchmark_flag():
    rows = [
        {"axis_a": "A", "axis_b": "B", "expected_sign": "1", "counts_in_benchmark": "True"},
        {"axis_a": "A", "axis_b": "C", "expected_sign": "-1", "counts_in_benchmark": "True"},
        {"axis_a": "X", "axis_b": "Y", "expected_sign": "1", "counts_in_benchmark": "False"},
        {"axis_a": "M", "axis_b": "N", "expected_sign": "", "counts_in_benchmark": "True"},
    ]
    # A source once (+1), C source once (b_to_a); X row excluded; empty sign skipped.
    assert graph_out_degree(rows) == {"A": 1, "C": 1}


# --- validation ------------------------------------------------------------

def test_validate_against_degree_monotone():
    potency = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
    degree = {"A": 4, "B": 3, "C": 2, "D": 1}
    res = validate_against_degree(potency, degree)
    assert res["rho"] > 0.9 and res["n"] == 4


def test_validate_deep_vs_shallow_direction():
    potency = {c: 2.0 for c in ["IL-12", "IL-32-beta", "OSM", "IL-22"]}
    potency.update({c: -2.0 for c in ["IL-4", "IL-10", "IL-2", "M-CSF"]})
    res = validate_deep_vs_shallow(
        potency, deep=["IL-12", "IL-32-beta", "OSM", "IL-22"],
        shallow=["IL-4", "IL-10", "IL-2", "M-CSF"], n_perm=2000, seed=0)
    assert res["obs_diff"] > 0 and res["p"] < 0.05
