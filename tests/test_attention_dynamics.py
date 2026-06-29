"""
Unit tests for cytokine_mil.analysis.attention_dynamics (CLAUDE.md §33).

Pure-array tests: no model, no IO. They verify the math of recruitment timing,
the relay-recruitment-lag direction sign, primary/secondary classification,
Gini bounds, and Spearman — on hand-constructed trajectories with a known answer.
"""

import numpy as np
import pytest

from cytokine_mil.analysis.attention_dynamics import (
    EXPECTED_DOMINANT,
    attention_primary,
    attention_primary_vs_groundtruth,
    celltype_recruitment,
    classify_primary_secondary,
    concentration_summary,
    gini,
    recruitment_order,
    relay_recruitment_lag,
    spearman,
)

EPOCHS = [10, 20, 30, 40, 50]


# --- gini -----------------------------------------------------------------

def test_gini_bounds_and_extremes():
    assert gini([]) == 0.0
    assert gini([0, 0, 0]) == 0.0
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-9)   # uniform
    g = gini([0, 0, 0, 1])                                       # concentrated
    assert 0.0 < g <= 1.0
    assert g == pytest.approx(0.75, abs=1e-9)


# --- recruitment timing ---------------------------------------------------

def test_celltype_recruitment_relative_threshold():
    # early type crosses 0.5*final at idx 1; late type at idx 3.
    traj = {"early": np.array([0, 1, 1, 1, 1.0]),
            "late":  np.array([0, 0, 0, 1, 1.0]),
            "none":  np.array([0, 0, 0, 0, 0.0])}
    rec = celltype_recruitment(traj, EPOCHS, rise_frac=0.5)
    assert rec["early"]["tau"] == 20 and rec["early"]["tau_idx"] == 1
    assert rec["late"]["tau"] == 40 and rec["late"]["tau_idx"] == 3
    assert rec["none"]["tau"] is None   # final <= 0


def test_recruitment_order_early_before_late():
    traj = {"late": np.array([0, 0, 0, 1, 1.0]),
            "early": np.array([0, 1, 1, 1, 1.0])}
    order = recruitment_order(traj, EPOCHS)["order"]
    assert [o[0] for o in order] == ["early", "late"]


def test_attention_primary_picks_highest_final():
    traj = {"Mono": np.array([0, 0.1, 0.5, 0.8, 0.9]),
            "NK":   np.array([0, 0.05, 0.05, 0.05, 0.1])}
    assert attention_primary(traj) == "Mono"


# --- relay-recruitment-lag direction --------------------------------------

def _const(donors, arr):
    return {d: np.asarray(arr, dtype=float) for d in donors}


def test_relay_lag_calls_A_upstream_when_relay_late_in_A():
    donors = ["D1", "D2", "D3"]
    # B's relay cell type = Mono (highest final in B's donor-mean trajectory).
    trajectory = {
        "B": {"Mono": np.array([0, 1, 1, 1, 1.0]), "NK": np.array([0, 0, 0, 0, 0.1])},
        "A": {"Mono": np.array([0, 0, 0, 1, 1.0])},
    }
    per_donor = {
        "A": {"Mono": _const(donors, [0, 0, 0, 1, 1.0])},   # late  -> tau 40
        "B": {"Mono": _const(donors, [0, 1, 1, 1, 1.0])},   # early -> tau 20
    }
    r = relay_recruitment_lag(trajectory, per_donor, EPOCHS, "A", "B", n_boot=500, seed=0)
    assert r["T_B"] == "Mono"
    assert r["mean_lag"] == pytest.approx(20.0)
    assert r["sign_consistency"] == 1.0
    assert r["ci_low"] > 0 and r["call"] == "A->B"


def test_relay_lag_ambiguous_when_symmetric():
    donors = ["D1", "D2", "D3"]
    trajectory = {"B": {"Mono": np.array([0, 1, 1, 1, 1.0])}, "A": {"Mono": np.array([0, 1, 1, 1, 1.0])}}
    same = {"A": {"Mono": _const(donors, [0, 1, 1, 1, 1.0])},
            "B": {"Mono": _const(donors, [0, 1, 1, 1, 1.0])}}
    r = relay_recruitment_lag(trajectory, same, EPOCHS, "A", "B", n_boot=500, seed=0)
    assert r["mean_lag"] == pytest.approx(0.0)
    assert r["call"] == "ambiguous"


def test_relay_lag_insufficient_donors():
    trajectory = {"B": {"Mono": np.array([0, 1, 1, 1, 1.0])}}
    per_donor = {"A": {"Mono": {"D1": np.array([0, 0, 0, 1, 1.0])}},
                 "B": {"Mono": {"D1": np.array([0, 1, 1, 1, 1.0])}}}
    r = relay_recruitment_lag(trajectory, per_donor, EPOCHS, "A", "B")
    assert r["call"] == "ambiguous" and r["n_donors"] == 1


# --- primary / secondary classification -----------------------------------

def test_classify_primary_and_secondary():
    trajectory = {
        "IL-12": {
            "NK":   np.array([0, 1, 1, 1, 1.0]),   # early high -> primary
            "Mono": np.array([0, 0, 0, 1, 1.0]),   # late + p_correct second rise -> secondary
        }
    }
    p_correct = {"IL-12": np.array([0.1, 0.2, 0.3, 0.5, 0.9])}
    labels = classify_primary_secondary(trajectory, EPOCHS, p_correct)["labels"]["IL-12"]
    assert labels["NK"] == "primary"
    assert labels["Mono"] == "secondary"


def test_classify_secondary_needs_second_rise():
    trajectory = {"X": {"Late": np.array([0, 0, 0, 1, 1.0])}}
    flat = {"X": np.array([0.9, 0.9, 0.9, 0.9, 0.9])}   # no second rise
    labels = classify_primary_secondary(trajectory, EPOCHS, flat)["labels"]["X"]
    assert labels["Late"] == "minor"


# --- concentration --------------------------------------------------------

def test_concentration_summary_slope_sign():
    conc = {"X": {"T": np.array([0.1, 0.2, 0.3, 0.4, 0.5])}}
    s = concentration_summary(conc, EPOCHS)["summary"]["X"]["T"]
    assert s["final"] == pytest.approx(0.5)
    assert s["slope"] > 0


# --- spearman / groundtruth ----------------------------------------------

def test_spearman_monotone():
    rho, n = spearman([1, 2, 3, 4], [10, 20, 30, 40])
    assert rho == pytest.approx(1.0) and n == 4
    rho, _ = spearman([1, 2, 3, 4], [40, 30, 20, 10])
    assert rho == pytest.approx(-1.0)


def test_attention_primary_vs_groundtruth_match():
    # NK is a known IL-12 direct responder and has the highest final attention.
    trajectory = {"IL-12": {"NK": np.array([0, 0.5, 0.9, 1.0, 1.0]),
                            "B_cell": np.array([0, 0.0, 0.0, 0.05, 0.1])}}
    out = attention_primary_vs_groundtruth(trajectory, EPOCHS,
                                           {"IL-12": EXPECTED_DOMINANT["IL-12"]}, top_k=3)
    assert out["per_cytokine"]["IL-12"]["match"] is True
    assert out["frac_match"] == pytest.approx(1.0)
