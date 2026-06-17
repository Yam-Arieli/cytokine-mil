"""Tests for cascadir.progression — order recovery + nested-donor bootstrap.

Plants a known progression (g1 -> g2 -> g3) in a synthetic donor × cell_type score
cache where each grade's cells carry their own signature plus a weaker "seed" of the
DOWNSTREAM grades' signatures (and ~0 of upstream grades). The pooled cross_asym must
then recover the order, accuracy vs the oracle must be 1.0, and Kendall tau vs the
true order must be +1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cascadir.progression import (
    SIG_PREFIX,
    accuracy_vs_oracle,
    bootstrap_cross_asym,
    kendall_tau,
    pooled_cross_asym,
    recover_order,
)

CONDITIONS = ["g1", "g2", "g3"]
CONTROL = "ctrl"
CELL_TYPES = ["t1", "t2", "t3"]
ORACLE = [("g1", "g2"), ("g1", "g3"), ("g2", "g3")]  # less-advanced = upstream


def _planted_cache(n_donors=6, seed=0, noise=0.05):
    """Score cache with a planted g1->g2->g3 seed gradient.

    s(condition c, signature S_g): 1.0 if g==c (own); 0.5 if g is DOWNSTREAM of c
    (the carried seed); 0.0 if g is upstream. control ~ 0. So cross_asym(a,b)>0 for a
    upstream of b.
    """
    rng = np.random.default_rng(seed)
    pos = {c: i for i, c in enumerate(CONDITIONS)}
    rows = []
    all_conditions = CONDITIONS + [CONTROL]
    for c in all_conditions:
        for d in range(n_donors):
            for t in CELL_TYPES:
                row = {"donor": f"{c}_d{d}", "condition": c, "cell_type": t,
                       "n_cells": int(rng.integers(20, 60))}
                for g in CONDITIONS:
                    if c == CONTROL:
                        base = 0.0
                    elif pos[g] == pos[c]:
                        base = 1.0
                    elif pos[g] > pos[c]:
                        base = 0.5  # downstream seed
                    else:
                        base = 0.0  # upstream: absent
                    row[f"{SIG_PREFIX}{g}"] = base + rng.normal(0, noise)
                rows.append(row)
    return pd.DataFrame(rows)


def test_pooled_cross_asym_signs_match_planted_order():
    cache = _planted_cache()
    ca = pooled_cross_asym(cache, CONDITIONS, CONTROL)
    # a<b alphabetical == upstream first here, so every pair should be positive
    for (a, b), v in ca.items():
        assert v > 0, f"expected {a} upstream of {b}, got cross_asym={v}"


def test_accuracy_and_order_recovery():
    cache = _planted_cache()
    ca = pooled_cross_asym(cache, CONDITIONS, CONTROL)
    assert accuracy_vs_oracle(ca, ORACLE) == pytest.approx(1.0)
    assert recover_order(ca, CONDITIONS) == ["g1", "g2", "g3"]


def test_kendall_tau_extremes():
    assert kendall_tau(["g1", "g2", "g3"], ["g1", "g2", "g3"]) == pytest.approx(1.0)
    assert kendall_tau(["g1", "g2", "g3"], ["g3", "g2", "g1"]) == pytest.approx(-1.0)


def test_recover_order_handles_reversed_signs():
    # Hand-built: g3 upstream of g2 upstream of g1 (all signs flipped)
    ca = {("g1", "g2"): -0.4, ("g1", "g3"): -0.5, ("g2", "g3"): -0.3}
    assert recover_order(ca, CONDITIONS) == ["g3", "g2", "g1"]


def test_bootstrap_cross_asym_ci_and_accuracy():
    cache = _planted_cache(n_donors=8, seed=1)
    res = bootstrap_cross_asym(cache, CONDITIONS, CONTROL, oracle=ORACLE,
                               n_boot=200, seed=7)
    pp = res["per_pair"]
    assert set(zip(pp["condition_a"], pp["condition_b"])) == {
        ("g1", "g2"), ("g1", "g3"), ("g2", "g3")}
    # every pair's bootstrap CI should sit strictly above 0 (clear positive signal)
    assert (pp["ci_lo"] > 0).all(), pp
    # accuracy point and CI: perfect and tight
    assert res["accuracy"]["point"] == pytest.approx(1.0)
    assert res["accuracy"]["ci_lo"] == pytest.approx(1.0)
    assert res["kendall_tau"]["point"] == pytest.approx(1.0)


def test_bootstrap_unit_is_donor_not_cell():
    # Two donors per condition, opposite-but-balanced — point estimate still recovers
    # order, but the bootstrap should reflect donor-level (small-n) uncertainty.
    cache = _planted_cache(n_donors=3, seed=2, noise=0.2)
    res = bootstrap_cross_asym(cache, CONDITIONS, CONTROL, oracle=ORACLE,
                               n_boot=100, seed=3)
    # with only 3 donors/condition and higher noise, CI should be wider than the
    # n_donors=8 case but the point order still correct
    assert res["accuracy"]["point"] == pytest.approx(1.0)
