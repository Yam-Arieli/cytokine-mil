"""Tests for cascadir.analysis — scoring direction calls against labels."""

from __future__ import annotations

import pandas as pd
import pytest

from cascadir import score_directions


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "condition_a": "CytA",
                "condition_b": "CytB",
                "cross_asym_median": 0.4,
                "directional_score_median": 0.1,
                "classification": "STRONG",
                "null_p": 0.0,
            },
            {
                "condition_a": "IFNb",
                "condition_b": "IFNg",
                "cross_asym_median": 0.05,
                "directional_score_median": -0.02,
                "classification": "STRONG",
                "null_p": 0.0,
            },
        ]
    )


def test_score_directions_accuracy():
    res = score_directions(_table(), [("CytA", "CytB"), ("IFNb", "IFNg")])
    assert res.n_found == 2
    assert res.n_scored == 2
    assert res.cross_accuracy == 1.0  # both directions correct
    # directional_score control: CytA correct (+), IFNb wrong (-) -> 0.5 (~chance)
    assert res.dirscore_accuracy == 0.5
    assert res.n_null_pass == 2
    assert res.classification_counts.get("STRONG") == 2


def test_score_directions_reversed_label_is_wrong():
    # claim CytB is upstream of CytA -> expected sign flips -> cross call is wrong
    res = score_directions(_table(), [("CytB", "CytA")])
    assert res.cross_accuracy == 0.0
    assert res.table.iloc[0]["called_upstream"] == "CytA"


def test_score_directions_ambiguous_excluded_from_accuracy():
    t = _table()
    t.loc[1, "classification"] = "AMBIGUOUS"
    res = score_directions(t, [("CytA", "CytB"), ("IFNb", "IFNg")])
    assert res.n_found == 2
    assert res.n_scored == 1  # the AMBIGUOUS one is not scored
    assert res.cross_accuracy == 1.0


def test_score_directions_missing_pair_marked_not_found():
    res = score_directions(_table(), [("X", "Y")])
    assert res.n_found == 0
    assert res.table.iloc[0]["found"] in (False, 0)


def test_score_directions_requires_columns():
    bad = pd.DataFrame({"condition_a": ["A"], "condition_b": ["B"]})
    with pytest.raises(ValueError):
        score_directions(bad, [("A", "B")])
