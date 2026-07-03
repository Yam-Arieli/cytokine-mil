"""
Unit tests for the confusion-direction scorer (training-dynamics cascade direction).

Verifies the direction convention end-to-end: if A-tubes leak softmax mass toward B
(cascade A->B, A carries S_B) but B-tubes do not leak toward A, then
Asym[A,B] > 0  =>  A->B, matching cross_asym's `+ => a upstream`.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from cytokine_mil.analysis.confusion_dynamics import (
    compute_asymmetry_score, compute_confusion_trajectory, compute_temporal_profile,
)
import score_confusion_direction as scd  # noqa: E402


def _softmax_rows(mass, T):
    """(K,T) softmax trajectory: constant per-class mass over T epochs, normalized."""
    m = np.asarray(mass, dtype=np.float32)
    m = m / m.sum()
    return np.repeat(m[:, None], T, axis=1)


def _records_cascade(T=10):
    """3 classes A,B,C. A-tubes leak toward B; B-tubes don't leak toward A."""
    recs = []
    for donor in ["D1", "D2", "D3"]:
        # A: high A, moderate B (cascade leak), low C
        recs.append({"cytokine": "A", "donor": donor,
                     "softmax_trajectory": _softmax_rows([0.6, 0.3, 0.1], T)})
        # B: high B, low A
        recs.append({"cytokine": "B", "donor": donor,
                     "softmax_trajectory": _softmax_rows([0.05, 0.9, 0.05], T)})
        # C: high C
        recs.append({"cytokine": "C", "donor": donor,
                     "softmax_trajectory": _softmax_rows([0.05, 0.05, 0.9], T)})
    return recs


def test_asymmetry_sign_encodes_direction():
    le = scd._LE(["A", "B", "C"])
    C, names = compute_confusion_trajectory(_records_cascade(), le)
    assert names == ["A", "B", "C"]
    asym = compute_asymmetry_score(C, late_epoch_fraction=0.3)
    ia, ib = 0, 1
    assert asym[ia, ib] > 0            # A leaks to B => A->B
    assert np.isclose(asym[ia, ib], -asym[ib, ia])   # antisymmetric


def test_le_shim():
    le = scd._LE(["X", "Y"])
    assert le.cytokines == ["X", "Y"]
    assert le.encode("Y") == 1


def test_safe_int():
    assert scd._safe_int("1") == 1
    assert scd._safe_int("-1") == -1
    assert scd._safe_int("") == 0
    assert scd._safe_int(None) == 0


def test_accuracy_counts():
    obs = [1.0, -1.0, 0.0, 1.0]
    exp = [1, 1, -1, 1]
    n_called, correct_called, correct_all, n_total = scd._accuracy(obs, exp)
    assert n_total == 4
    assert n_called == 3               # the 0.0 is uncalled
    assert correct_all == 2            # positions 0 and 3
    assert correct_called == 2


def test_temporal_profile_late_vs_early():
    T = 20
    late = np.zeros((1, 1, T), dtype=np.float32)
    late[0, 0] = np.linspace(0, 1, T)          # peaks at the end
    early = np.zeros((1, 1, T), dtype=np.float32)
    early[0, 0, 1] = 1.0                        # peaks near the start
    assert compute_temporal_profile(late, 0, 0)["profile_type"] == "late"
    assert compute_temporal_profile(early, 0, 0)["profile_type"] == "early"
