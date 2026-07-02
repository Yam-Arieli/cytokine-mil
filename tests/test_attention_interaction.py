"""
Unit tests for §34 cell x cell interaction readouts (attention_interaction.py).
Pure numpy on hand-built interaction dicts — no model, no IO.
"""

import numpy as np

from cytokine_mil.analysis.attention_interaction import (
    asymmetry_matrix, directed_pairs, interaction_matrix, offdiagonal_summary,
    parse_pair, relay_interaction_direction,
)

SEP = "||"
EP = [1, 2, 3]


def test_parse_pair():
    assert parse_pair("NK||CD14_Mono") == ("NK", "CD14_Mono")


def test_offdiagonal_gate_pass_and_fail():
    # high cross-type fraction -> pass; near-zero -> fail
    hi = {"A": np.array([0.5, 0.6, 0.7]), "B": np.array([0.4, 0.5, 0.6])}
    lo = {"A": np.array([0.05, 0.04, 0.03])}
    assert offdiagonal_summary(hi, EP, min_frac=0.2)["gate_pass"] is True
    assert offdiagonal_summary(lo, EP, min_frac=0.2)["gate_pass"] is False


def test_interaction_matrix_placement():
    cyt = {
        f"NK{SEP}NK": np.array([0.1, 0.2, 0.3]),
        f"NK{SEP}Mono": np.array([0.4, 0.5, 0.6]),
        f"Mono{SEP}NK": np.array([0.7, 0.8, 0.9]),
        f"Mono{SEP}Mono": np.array([0.0, 0.0, 0.1]),
    }
    M, cts = interaction_matrix(cyt, epoch_idx=-1)
    assert cts == ["Mono", "NK"]  # sorted
    i, j = cts.index("NK"), cts.index("Mono")
    assert M[i, i] == 0.3          # NK->NK final
    assert M[i, j] == 0.6          # NK->Mono final
    assert M[j, i] == 0.9          # Mono->NK final


def test_asymmetry_is_antisymmetric():
    M = np.array([[0.1, 0.6], [0.9, 0.2]])
    A = asymmetry_matrix(M)
    assert np.allclose(A, -A.T)
    assert A[0, 1] == 0.6 - 0.9


def test_directed_pairs_ranks_by_abs_asym():
    cyt = {
        f"NK{SEP}Mono": np.array([0.9]),
        f"Mono{SEP}NK": np.array([0.1]),
        f"NK{SEP}B": np.array([0.2]),
        f"B{SEP}NK": np.array([0.15]),
        f"NK{SEP}NK": np.array([0.0]), f"Mono{SEP}Mono": np.array([0.0]),
        f"B{SEP}B": np.array([0.0]), f"Mono{SEP}B": np.array([0.0]),
        f"B{SEP}Mono": np.array([0.0]),
    }
    dp = directed_pairs(cyt, epoch_idx=-1, top_k=3)
    top = dp["pairs"][0]
    assert {top[0], top[1]} == {"NK", "Mono"}
    assert abs(top[2]) == abs(0.9 - 0.1)


def test_relay_direction_sign_and_call():
    # Cascade A->B: in A's tubes T_B attends to T_A (0.8) more than
    # in B's tubes T_A attends to T_B (0.2) -> D>0 -> a_to_b.
    interaction = {
        "A": {f"T_B{SEP}T_A": np.array([0.5, 0.8])},
        "B": {f"T_A{SEP}T_B": np.array([0.3, 0.2])},
    }
    r = relay_interaction_direction(interaction, "A", "B", T_A="T_A", T_B="T_B")
    assert r["D"] == 0.8 - 0.2
    assert r["call"] == "a_to_b"
    # reverse
    r2 = relay_interaction_direction(
        {"A": {f"T_B{SEP}T_A": np.array([0.1])}, "B": {f"T_A{SEP}T_B": np.array([0.9])}},
        "A", "B", "T_A", "T_B")
    assert r2["call"] == "b_to_a"


def test_relay_direction_missing_entry_is_none():
    r = relay_interaction_direction({"A": {}, "B": {}}, "A", "B", "T_A", "T_B")
    assert r["call"] == "none"
    assert np.isnan(r["D"])
