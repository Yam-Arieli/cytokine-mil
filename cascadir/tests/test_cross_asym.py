"""Tests for cascadir.cross_asym — the scientific core (deterministic, no training).

We hand-build a (condition, cell_type) -> array map with a known cascade and known
signatures, so the cross_asym math is tested in isolation from training noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascadir import (
    aggregate_direction,
    classify_call,
    direction_call,
    directional_asymmetry_test,
    random_gene_set_null,
)
from cascadir.config import CrossAsymConfig
from cascadir.exceptions import SignatureError
from cascadir.types import Signature

G = 30
GENE_NAMES = tuple(f"gene{i}" for i in range(G))
S_A = Signature("A", tuple(f"gene{i}" for i in range(0, 10)), tuple([1.0] * 10), 10)
S_B = Signature("B", tuple(f"gene{i}" for i in range(10, 20)), tuple([1.0] * 10), 10)


def _planted_cells():
    """A upstream of B: A-cells carry genes 0-19 (own + autocrine B), B-cells 10-19."""
    rng = np.random.default_rng(0)
    n = 50
    cbp: dict[tuple[str, str], np.ndarray] = {}
    for ct in ("Mono", "NK"):
        pbs = np.clip(rng.normal(0.1, 0.02, (n, G)), 0, None).astype(np.float32)
        a = np.clip(rng.normal(0.1, 0.02, (n, G)), 0, None).astype(np.float32)
        a[:, 0:20] += 2.0
        b = np.clip(rng.normal(0.1, 0.02, (n, G)), 0, None).astype(np.float32)
        b[:, 10:20] += 2.0
        cbp[("PBS", ct)] = pbs
        cbp[("A", ct)] = a
        cbp[("B", ct)] = b
    return cbp


def test_known_direction_recovered():
    cbp = _planted_cells()
    sigs = {"A": S_A, "B": S_B}
    call = direction_call(
        cbp, sigs, GENE_NAMES, "A", "B", config=CrossAsymConfig(n_null_perms=0)
    )
    assert call.condition_a == "A" and call.condition_b == "B"
    assert call.cross_asym_median > 0
    assert call.direction == "a_to_b"
    assert call.upstream == "A"
    assert call.classification == "STRONG"


def test_cross_asym_is_antisymmetric():
    cbp = _planted_cells()
    sig_idx = {"A": np.arange(0, 10), "B": np.arange(10, 20)}
    df_ab = directional_asymmetry_test(cbp, sig_idx, "A", "B")
    df_ba = directional_asymmetry_test(cbp, sig_idx, "B", "A")
    assert np.allclose(
        df_ab["cross_asym"].to_numpy(), -df_ba["cross_asym"].to_numpy(), atol=1e-6
    )


def test_aggregate_direction_basic():
    import pandas as pd

    df = pd.DataFrame({"cross_asym": [0.2, 0.3, -0.1]})
    med, cons, npos, nneg = aggregate_direction(df)
    assert med == 0.2
    assert npos == 2 and nneg == 1
    assert cons == pytest.approx(2 / 3)


def test_classify_thresholds():
    cfg = CrossAsymConfig()
    assert classify_call(0.005, 1.0, cfg) == "AMBIGUOUS"   # below magnitude
    assert classify_call(0.05, 0.80, cfg) == "STRONG"
    assert classify_call(0.05, 0.60, cfg) == "WEAK"
    assert classify_call(0.05, 0.40, cfg) == "AMBIGUOUS"   # below weak consensus


def test_identical_signatures_raise():
    cbp = _planted_cells()
    same = {
        "A": Signature("A", S_A.genes, S_A.ig_scores, 10),
        "B": Signature("B", S_A.genes, S_A.ig_scores, 10),  # identical genes
    }
    with pytest.raises(SignatureError):
        direction_call(cbp, same, GENE_NAMES, "A", "B")


def test_null_pool_too_small_raises():
    cbp = _planted_cells()
    excluded = set(range(25))  # leave only 5 genes in the pool
    with pytest.raises(SignatureError):
        random_gene_set_null(
            cbp, "A", "B", size_a=10, size_b=10, n_genes=G,
            excluded_indices=excluded, n_perms=5,
        )


def test_null_returns_array():
    cbp = _planted_cells()
    null = random_gene_set_null(
        cbp, "A", "B", size_a=5, size_b=5, n_genes=G,
        excluded_indices=set(range(0, 20)), n_perms=10, seed=1,
    )
    assert null.shape == (10,)
