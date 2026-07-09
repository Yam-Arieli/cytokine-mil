"""Shared fixtures for the cascade_forge test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def simple_cascades():
    """A 2-hop chain, a fan-out, a feedback loop, and a downstream-only label."""
    return {
        "A": {"B": (0.7, 1.0)},          # A -> B
        "B": {"C": (0.6, 1.0)},          # B -> C  (2-hop chain A -> B -> C)
        "D": {"E": 0.6, "F": (0.5,)},    # fan-out; scalar and 1-tuple both mean delta 1.0
        "G": {"H": (0.7, 1.0)},
        "H": {"G": (0.4, 1.0)},          # feedback loop G <-> H
        # C, E, F appear only downstream -> exist with no outgoing cascade
    }
