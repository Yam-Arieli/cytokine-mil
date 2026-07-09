"""cascade_forge — author a cascade graph, forge a single-cell snapshot with known
ground-truth direction, to benchmark ``cascadir``.

Quick start::

    import cascade_forge as cf

    cascades = {"IL12": {"IFNg": (0.7, 2.0)}, "TNF": {"IL6": 0.6}}
    result = cf.simulate(cascades, snapshot_times=[3.0], n_donors=6)
    adata = result.adata               # AnnData meeting the cascadir data contract
    result.direct_edges                # [("IL12","IFNg"), ("TNF","IL6")] -> cascadir.benchmark
"""

from __future__ import annotations

from .build import CascadeSimulator, SimulationResult, simulate
from .dynamics import propagate, propagate_all
from .expression import ExpressionConfig, GeneModel, build_gene_model, generate_tube
from .graph import (
    CascadeGraph,
    bidirectional_pairs,
    direct_edges,
    normalize_cascades,
    reachable_edges,
)

__version__ = "0.1.0"

__all__ = [
    "CascadeSimulator",
    "SimulationResult",
    "simulate",
    "CascadeGraph",
    "normalize_cascades",
    "direct_edges",
    "reachable_edges",
    "bidirectional_pairs",
    "propagate",
    "propagate_all",
    "ExpressionConfig",
    "GeneModel",
    "build_gene_model",
    "generate_tube",
]
