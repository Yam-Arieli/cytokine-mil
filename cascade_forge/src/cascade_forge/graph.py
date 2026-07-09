"""Parse and normalize the user-authored cascade dict.

The public ground truth is a nested dict::

    cascades = {
        source_label: { downstream_label: (strength, pseudo_time_delta), ... },
        ...
    }

Each edge value is normalized to ``(strength, delta)``:

* a bare number ``w``            -> ``(w, 1.0)``
* a 1-tuple/list ``(w,)``        -> ``(w, 1.0)``   (missing delta defaults to 1.0)
* a 2-tuple/list ``(w, delta)``  -> ``(w, delta)``

A label that appears only as a *downstream* target (never a key) still exists as a
label with no outgoing cascade (per the spec). ``labels`` is therefore the union of all
sources and all targets, sorted for determinism.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Real
from typing import Dict, List, Mapping, Sequence, Tuple

Edge = Tuple[str, str]
NormEdges = Dict[str, Dict[str, Tuple[float, float]]]


def _normalize_value(value: object, src: str, dst: str) -> Tuple[float, float]:
    """Coerce one edge value into ``(strength, delta)`` with delta defaulting to 1.0."""
    if isinstance(value, Real) and not isinstance(value, bool):
        strength, delta = float(value), 1.0
    elif isinstance(value, (tuple, list)):
        if len(value) == 1:
            strength, delta = float(value[0]), 1.0
        elif len(value) == 2:
            strength, delta = float(value[0]), float(value[1])
        else:
            raise ValueError(
                f"cascade edge {src!r}->{dst!r}: value must be a number, (strength,), "
                f"or (strength, delta); got {value!r}"
            )
    else:
        raise ValueError(
            f"cascade edge {src!r}->{dst!r}: value must be a number, (strength,), "
            f"or (strength, delta); got {value!r}"
        )
    if strength <= 0:
        raise ValueError(f"cascade edge {src!r}->{dst!r}: strength must be > 0, got {strength}")
    if delta <= 0:
        raise ValueError(f"cascade edge {src!r}->{dst!r}: pseudo-time delta must be > 0, got {delta}")
    return strength, delta


def normalize_cascades(cascades: Mapping[str, Mapping[str, object]]) -> Tuple[List[str], NormEdges]:
    """Return ``(labels, edges)``.

    Args:
        cascades: the user dict ``{src: {dst: value}}``.

    Returns:
        labels: sorted union of all source and downstream labels.
        edges: ``{src: {dst: (strength, delta)}}`` with every value normalized.
    """
    if not isinstance(cascades, Mapping):
        raise TypeError(f"cascades must be a dict/Mapping, got {type(cascades).__name__}")

    edges: NormEdges = {}
    labels: set[str] = set()
    for src, downstream in cascades.items():
        src = str(src)
        labels.add(src)
        if not isinstance(downstream, Mapping):
            raise TypeError(
                f"cascades[{src!r}] must be a dict of {{downstream: value}}, "
                f"got {type(downstream).__name__}"
            )
        edges.setdefault(src, {})
        for dst, value in downstream.items():
            dst = str(dst)
            if dst == src:
                raise ValueError(f"self-edge {src!r}->{src!r} is not allowed")
            labels.add(dst)
            edges[src][dst] = _normalize_value(value, src, dst)
    return sorted(labels), edges


def direct_edges(edges: NormEdges) -> List[Edge]:
    """Every authored ``(source, downstream)`` pair, sorted."""
    out = [(src, dst) for src, downstream in edges.items() for dst in downstream]
    return sorted(out)


def reachable_edges(labels: Sequence[str], edges: NormEdges) -> List[Edge]:
    """Transitive closure: every ``(a, b)`` where b is reachable from a (b != a).

    These are all pairs for which ``a`` lies upstream of ``b`` along some directed path
    — the richer (but weaker, for deep hops) directional oracle. Cycles are handled:
    if a and b are mutually reachable, both ``(a, b)`` and ``(b, a)`` are returned (see
    :func:`bidirectional_pairs`).
    """
    adj: Dict[str, List[str]] = {lab: list(edges.get(lab, {}).keys()) for lab in labels}
    out: List[Edge] = []
    for a in labels:
        seen: set[str] = set()
        queue = deque(adj[a])
        while queue:
            node = queue.popleft()
            if node in seen or node == a:
                continue
            seen.add(node)
            queue.extend(adj.get(node, ()))
        for b in sorted(seen):
            out.append((a, b))
    return sorted(out)


def bidirectional_pairs(reach: Sequence[Edge]) -> List[Tuple[str, str]]:
    """Unordered pairs ``{a, b}`` that are mutually reachable (feedback loops).

    Direction for these is genuinely ambiguous (e.g. IL-12 <-> IFN-gamma); a signed
    direction benchmark should exclude them. Returned canonicalized (a < b), sorted.
    """
    reach_set = set(reach)
    pairs = {
        tuple(sorted((a, b)))
        for (a, b) in reach
        if (b, a) in reach_set
    }
    return sorted(pairs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CascadeGraph:
    """Normalized cascade ground truth."""

    labels: List[str]
    edges: NormEdges
    direct: List[Edge]
    reachable: List[Edge]
    bidirectional: List[Tuple[str, str]]

    @classmethod
    def from_dict(cls, cascades: Mapping[str, Mapping[str, object]]) -> "CascadeGraph":
        labels, edges = normalize_cascades(cascades)
        reach = reachable_edges(labels, edges)
        return cls(
            labels=labels,
            edges=edges,
            direct=direct_edges(edges),
            reachable=reach,
            bidirectional=bidirectional_pairs(reach),
        )

    def to_ground_truth(self) -> dict:
        """JSON-serializable ground-truth summary (for ``adata.uns`` / ``ground_truth.json``)."""
        return {
            "labels": list(self.labels),
            "edges": {
                src: {dst: list(wt) for dst, wt in downstream.items()}
                for src, downstream in self.edges.items()
            },
            "direct_edges": [list(e) for e in self.direct],
            "reachable_edges": [list(e) for e in self.reachable],
            "bidirectional_pairs": [list(p) for p in self.bidirectional],
        }
