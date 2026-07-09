"""Pseudo-time activation propagation — the core simulation dynamics.

For each *applied* label ``L`` we compute an **activation vector** ``a_L(t)`` over all
labels: how strongly each label's transcriptional program is switched on at pseudo-time
``t``, given that only ``L`` was applied at ``t = 0``. This is the literal realization of
the user's spec: copy the cascade dict per label, seed only that label, then take a small
step and recompute the strength of the effect, then another step, and so on.

Model
-----
* The applied label ``L`` is a persistent stimulus, **clamped at 1.0** for all t.
* Every edge ``X -> Y`` with ``(strength w, delta tau)`` drives ``Y`` via first-order
  kinetics toward the target ``w * a_X(t)``::

      d y_{X->Y} / dt = (1 / tau) * (w * a_X(t) - y_{X->Y}(t))

  so ``tau`` is the time constant for "one full iteration" (``y`` reaches ~63% of its
  target at ``t = tau``, ~95% at ``3 tau``). A non-applied label's activation is the sum
  of its incoming edge contributions, ``a_Y(t) = sum_{X->Y} y_{X->Y}(t)``.
* Integrated with small explicit-Euler steps ``dt``.

Consequences (used as tests)
----------------------------
* **Steady state = path product.** For a chain ``X -> Y -> Z`` with strengths
  ``w1, w2``, ``a_Y -> w1`` and ``a_Z -> w1 * w2`` as ``t -> inf``.
* **Multi-hop lag / ordering.** ``a_Y`` rises before ``a_Z``: deeper labels come on
  later — the pseudo-time ordering of the cascade.
* **Cycles are bounded** when the loop gain (product of strengths around the loop) < 1;
  an ``activation_cap`` guards against divergence and warns if a loop gain >= 1.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Sequence

from .graph import NormEdges

ActivationVector = Dict[str, float]


def propagate(
    applied_label: str,
    labels: Sequence[str],
    edges: NormEdges,
    snapshot_times: Sequence[float],
    *,
    dt_step: float = 0.05,
    activation_cap: float = 10.0,
) -> Dict[float, ActivationVector]:
    """Activation of every label at each snapshot time, for one applied label.

    Args:
        applied_label: the label held on (clamped at 1.0) from ``t = 0``.
        labels: all labels in the system.
        edges: normalized ``{src: {dst: (strength, delta)}}`` (see :mod:`.graph`).
        snapshot_times: pseudo-times at which to read the activation (>= 0).
        dt_step: explicit-Euler integration step.
        activation_cap: hard cap on any activation (guards runaway feedback loops).

    Returns:
        ``{t: {label: activation}}`` for each ``t`` in ``snapshot_times``.
    """
    if applied_label not in labels:
        raise ValueError(f"applied_label {applied_label!r} not in labels")
    if dt_step <= 0:
        raise ValueError(f"dt_step must be > 0, got {dt_step}")
    times = [float(t) for t in snapshot_times]
    if any(t < 0 for t in times):
        raise ValueError("snapshot_times must be >= 0")

    # Flatten edges into parallel lists for a tight integration loop.
    edge_list = [
        (src, dst, w, tau)
        for src, downstream in edges.items()
        for dst, (w, tau) in downstream.items()
    ]

    # State: current activation per label, and per-edge contribution y_{X->Y}.
    a: ActivationVector = {lab: 0.0 for lab in labels}
    a[applied_label] = 1.0
    y: List[float] = [0.0 for _ in edge_list]

    max_t = max(times) if times else 0.0
    n_steps = int(math.ceil(max_t / dt_step + 1e-9))

    # Record the initial state, then step; sample each snapshot at the nearest step.
    # history[k] is the activation vector after k*dt_step of pseudo-time.
    history: List[ActivationVector] = [dict(a)]
    capped = False
    for _ in range(n_steps):
        # Simultaneous Euler update: compute all targets from the current `a`, then
        # advance every edge, then recompute the non-applied activations.
        for i, (src, _dst, w, tau) in enumerate(edge_list):
            target = w * a[src]
            y[i] += (dt_step / tau) * (target - y[i])
        # Recompute activations (applied label stays clamped).
        new_a: ActivationVector = {lab: 0.0 for lab in labels}
        new_a[applied_label] = 1.0
        for i, (_src, dst, _w, _tau) in enumerate(edge_list):
            if dst == applied_label:
                continue  # applied label is clamped; incoming edges do not move it
            new_a[dst] += y[i]
        for lab in labels:
            if new_a[lab] > activation_cap:
                new_a[lab] = activation_cap
                capped = True
        a = new_a
        history.append(dict(a))

    if capped:
        warnings.warn(
            f"activation hit the cap ({activation_cap}) while propagating from "
            f"{applied_label!r}; a feedback loop gain may be >= 1. Results are clamped.",
            RuntimeWarning,
            stacklevel=2,
        )

    out: Dict[float, ActivationVector] = {}
    for t in times:
        k = int(round(t / dt_step))
        k = max(0, min(k, len(history) - 1))
        out[t] = history[k]
    return out


def propagate_all(
    labels: Sequence[str],
    edges: NormEdges,
    snapshot_times: Sequence[float],
    *,
    dt_step: float = 0.05,
    activation_cap: float = 10.0,
) -> Dict[str, Dict[float, ActivationVector]]:
    """Run :func:`propagate` for every label as the applied stimulus.

    Returns ``{applied_label: {t: {label: activation}}}``.
    """
    return {
        applied: propagate(
            applied, labels, edges, snapshot_times,
            dt_step=dt_step, activation_cap=activation_cap,
        )
        for applied in labels
    }
