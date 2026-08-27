"""Discovered per-condition signatures via Integrated Gradients (the "bridge").

For each trained binary model (stimulus-vs-control), we attribute the
"this is the stimulus" logit back to the input genes using Integrated Gradients
with the control (PBS) mean as the baseline. The top-``top_n`` genes by mean
attribution are that condition's **discovered signature** ``S_X`` — a data-driven,
condition-specific gene set (no curated pathway lists). These signatures are what
the cross_asym statistic cross-engages.
"""

from __future__ import annotations

import logging
from collections import Counter
from math import comb

import numpy as np
import pandas as pd
import torch

from cascadir.exceptions import SignatureError
from cascadir.models import AbMil
from cascadir.train import resolve_device
from cascadir.types import PseudoTubeSet, Signature

logger = logging.getLogger("cascadir")


def integrated_gradients(
    model: torch.nn.Module,
    X: torch.Tensor,
    *,
    target_class: int = 0,
    baseline: torch.Tensor,
    n_steps: int = 20,
) -> torch.Tensor:
    """Integrated Gradients of ``logit[target_class]`` w.r.t. ``X``.

    Uses the midpoint rule over ``n_steps`` interpolation points between
    ``baseline`` and ``X`` (faithful to the validated probe). The model must return
    ``(logits, ...)`` as its first output.

    Args:
        model: A model whose ``forward(x)`` returns ``(logits, *_)``.
        X: ``(N, G)`` input on the model's device.
        target_class: Logit index to attribute (0 = positive/stimulus for binary).
        baseline: ``(N, G)`` (or broadcastable) reference input.
        n_steps: Interpolation steps.

    Returns:
        ``(N, G)`` per-cell, per-gene attribution (``delta * mean grad``).
    """
    delta = X - baseline
    alphas = torch.linspace(
        0.5 / n_steps, 1.0 - 0.5 / n_steps, n_steps, device=X.device
    )
    grads_accum = torch.zeros_like(X)
    for alpha in alphas:
        x_interp = (baseline + alpha * delta).detach().clone().requires_grad_(True)
        logits = model(x_interp)[0]
        loss = logits[target_class]
        grad = torch.autograd.grad(loss, x_interp, create_graph=False)[0]
        grads_accum = grads_accum + grad
    return delta * (grads_accum / n_steps)


def _control_baseline(tube_set: PseudoTubeSet, device: torch.device) -> torch.Tensor:
    """Per-gene baseline = mean over control tubes of each tube's gene-mean (G,)."""
    ctrl = [t for t in tube_set.tubes if t.condition == tube_set.control_label]
    if not ctrl:
        raise SignatureError(
            f"No control ({tube_set.control_label!r}) tubes available to build the "
            "Integrated-Gradients baseline."
        )
    tube_means = np.stack([t.X.mean(axis=0) for t in ctrl], axis=0)  # (n_ctrl, G)
    base = tube_means.mean(axis=0).astype(np.float32)                # (G,)
    return torch.from_numpy(base).to(device)


def derive_signature(
    model: AbMil,
    tube_set: PseudoTubeSet,
    condition: str,
    *,
    control_label: str | None = None,
    top_n: int = 50,
    n_steps: int = 20,
    device: str | torch.device | None = None,
) -> Signature:
    """Discover ``condition``'s signature ``S_X`` from its trained binary model.

    Args:
        model: The trained binary AB-MIL for ``condition`` (positive class = 0).
        tube_set: The pseudo-tube set (provides the control baseline + gene order).
        condition: The stimulus whose signature to derive.
        control_label: Override the set's control label if needed.
        top_n: Signature size (top genes by mean IG).
        n_steps: IG interpolation steps.
        device: Where to run attribution.

    Returns:
        A :class:`Signature` of up to ``top_n`` genes, most-attributed first.

    Raises:
        SignatureError: if ``condition`` has no tubes, or no control baseline exists.
    """
    dev = resolve_device(device)
    if control_label is not None and control_label != tube_set.control_label:
        tube_set = PseudoTubeSet(
            tubes=tube_set.tubes,
            gene_names=tube_set.gene_names,
            control_label=control_label,
        )

    cond_tubes = [t for t in tube_set.tubes if t.condition == condition]
    if not cond_tubes:
        raise SignatureError(f"No tubes for condition {condition!r}; cannot derive S_X.")

    gene_names = tube_set.gene_names
    g = len(gene_names)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    baseline = _control_baseline(tube_set, dev)  # (G,)
    ig_accum = np.zeros(g, dtype=np.float64)
    n_used = 0
    for t in cond_tubes:
        X = torch.from_numpy(np.ascontiguousarray(t.X, dtype=np.float32)).to(dev)
        base = baseline.unsqueeze(0).expand_as(X).contiguous()
        ig = integrated_gradients(
            model, X, target_class=0, baseline=base, n_steps=n_steps
        )
        ig_accum += ig.mean(dim=0).detach().cpu().numpy()
        n_used += 1

    ig_mean = ig_accum / max(n_used, 1)
    order = np.argsort(-ig_mean)
    k = min(top_n, g)
    genes = tuple(gene_names[i] for i in order[:k])
    scores = tuple(float(ig_mean[i]) for i in order[:k])
    return Signature(condition=condition, genes=genes, ig_scores=scores, top_n=top_n)


def derive_signatures(
    models: dict[str, AbMil],
    tube_set: PseudoTubeSet,
    *,
    top_n: int = 50,
    n_steps: int = 20,
    device: str | torch.device | None = None,
) -> dict[str, Signature]:
    """Derive signatures for every trained binary model. Returns ``{condition: Signature}``."""
    out: dict[str, Signature] = {}
    for cond, model in models.items():
        logger.info("derive_signatures: %s", cond)
        out[cond] = derive_signature(
            model, tube_set, cond, top_n=top_n, n_steps=n_steps, device=device
        )
    return out


# ---------------------------------------------------------------------------
# Promiscuous-gene curation (OPT-IN)
#
# A gene that turns up in many conditions' signatures is, by construction, not
# condition-specific: it is part of the shared activation program every stimulus
# induces. Removing such genes from ALL signatures sharpens what remains. The one
# thing you must not do is pick the cap by hand — see
# `null_calibrated_max_occurrences` for why.
# ---------------------------------------------------------------------------


def signature_gene_occurrences(signatures: dict[str, Signature]) -> dict[str, int]:
    """How many signatures each gene appears in. ``{gene: count}``, unsorted."""
    return dict(Counter(g for sig in signatures.values() for g in sig.genes))


def null_expected_removal(n_conditions: int, top_n: int, n_genes: int, cap: int) -> float:
    """Expected fraction of ONE signature removed by ``cap`` under a uniform-random null.

    Under the null every signature is ``top_n`` genes drawn uniformly from ``n_genes``.
    A gene already in this signature is removed iff at least ``cap`` of the OTHER
    ``n_conditions - 1`` signatures also contain it, so the fraction removed is
    ``P(Binom(n_conditions - 1, top_n / n_genes) >= cap)``.
    """
    if n_conditions < 1 or top_n < 1 or n_genes < 1:
        raise ValueError("null_expected_removal: n_conditions, top_n, n_genes must be >= 1.")
    if cap < 1:
        raise ValueError(f"null_expected_removal: cap must be >= 1; got {cap}.")
    k_other = n_conditions - 1
    if cap > k_other:
        return 0.0
    p = min(top_n / n_genes, 1.0)
    below = sum(comb(k_other, k) * p**k * (1.0 - p) ** (k_other - k) for k in range(cap))
    return max(0.0, min(1.0, 1.0 - below))


def null_calibrated_max_occurrences(
    n_conditions: int,
    top_n: int,
    n_genes: int,
    *,
    target_null_removal: float = 0.1052,
) -> int:
    """Smallest occurrence cap whose *expected null damage* stays within a budget.

    A fixed cap means wildly different things at different scales, which makes
    hand-picked caps dangerous. With K signatures of n genes drawn from G, a gene's
    expected occurrence count under a uniform-random null is ``K * n / G`` — for
    K=90, n=200, G=4000 that is **4.5**, so a cap of 3 would delete ~83% of every
    signature even if the signatures were perfectly random. This function instead fixes
    the *stringency*: it returns the smallest cap for which
    :func:`null_expected_removal` does not exceed ``target_null_removal``.

    The default target is ``null_expected_removal(24, 200, 4000, cap=3) = 0.1052`` —
    i.e. "as stringent at your scale as a cap of 3 is on a 24-condition panel at
    top-200". Removal observed **in excess** of the returned cap's null expectation is
    a measure of how far the real signatures are from independent.

    Returns:
        The calibrated cap, in ``[1, n_conditions]``.
    """
    if not 0.0 < target_null_removal < 1.0:
        raise ValueError(
            "null_calibrated_max_occurrences: target_null_removal must be in (0, 1); "
            f"got {target_null_removal}."
        )
    for cap in range(1, n_conditions + 1):
        if null_expected_removal(n_conditions, top_n, n_genes, cap) <= target_null_removal:
            return cap
    return n_conditions


def curate_signatures(
    signatures: dict[str, Signature],
    *,
    max_occurrences: int,
    min_genes: int = 1,
) -> tuple[dict[str, Signature], pd.DataFrame]:
    """Remove genes occurring in more than ``max_occurrences`` signatures, from ALL of them.

    The removal is global and symmetric: a gene over the cap is dropped from every
    signature that contains it, including the one where it ranked first. ``genes`` and
    ``ig_scores`` are filtered with the same mask so they stay index-aligned, and the
    original ``top_n`` is preserved on the rebuilt :class:`Signature` (it records the
    size that was *requested*; ``len(genes)`` is what survived).

    Conditions left with fewer than ``min_genes`` genes are **dropped from the returned
    dict** — :class:`Signature` rejects an empty gene tuple, and a one-gene signature is
    not a signature. Dropped conditions simply disappear from downstream pair tables
    rather than producing NaNs, so compare a curated run against an uncurated one on the
    intersection of their conditions.

    Args:
        signatures: ``{condition: Signature}``, e.g. from :func:`derive_signatures`.
        max_occurrences: Cap on how many signatures a gene may appear in. Choose it with
            :func:`null_calibrated_max_occurrences`, not by hand.
        min_genes: Minimum surviving genes for a condition to be kept (default 1).

    Returns:
        ``(curated, report)``. ``report`` has one row per input condition with columns
        ``condition, n_before, n_after, n_removed, frac_removed, dropped``.

    Raises:
        ValueError: if ``max_occurrences`` or ``min_genes`` is < 1.
    """
    if max_occurrences < 1:
        raise ValueError(
            f"curate_signatures: max_occurrences must be >= 1; got {max_occurrences}."
        )
    if min_genes < 1:
        raise ValueError(f"curate_signatures: min_genes must be >= 1; got {min_genes}.")

    counts = signature_gene_occurrences(signatures)
    over_cap = {g for g, n in counts.items() if n > max_occurrences}

    curated: dict[str, Signature] = {}
    rows: list[dict] = []
    for condition, sig in signatures.items():
        keep = [i for i, g in enumerate(sig.genes) if g not in over_cap]
        n_after = len(keep)
        dropped = n_after < min_genes
        if not dropped:
            curated[condition] = Signature(
                condition=sig.condition,
                genes=tuple(sig.genes[i] for i in keep),
                ig_scores=tuple(sig.ig_scores[i] for i in keep),
                top_n=sig.top_n,
            )
        rows.append({
            "condition": condition,
            "n_before": len(sig.genes),
            "n_after": n_after,
            "n_removed": len(sig.genes) - n_after,
            "frac_removed": (len(sig.genes) - n_after) / max(len(sig.genes), 1),
            "dropped": dropped,
        })

    report = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)
    logger.info(
        "curate_signatures: cap=%d removed %d of %d distinct genes; "
        "signature size %d -> %d (median); dropped %d of %d conditions.",
        max_occurrences, len(over_cap), len(counts),
        int(report["n_before"].median()), int(report["n_after"].median()),
        int(report["dropped"].sum()), len(report),
    )
    return curated, report
