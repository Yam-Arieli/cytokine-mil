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

import numpy as np
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
