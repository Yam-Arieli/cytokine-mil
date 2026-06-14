"""
Per-gene attribution for AB-MIL models (centralised).

Lifted from scripts/run_binary_ig_probe.py so the IG core can be reused at arbitrary
training checkpoints (gene-learning-order experiment). Two attributions:

- integrated_gradients : faithful, multi-step (PBS-mean baseline). Use for final / sanity.
- raw_gradient        : single backward pass, delta * dlogit/dx. ~n_steps cheaper —
                        the literal "follow the gradient" for per-epoch trajectories.

Both return (N_cells, G_genes); aggregate to per-gene with `mean_over_cells`.

The model forward must return (logits, attention, H) like CytokineABMIL.forward.
"""

from __future__ import annotations

import numpy as np
import torch


def integrated_gradients(model, X, target_class, baseline, n_steps: int = 20):
    """IG of logit[target_class] w.r.t. X (N, G). Returns (N, G)."""
    delta = X - baseline
    alphas = torch.linspace(0.5 / n_steps, 1.0 - 0.5 / n_steps, n_steps, device=X.device)
    grads = torch.zeros_like(X)
    for alpha in alphas:
        xi = (baseline + alpha * delta).detach().clone().requires_grad_(True)
        logits, _, _ = model(xi)
        grad = torch.autograd.grad(logits[target_class], xi, create_graph=False)[0]
        grads = grads + grad
    return delta * (grads / n_steps)


def raw_gradient(model, X, target_class, baseline=None):
    """Single-pass attribution: (X - baseline) * dlogit[target_class]/dX. Returns (N, G).
    If baseline is None, returns the plain input-gradient dlogit/dX."""
    xi = X.detach().clone().requires_grad_(True)
    logits, _, _ = model(xi)
    grad = torch.autograd.grad(logits[target_class], xi, create_graph=False)[0]
    return grad if baseline is None else (X - baseline) * grad


def mean_over_cells(attr: torch.Tensor) -> np.ndarray:
    """(N, G) per-cell attribution -> (G,) per-gene (mean across cells)."""
    return attr.detach().mean(dim=0).cpu().numpy().astype(np.float64)
