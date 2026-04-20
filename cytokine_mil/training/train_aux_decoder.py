"""
Train the AuxDecoder on a precomputed in-memory tube cache.

Uses SGD with momentum (consistent with the rest of training, per CLAUDE.md §7)
and MSE loss between per-cell decoder predictions and the cached bag-level
softmax probabilities.

Loss per tube:
    pred_i   = softmax( decoder(h_i) )        # (N, K) per-cell prediction
    loss     = MSE( pred_i, ŷ_bag.expand(N) ) # ŷ_bag = cached softmax(model_logits)

No disk I/O during training — all data is read from the in-memory cache
built by training.cache.build_cache.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.optim import SGD

from cytokine_mil.models.aux_decoder import AuxDecoder


def train_aux_decoder(
    model: AuxDecoder,
    cache: List[Dict],
    n_epochs: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
    momentum: float = 0.9,
    verbose: bool = True,
) -> AuxDecoder:
    """
    Train AuxDecoder on precomputed cache with SGD+momentum and MSE loss.

    Args:
        model:    AuxDecoder instance (untrained).
        cache:    List of dicts from build_cache; each has keys "H" and "y_hat".
        n_epochs: Number of full passes over the cache.
        lr:       Learning rate for SGD.
        device:   torch device.
        seed:     Random seed for reproducible parameter init / shuffle order.
        momentum: SGD momentum (default 0.9, consistent with MIL training).
        verbose:  Print epoch loss summary.

    Returns:
        Trained AuxDecoder (same object, mutated in-place) in eval mode.
    """
    torch.manual_seed(seed)
    model.to(device)
    model.train()

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    n_tubes = len(cache)

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0

        for entry in cache:
            H = entry["H"].to(device)           # (N, input_dim)
            y_hat = entry["y_hat"].to(device)   # (K,) bag-level softmax probs

            logits = model(H)                   # (N, n_classes)
            pred = F.softmax(logits, dim=1)     # (N, K) per-cell predictions

            target = y_hat.unsqueeze(0).expand_as(pred)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        mean_loss = epoch_loss / n_tubes
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == n_epochs):
            print(f"  epoch {epoch:3d}/{n_epochs}  mean_loss={mean_loss:.6f}", flush=True)

    model.eval()
    return model
