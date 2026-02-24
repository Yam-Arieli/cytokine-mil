"""
Stage 1: pre-train the InstanceEncoder with supervised cell-type classification.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cytokine_mil.models.instance_encoder import InstanceEncoder


def train_encoder(
    encoder: InstanceEncoder,
    dataloader: DataLoader,
    n_epochs: int,
    lr: float = 0.01,
    momentum: float = 0.9,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> InstanceEncoder:
    """
    Pre-train the InstanceEncoder using cell-type labels.

    The encoder must have been constructed with n_cell_types set so that
    encoder.cell_type_head is available. After this function returns, only
    the backbone weights (not the head) are used in Stage 2.

    Args:
        encoder: InstanceEncoder with cell_type_head.
        dataloader: DataLoader over a CellDataset yielding (x, cell_type_label).
        n_epochs: Number of training epochs.
        lr: SGD learning rate.
        momentum: SGD momentum.
        device: Target device. Defaults to CUDA if available.
        verbose: Print per-epoch metrics.
    Returns:
        The trained encoder (in-place, also returned for convenience).
    """
    if encoder.cell_type_head is None:
        raise ValueError(
            "encoder.cell_type_head is None — rebuild InstanceEncoder with n_cell_types set."
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, n_epochs + 1):
        epoch_loss, n_correct, n_total = _run_epoch(
            encoder, dataloader, optimizer, criterion, device
        )
        if verbose:
            acc = n_correct / max(n_total, 1)
            print(
                f"[Stage 1] Epoch {epoch:3d}/{n_epochs} | "
                f"loss={epoch_loss:.4f} | acc={acc:.4f}"
            )

    return encoder


def _run_epoch(
    encoder: InstanceEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Run one training epoch. Returns (mean_loss, n_correct, n_total)."""
    encoder.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0

    for X, labels in tqdm(dataloader, leave=False):
        X = X.to(device)
        labels = labels.to(device)

        h = encoder(X)
        logits = encoder.cell_type_head(h)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(X)
        n_correct += (logits.argmax(dim=1) == labels).sum().item()
        n_total += len(X)

    return total_loss / max(n_total, 1), n_correct, n_total
