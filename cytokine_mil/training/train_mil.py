"""
Stage 2/3: train the full CytokineABMIL model with mega-batch gradient accumulation.

Returns a dynamics dict containing per-tube learning trajectories that are
consumed by cytokine_mil.analysis.dynamics for cascade inference.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.models.cytokine_abmil import CytokineABMIL
from cytokine_mil.training.trainer import (
    build_cytokine_queues,
    generate_epoch_megabatches,
)


def train_mil(
    model: CytokineABMIL,
    dataset: PseudoTubeDataset,
    n_epochs: int,
    lr: float = 0.01,
    momentum: float = 0.9,
    lr_scheduler: Optional[str] = None,
    lr_warmup_epochs: int = 0,
    log_every_n_epochs: int = 1,
    device: Optional[torch.device] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Train the CytokineABMIL model and record dynamics trajectories.

    Uses mega-batch gradient accumulation: one tube per cytokine per step,
    gradients accumulated across all cytokines, optimizer stepped once.
    SGD with momentum is used to produce smooth, monotonic learning curves
    suitable for dynamics analysis.

    Args:
        model: CytokineABMIL (encoder may be frozen for Stage 2).
        dataset: PseudoTubeDataset.
        n_epochs: Training epochs.
        lr: SGD learning rate.
        momentum: SGD momentum.
        lr_scheduler: Optional scheduler type ('cosine' or None).
        lr_warmup_epochs: Linear warmup epochs (0 to disable).
        log_every_n_epochs: Frequency of dynamics snapshots.
        device: Target device.
        seed: RNG seed for reproducible mega-batch sampling.
        verbose: Print per-epoch loss.
    Returns:
        dynamics: dict with keys:
            'logged_epochs': list of epoch indices where dynamics were recorded.
            'records': list of per-tube dicts with trajectory data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, lr, momentum)
    scheduler = _build_scheduler(optimizer, n_epochs, lr_scheduler, lr_warmup_epochs)
    rng = np.random.default_rng(seed)

    entries = dataset.get_entries()
    queues = build_cytokine_queues(entries, dataset.label_encoder)

    logged_epochs: List[int] = []
    tube_trajectories: Dict[int, Dict] = _init_tube_trajectories(entries)

    for epoch in range(1, n_epochs + 1):
        epoch_loss = _train_epoch(model, dataset, queues, optimizer, criterion, device, rng)

        if lr_warmup_epochs > 0 and epoch <= lr_warmup_epochs:
            _apply_warmup(optimizer, lr, epoch, lr_warmup_epochs)
        elif scheduler is not None:
            scheduler.step()

        if epoch % log_every_n_epochs == 0 or epoch == n_epochs:
            _log_dynamics(model, dataset, entries, tube_trajectories, device)
            logged_epochs.append(epoch)

        if verbose:
            print(f"[Stage 2/3] Epoch {epoch:3d}/{n_epochs} | loss={epoch_loss:.4f}")

    records = _build_records(entries, tube_trajectories)
    return {"logged_epochs": logged_epochs, "records": records}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_optimizer(
    model: CytokineABMIL, lr: float, momentum: float
) -> torch.optim.SGD:
    return torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
    )


def _build_scheduler(optimizer, n_epochs, scheduler_type, warmup_epochs):
    if scheduler_type == "cosine":
        effective_epochs = max(n_epochs - warmup_epochs, 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_epochs)
    return None


def _apply_warmup(optimizer, base_lr: float, epoch: int, warmup_epochs: int) -> None:
    scale = epoch / warmup_epochs
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


def _train_epoch(
    model: CytokineABMIL,
    dataset: PseudoTubeDataset,
    queues: Dict[int, List[int]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    """Run one epoch of mega-batch training. Returns mean loss."""
    model.train()
    megabatches = generate_epoch_megabatches(queues, rng)
    total_loss = 0.0

    for mb_indices in tqdm(megabatches, leave=False):
        optimizer.zero_grad()
        mb_loss = 0.0
        n = len(mb_indices)

        for _cyt_idx, ds_idx in mb_indices.items():
            X, label, _donor, _cyt_name = dataset[ds_idx]
            X = X.to(device)
            label_t = torch.tensor([label], dtype=torch.long, device=device)

            y_hat, _a, _H = model(X)
            loss = criterion(y_hat.unsqueeze(0), label_t) / n
            loss.backward()
            mb_loss += loss.item()

        optimizer.step()
        total_loss += mb_loss

    return total_loss / max(len(megabatches), 1)


def _init_tube_trajectories(entries: List[dict]) -> Dict[int, Dict]:
    """Create an empty trajectory dict keyed by dataset index."""
    return {
        i: {
            "p_correct": [],
            "entropy": [],
            "instance_confidence_final": None,
        }
        for i in range(len(entries))
    }


@torch.no_grad()
def _log_dynamics(
    model: CytokineABMIL,
    dataset: PseudoTubeDataset,
    entries: List[dict],
    tube_trajectories: Dict[int, Dict],
    device: torch.device,
) -> None:
    """
    Evaluate all tubes and append one snapshot to each tube's trajectory.

    Runs in eval mode with no_grad to avoid memory accumulation.
    """
    model.eval()
    for idx, entry in enumerate(entries):
        X, label, _donor, _cyt_name = dataset[idx]
        X = X.to(device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)

        y_hat, a, _H = model(X)
        probs = F.softmax(y_hat, dim=0)
        p_correct = probs[label].item()

        entropy = _compute_entropy(a)
        instance_conf = (a * p_correct).cpu().numpy()

        traj = tube_trajectories[idx]
        traj["p_correct"].append(p_correct)
        traj["entropy"].append(entropy)
        traj["instance_confidence_final"] = instance_conf  # overwrite each epoch


def _compute_entropy(a: torch.Tensor) -> float:
    """Shannon entropy of attention weights (nats). Clipped for stability."""
    a_safe = a.clamp(min=1e-10)
    return float(-(a_safe * a_safe.log()).sum())


def _build_records(
    entries: List[dict], tube_trajectories: Dict[int, Dict]
) -> List[Dict]:
    """Combine manifest metadata with trajectory data into flat records."""
    records = []
    for idx, entry in enumerate(entries):
        traj = tube_trajectories[idx]
        records.append(
            {
                "cytokine": entry["cytokine"],
                "donor": entry["donor"],
                "tube_idx": entry["tube_idx"],
                "tube_path": entry["path"],
                "n_cells": entry["n_cells"],
                "p_correct_trajectory": traj["p_correct"],
                "entropy_trajectory": traj["entropy"],
                "instance_confidence_final": traj["instance_confidence_final"],
            }
        )
    return records
