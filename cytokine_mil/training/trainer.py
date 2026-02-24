"""
Shared training helpers: mega-batch logic and cytokine queue management.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from cytokine_mil.models.cytokine_abmil import CytokineABMIL


def build_cytokine_queues(entries: List[dict], label_encoder) -> Dict[int, List[int]]:
    """
    Group dataset indices by cytokine label.

    Args:
        entries: Raw manifest entries from PseudoTubeDataset.get_entries().
        label_encoder: Fitted CytokineLabel instance.
    Returns:
        queues: dict mapping cytokine_index -> list of dataset indices.
    """
    queues: Dict[int, List[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        cyt_label = label_encoder.encode(entry["cytokine"])
        queues[cyt_label].append(idx)
    return dict(queues)


def sample_megabatch_indices(
    queues: Dict[int, List[int]], rng: np.random.Generator
) -> Dict[int, int]:
    """
    Sample one dataset index per cytokine to form a single mega-batch.

    Args:
        queues: dict mapping cytokine_index -> list of available indices.
        rng: numpy random generator.
    Returns:
        mapping of cytokine_index -> sampled dataset index.
    """
    return {cyt: int(rng.choice(indices)) for cyt, indices in queues.items()}


def generate_epoch_megabatches(
    queues: Dict[int, List[int]], rng: np.random.Generator
) -> List[Dict[int, int]]:
    """
    Generate one full epoch of mega-batch index mappings.

    Shuffles each cytokine's queue and pairs up tubes so each tube appears
    approximately once per epoch. Shorter queues are cycled as needed.

    Args:
        queues: dict mapping cytokine_index -> list of dataset indices.
        rng: numpy random generator.
    Returns:
        List of mega-batch dicts, each mapping cytokine_index -> dataset_index.
    """
    shuffled: Dict[int, List[int]] = {
        cyt: rng.permutation(indices).tolist() for cyt, indices in queues.items()
    }
    n_steps = max(len(v) for v in shuffled.values())

    # Cycle shorter queues to match the longest
    for cyt, indices in shuffled.items():
        deficit = n_steps - len(indices)
        if deficit > 0:
            extra = rng.choice(indices, size=deficit).tolist()
            shuffled[cyt] = indices + extra

    return [{cyt: shuffled[cyt][step] for cyt in shuffled} for step in range(n_steps)]


def train_one_megabatch(
    model: CytokineABMIL,
    optimizer: torch.optim.Optimizer,
    tubes_per_cytokine: Dict[int, Tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Accumulate gradients over all cytokines and step once.

    One mega-batch = one pseudo-tube from each cytokine condition.
    Loss is averaged across cytokines before the backward pass so that the
    gradient scale is independent of K.

    Args:
        model: CytokineABMIL in train mode.
        optimizer: SGD optimizer (zero_grad called inside).
        tubes_per_cytokine: dict mapping cytokine_index -> (X, label).
            X is (N, G) on the correct device; label is a Python int.
        criterion: CrossEntropyLoss (reduction='mean').
        device: torch device.
    Returns:
        Mean loss across cytokines (scalar float).
    """
    optimizer.zero_grad()
    total_loss = 0.0
    n = len(tubes_per_cytokine)

    for _cyt_idx, (X, label) in tubes_per_cytokine.items():
        X = X.to(device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)
        y_hat, _a, _H = model(X)
        loss = criterion(y_hat.unsqueeze(0), label_t) / n
        loss.backward()
        total_loss += loss.item()

    optimizer.step()
    return total_loss
