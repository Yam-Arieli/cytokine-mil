"""
Precomputed model output cache for auxiliary decoder training.

Eliminates all h5ad I/O from the training loop by running a single forward
pass through the frozen MIL model before training begins.
"""

from typing import Dict, List, Optional, Tuple

import scanpy as sc
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def build_cache(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
) -> List[Dict]:
    """
    Single forward pass through all tubes. Caches H, y_hat, label, donor,
    and cell_type labels in RAM.

    Cell-type labels are loaded from h5ad obs["cell_type"] (post-hoc only —
    never passed to the MIL model).

    Args:
        model: Frozen CytokineABMIL or CytokineABMIL_V2 (eval mode).
        dataset: PseudoTubeDataset with .entries attribute.
        device: torch device.

    Returns:
        List of dicts, one per tube:
            "H":          (N, embed_dim) float CPU tensor
            "y_hat":      (K,) probability CPU tensor (softmax of logits)
            "label":      int cytokine class index
            "donor":      str donor name
            "cell_types": list[str] of length N
    """
    model.eval()
    cache: List[Dict] = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            X, label, donor, _cytokine = dataset[idx]
            X = X.to(device)

            outputs = model(X)
            if len(outputs) == 3:
                logits, _a, H = outputs
            elif len(outputs) == 4:
                logits, _a_sa, _a_ca, H = outputs
            else:
                raise ValueError(
                    f"Unexpected model output length {len(outputs)}; "
                    "expected 3 (v1) or 4 (v2)."
                )

            if logits.dim() == 2:
                logits = logits.squeeze(0)

            y_hat = F.softmax(logits, dim=0).cpu()
            H_cpu = H.cpu()

            entry_path = dataset.entries[idx]["path"]
            cell_types = _load_cell_types(entry_path)

            cache.append({
                "H": H_cpu,
                "y_hat": y_hat,
                "label": int(label),
                "donor": donor,
                "cell_types": cell_types,
            })

            if (idx + 1) % 500 == 0:
                print(f"  build_cache: {idx + 1}/{len(dataset)} tubes", flush=True)

    return cache


def _load_cell_types(path: str) -> List[str]:
    """Load cell_type obs column from an h5ad file. Returns 'unknown' if absent."""
    adata = sc.read_h5ad(path)
    if "cell_type" in adata.obs.columns:
        return list(adata.obs["cell_type"].values)
    return ["unknown"] * adata.n_obs


class CachedTubeDataset(Dataset):
    """
    Wraps a precomputed cache list for aux decoder training.

    Each item is a tube whose H and y_hat were precomputed by build_cache.
    No disk I/O during iteration.

    Returns per item:
        H:          (N, embed_dim) float tensor
        y_hat:      (K,) probability tensor
        label:      int
        donor:      str
        cell_types: list[str]
    """

    def __init__(self, cache: List[Dict]) -> None:
        self._cache = cache

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Tuple:
        entry = self._cache[idx]
        return (
            entry["H"],
            entry["y_hat"],
            entry["label"],
            entry["donor"],
            entry["cell_types"],
        )
