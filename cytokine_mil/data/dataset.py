"""
Dataset classes for pseudo-tube bags and cell-level encoder pre-training.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset

from cytokine_mil.data.label_encoder import CytokineLabel


class PseudoTubeDataset(Dataset):
    """
    Dataset for pseudo-tube bags used in Stage 2/3 MIL training.

    Reads manifest.json at init. Loads one .h5ad file per __getitem__ call.
    Returns (X, label, donor, cytokine_name) where X is a float tensor of
    shape (N_cells, N_genes).
    """

    def __init__(
        self,
        manifest_path: str,
        label_encoder: CytokineLabel,
        gene_names: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            manifest_path: Path to manifest.json.
            label_encoder: Fitted CytokineLabel instance.
            gene_names: If provided, only these genes are returned (in order).
        """
        self.label_encoder = label_encoder
        self.gene_names = gene_names
        self.entries = self._read_manifest(manifest_path)

    def _read_manifest(self, path: str) -> List[dict]:
        with open(path) as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        entry = self.entries[idx]
        adata = sc.read_h5ad(entry["path"])
        X = self._extract_matrix(adata)
        label = self.label_encoder.encode(entry["cytokine"])
        return X, label, entry["donor"], entry["cytokine"]

    def _extract_matrix(self, adata) -> torch.Tensor:
        if self.gene_names is not None:
            adata = adata[:, self.gene_names]
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return torch.FloatTensor(np.asarray(X, dtype=np.float32))

    def get_entries(self) -> List[dict]:
        """Return raw manifest entries for index-building by the trainer."""
        return self.entries


class CellDataset(Dataset):
    """
    Flat cell-level dataset for Stage 1 encoder pre-training.

    Reads pseudo-tube .h5ad files and exposes individual cells with
    their cell_type labels.

    preload=True (recommended for Stage 1): loads all tubes at init into a
    contiguous numpy array — enables fast shuffle without disk I/O per batch.
    Use with a subsampled manifest (one tube per cytokine ≈ 91 tubes ≈ 40k
    cells ≈ 640 MB) to keep memory usage reasonable.

    preload=False: lazy loading with an LRU tube cache. Shuffle should be
    disabled (num_workers=0, shuffle=False) or the cache will be ineffective.
    """

    def __init__(
        self,
        manifest_path: str,
        gene_names: Optional[List[str]] = None,
        tube_cache_size: int = 64,
        preload: bool = False,
    ) -> None:
        """
        Args:
            manifest_path: Path to manifest.json.
            gene_names: Optional list of gene names to subset columns.
            tube_cache_size: Number of tubes held in the LRU cache (preload=False only).
            preload: If True, load all data into memory at init.
        """
        self.gene_names = gene_names

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.cell_type_to_idx = self._build_cell_type_map(manifest)

        if preload:
            self._X, self._labels = self._preload(manifest)
            self._preloaded = True
        else:
            self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            self._cache_order: List[str] = []
            self._cache_size = tube_cache_size
            self._index = self._build_index(manifest)
            self._preloaded = False

    def _build_cell_type_map(self, manifest: List[dict]) -> Dict[str, int]:
        all_types: set = set()
        for entry in manifest:
            all_types.update(entry.get("cell_types_included", []))
        return {ct: i for i, ct in enumerate(sorted(all_types))}

    # ------------------------------------------------------------------
    # preload path
    # ------------------------------------------------------------------

    def _preload(self, manifest: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Load all tubes sequentially and concatenate into contiguous arrays."""
        all_X: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        for entry in manifest:
            X, cell_types = self._read_tube(entry["path"])
            labels = np.array(
                [self.cell_type_to_idx.get(str(ct), 0) for ct in cell_types],
                dtype=np.int64,
            )
            all_X.append(X)
            all_labels.append(labels)
        return np.concatenate(all_X, axis=0), np.concatenate(all_labels, axis=0)

    # ------------------------------------------------------------------
    # lazy-load path
    # ------------------------------------------------------------------

    def _build_index(self, manifest: List[dict]) -> List[Tuple[str, int]]:
        """Build a flat list of (tube_path, within_tube_cell_idx) pairs."""
        index = []
        for entry in manifest:
            n_cells = entry["n_cells"]
            for i in range(n_cells):
                index.append((entry["path"], i))
        return index

    def _load_tube(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if path not in self._cache:
            self._evict_if_full()
            X, cell_types = self._read_tube(path)
            self._cache[path] = (X, cell_types)
            self._cache_order.append(path)
        return self._cache[path]

    def _evict_if_full(self) -> None:
        if len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

    # ------------------------------------------------------------------
    # shared I/O helper
    # ------------------------------------------------------------------

    def _read_tube(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        adata = sc.read_h5ad(path)
        if self.gene_names is not None:
            adata = adata[:, self.gene_names]
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        cell_types = (
            adata.obs["cell_type"].values
            if "cell_type" in adata.obs.columns
            else np.full(len(X), "unknown")
        )
        return X, cell_types

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def n_cell_types(self) -> int:
        return len(self.cell_type_to_idx)

    def __len__(self) -> int:
        return len(self._X) if self._preloaded else len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self._preloaded:
            return torch.FloatTensor(self._X[idx]), int(self._labels[idx])
        path, cell_idx = self._index[idx]
        X, cell_types = self._load_tube(path)
        x = torch.FloatTensor(X[cell_idx])
        label = self.cell_type_to_idx.get(str(cell_types[cell_idx]), 0)
        return x, label


def collate_fn(
    batch: List[Tuple[torch.Tensor, int, str, str]],
) -> Tuple[List[torch.Tensor], List[int], List[str], List[str]]:
    """
    Custom collate for variable-length pseudo-tube bags.

    Returns lists rather than stacked tensors because tubes may have
    different numbers of cells (N). The trainer processes each tube
    individually in the mega-batch loop.
    """
    Xs, labels, donors, cytokine_names = zip(*batch)
    return list(Xs), list(labels), list(donors), list(cytokine_names)
