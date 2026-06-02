"""In-memory pseudo-tube construction (no files written).

A pseudo-tube is a *bag* of cells drawn from one ``(condition, donor)``, sampled
**stratified by cell type** so that differences in cell-type abundance do not drive
the learned signal. Variable tube sizes are preserved on purpose (a condition with
fewer eligible cell types yields smaller tubes — that is biological signal, not
noise to be equalized).

This is the decoupled, file-free replacement for the source project's
manifest+h5ad folder: it returns a :class:`PseudoTubeSet` held entirely in memory.
"""

from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData

from cascadir.exceptions import DataValidationError, InsufficientDataError
from cascadir.types import PseudoTube, PseudoTubeSet


def _sample_one_tube(
    cell_types: np.ndarray,
    rng: np.random.Generator,
    n_per_cell_type: int,
    min_cells: int,
) -> tuple[np.ndarray, list[str]]:
    """Stratified sample of row positions for one tube.

    For each cell type with >= ``min_cells`` cells, draw up to ``n_per_cell_type``
    rows without replacement. Returns (sorted row indices, cell types included).
    Faithful to the validated sampler.
    """
    idx: list[int] = []
    types_included: list[str] = []
    for ct in sorted(set(cell_types.tolist())):
        ct_pos = np.where(cell_types == ct)[0]
        if len(ct_pos) < min_cells:
            continue
        take = min(n_per_cell_type, len(ct_pos))
        chosen = rng.choice(ct_pos, size=take, replace=False)
        idx.extend(chosen.tolist())
        types_included.append(ct)
    return np.array(sorted(idx), dtype=int), types_included


def _dense_rows(X, mask: np.ndarray) -> np.ndarray:
    """Return X[mask] as a dense float32 array (sparse-aware)."""
    sub = X[mask]
    if hasattr(sub, "toarray"):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float32)


def build_pseudotubes(
    adata: AnnData,
    *,
    condition_col: str,
    donor_col: str,
    celltype_col: str,
    control_label: str = "PBS",
    n_per_cell_type: int = 30,
    min_cells: int = 10,
    n_tubes: int = 10,
    seed: int = 0,
) -> PseudoTubeSet:
    """Build pseudo-tubes from a preprocessed AnnData, fully in memory.

    Args:
        adata: cells x genes AnnData, already log-normalized and HVG-subset (see
            :func:`cascadir.preprocess.preprocess`). ``obs`` must carry the three
            named columns.
        condition_col / donor_col / celltype_col: ``obs`` column names.
        control_label: The control condition (must appear among the tubes).
        n_per_cell_type: Cells sampled per cell type per tube.
        min_cells: Per-cell-type and per-tube minimum cell count.
        n_tubes: Tubes built per ``(condition, donor)``.
        seed: RNG seed.

    Returns:
        A :class:`PseudoTubeSet`.

    Raises:
        DataValidationError: if a required ``obs`` column is missing, or the control
            condition ends up unrepresented.
        InsufficientDataError: if no tube could be built (too few cells everywhere).
    """
    for c in (condition_col, donor_col, celltype_col):
        if c not in adata.obs:
            raise DataValidationError(
                f"build_pseudotubes: obs is missing column {c!r}. "
                f"Present: {list(adata.obs.columns)}."
            )

    rng = np.random.default_rng(seed)
    gene_names = tuple(map(str, adata.var_names))
    conditions = adata.obs[condition_col].astype(str).to_numpy()
    donors = adata.obs[donor_col].astype(str).to_numpy()
    cell_types_all = adata.obs[celltype_col].astype(str).to_numpy()

    tubes: list[PseudoTube] = []
    pairs = sorted(set(zip(conditions.tolist(), donors.tolist())))
    for cond, donor in pairs:
        mask = (conditions == cond) & (donors == donor)
        if int(mask.sum()) < min_cells:
            continue
        X_sub = _dense_rows(adata.X, mask)
        ct_sub = cell_types_all[mask]
        for t in range(n_tubes):
            idx, types_inc = _sample_one_tube(ct_sub, rng, n_per_cell_type, min_cells)
            if len(idx) < min_cells:
                continue
            tubes.append(
                PseudoTube(
                    X=X_sub[idx].copy(),
                    condition=str(cond),
                    donor=str(donor),
                    cell_types=tuple(ct_sub[idx].tolist()),
                    cell_types_included=tuple(types_inc),
                    tube_idx=t,
                )
            )

    if not tubes:
        raise InsufficientDataError(
            "build_pseudotubes produced no tubes — every (condition, donor) had "
            f"fewer than min_cells={min_cells} usable cells. Lower min_cells / "
            "n_per_cell_type, or check your cell-type labels."
        )
    if control_label not in {t.condition for t in tubes}:
        raise DataValidationError(
            f"control_label {control_label!r} is not represented among the built "
            "tubes; cross_asym requires a control baseline. Check that control cells "
            "survived preprocessing and have >= min_cells per (donor, cell_type)."
        )
    return PseudoTubeSet(
        tubes=tubes, gene_names=gene_names, control_label=control_label
    )


class InMemoryTubeDataset:
    """A minimal torch-style dataset over a :class:`PseudoTubeSet`.

    Implements exactly the interface the trainer needs (``__len__``,
    ``__getitem__`` -> ``(X, label, donor, condition)``, ``get_entries``,
    ``label_encoder``) so no on-disk ``PseudoTubeDataset`` is required.

    Args:
        tube_set: The tubes to serve. For a binary model, pass a ``tube_set``
            already filtered to ``{positive, control}``.
        label_encoder: Something with ``encode(str)->int`` (e.g.
            :class:`cascadir.types.BinaryLabel`). Every tube's condition must be
            encodable by it.
    """

    def __init__(self, tube_set: PseudoTubeSet, label_encoder) -> None:
        self.tube_set = tube_set
        self.label_encoder = label_encoder
        self._entries: list[dict] = []
        for t in tube_set.tubes:
            # encode eagerly so a mismatched label fails loudly at construction
            label_encoder.encode(t.condition)
            self._entries.append(
                {
                    "condition": t.condition,
                    "cytokine": t.condition,  # alias for trainer compatibility
                    "donor": t.donor,
                    "tube_idx": t.tube_idx,
                    "n_cells": t.n_cells,
                    "cell_types_included": list(t.cell_types_included),
                }
            )

    def __len__(self) -> int:
        return len(self.tube_set.tubes)

    def get_entries(self) -> list[dict]:
        return self._entries

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, str]:
        t = self.tube_set.tubes[idx]
        X = torch.from_numpy(np.ascontiguousarray(t.X, dtype=np.float32))
        label = int(self.label_encoder.encode(t.condition))
        return X, label, t.donor, t.condition
