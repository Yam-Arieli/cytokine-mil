"""
Oesinghaus pseudotube loader for the §24 directional asymmetry pipeline.

Wraps ``cytokine_mil.analysis.eda_pair_benchmark.load_phase1_cells`` —
that function already pools cells by (cytokine, cell_type) from any
manifest whose entries carry ``path`` and ``cytokine`` keys and whose
``.h5ad`` files have ``adata.obs["cell_type"]``. The Oesinghaus manifest
satisfies both, so the only thing this wrapper adds is:

  1. **Cytokine subsetting** before reading — the full Oesinghaus manifest
     has ~10 000 tubes (91 cytokines × ~12 donors × ~10 tubes); we usually
     only need a handful. Filtering the manifest first keeps the file-IO
     bounded by ``len(cytokines) × n_donors × n_tubes_per_donor``.

  2. **Tube cap** (``max_tubes_per_cytokine``) for cheap smoke runs.

  3. **Implicit PBS inclusion** — ``directional_asymmetry_test`` always
     needs PBS cells per (cytokine, cell_type), so we add ``PBS`` to the
     subset unconditionally unless the caller passes it explicitly.

Allowed imports (must stay in sync with the pipeline-driver dependency
audit): ``json, pathlib, typing, numpy``. ``load_phase1_cells`` itself
pulls in ``scanpy`` + ``scipy.sparse``, but those are already required by
the rest of the analysis package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cytokine_mil.analysis.eda_pair_benchmark import load_phase1_cells


def load_oesinghaus_cells_by_pair(
    manifest_path: str,
    cytokines: Sequence[str],
    hvg_path: str,
    max_tubes_per_cytokine: Optional[int] = None,
    pbs_label: str = "PBS",
    include_donors: Optional[Sequence[str]] = None,
    exclude_donors: Optional[Sequence[str]] = None,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], List[str]]:
    """
    Load Oesinghaus pseudotubes restricted to a cytokine subset, pooled by
    (cytokine, cell_type).

    Args:
        manifest_path: Path to ``manifest.json``.
        cytokines:     Cytokines to include. ``pbs_label`` is always added.
        hvg_path:      Path to ``hvg_list.json`` — used to align gene columns
                       across tubes (the same set the binary models were
                       trained on).
        max_tubes_per_cytokine: If set, keep only the first N manifest
                       entries per cytokine (after manifest's own ordering).
                       Useful for smoke tests.
        pbs_label:     Control class name in the manifest (default "PBS").
        include_donors: If set, only manifest entries whose donor is in this
                       list are kept. Mutually exclusive with exclude_donors
                       (exclude_donors overrides if both given).
        exclude_donors: If set, manifest entries whose donor is in this list
                       are dropped. Useful for "use train donors only"
                       (exclude_donors=["Donor2","Donor3"]) to keep §24
                       consistent with the binary models' training split.

    Returns:
        cells_by_pair: ``{(cytokine, cell_type) -> (N_cells, G_hvgs) float32 array}``.
        gene_names:    HVG names in column order.
    """
    manifest_path = str(manifest_path)
    hvg_path = str(hvg_path)

    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"manifest_path does not exist: {manifest_path}")
    if not Path(hvg_path).exists():
        raise FileNotFoundError(f"hvg_path does not exist: {hvg_path}")

    # ---- HVG list (column-order anchor for all tubes) ----
    with open(hvg_path) as f:
        gene_names = json.load(f)
    if not isinstance(gene_names, list) or not gene_names:
        raise ValueError(f"HVG file {hvg_path} is not a non-empty JSON list.")

    # ---- Filter manifest down to the cytokine subset (+ optional donor filter) ----
    with open(manifest_path) as f:
        full_manifest = json.load(f)

    keep_cyt = set(cytokines) | {pbs_label}

    # Resolve donor inclusion/exclusion to a single positive include set.
    available_donors = sorted({e.get("donor") for e in full_manifest if e.get("donor")})
    if exclude_donors:
        donor_include_set: Optional[set] = {
            d for d in available_donors if d not in set(exclude_donors)
        }
    elif include_donors:
        donor_include_set = set(include_donors)
    else:
        donor_include_set = None  # keep all donors

    per_cyt_count: Dict[str, int] = {c: 0 for c in keep_cyt}
    filtered: List[dict] = []
    for entry in full_manifest:
        cyt = entry.get("cytokine")
        if cyt not in keep_cyt:
            continue
        if (donor_include_set is not None
                and entry.get("donor") not in donor_include_set):
            continue
        if max_tubes_per_cytokine is not None:
            if per_cyt_count[cyt] >= max_tubes_per_cytokine:
                continue
            per_cyt_count[cyt] += 1
        filtered.append(entry)

    if not filtered:
        raise ValueError(
            f"No manifest entries matched cytokines={sorted(keep_cyt)}. "
            f"Available in manifest: "
            f"{sorted({e.get('cytokine') for e in full_manifest})}"
        )

    # ---- Write filtered manifest to a temp path so load_phase1_cells can read it ----
    # We persist it next to the original manifest so it can be re-inspected
    # post hoc, but with a deterministic name keyed off the cytokine list.
    cyt_hash = "_".join(sorted(c.replace("/", "_") for c in keep_cyt))[:80]
    tmp_manifest_path = (
        Path(manifest_path).parent
        / f"_oesinghaus_filtered_{cyt_hash}.json"
    )
    with open(tmp_manifest_path, "w") as fh:
        json.dump(filtered, fh)

    # ---- Reuse the generic loader ----
    # We already filtered by donor at the manifest level above, so pass
    # donors=None to load_phase1_cells (its donor filter would apply the
    # same constraint redundantly).
    cells_by_pair, resolved_genes = load_phase1_cells(
        manifest_path=str(tmp_manifest_path),
        gene_names=gene_names,
        time_filter=None,        # Oesinghaus is 24h-only, no time field
        donors=None,
    )

    # The loader returns resolved_genes from the first tube it reads; align to
    # the canonical hvg_path order. If they differ, prefer the hvg_path order
    # (which is what the binary models were trained against).
    if resolved_genes and resolved_genes != gene_names:
        # Build a column re-index map from resolved_genes -> gene_names.
        # If load_phase1_cells already subset to gene_names via the
        # ``gene_names=`` argument, resolved_genes will equal gene_names; this
        # branch is only a defensive guard.
        idx = {g: i for i, g in enumerate(resolved_genes)}
        try:
            perm = np.array([idx[g] for g in gene_names], dtype=np.int64)
        except KeyError as e:
            raise RuntimeError(
                f"HVG gene {e} present in hvg_list but missing from the "
                f"first tube's var_names; data inconsistency."
            )
        cells_by_pair = {
            k: v[:, perm].copy() for k, v in cells_by_pair.items()
        }

    return cells_by_pair, gene_names
