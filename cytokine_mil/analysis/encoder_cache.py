"""
Shared utilities for embedding training tubes with a (frozen) encoder.

Provides:
  - `_load_label_encoder` : load CytokineLabel from an experiment dir
  - `_embed_tube`         : encoder forward pass on a single AnnData tube
  - `build_cache`         : embed every training tube, return list of per-tube dicts
  - `compute_pbs_centroids` : per-cell-type PBS centroid from training PBS tubes only

Originally inlined in `scripts/run_geo_extract.py`; extracted so multiple
downstream analysis scripts (geo directional bias, alignment-based pair
detection, etc.) can share the same cache-building logic without
script-to-script imports.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import anndata
import numpy as np
import torch
from tqdm import tqdm

from cytokine_mil.data.label_encoder import CytokineLabel


VAL_DONORS = {"Donor2", "Donor3"}


def _load_label_encoder(exp_dir: Path) -> CytokineLabel:
    """Reconstruct a CytokineLabel from `label_encoder.json` in an experiment dir."""
    with open(exp_dir / "label_encoder.json") as f:
        data = json.load(f)
    cytos = data["cytokines"]
    le = CytokineLabel()
    le._label_to_idx = {c: i for i, c in enumerate(cytos)}
    le._idx_to_label = {i: c for i, c in enumerate(cytos)}
    return le


@torch.no_grad()
def _embed_tube(encoder, adata, gene_names, device):
    """Run encoder on a single tube's expression matrix → (N, embed_dim) CPU tensor."""
    if gene_names is not None:
        adata = adata[:, gene_names]
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_t = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)
    return encoder(X_t).cpu()


@torch.no_grad()
def build_cache(encoder, train_manifest, gene_names, label_encoder, device,
                val_donors=VAL_DONORS):
    """Run encoder on every training tube; return list of per-tube dicts.

    Tubes belonging to `val_donors` are skipped (donor-level hold-out).

    Each cache entry has keys:
        "H"          : torch.FloatTensor (N, embed_dim)  — cell embeddings
        "label"      : int                                — cytokine label index
        "cell_types" : list[str] of length N             — per-cell cell-type label
        "donor"      : str                                — donor identifier
    """
    encoder.eval()
    encoder.to(device)
    cache = []

    for entry in tqdm(train_manifest, desc="Embedding training tubes"):
        donor = entry.get("donor", "unknown")
        if donor in val_donors:
            continue
        try:
            adata = anndata.read_h5ad(entry["path"])
            H = _embed_tube(encoder, adata, gene_names, device)
            ct_labels = (
                adata.obs["cell_type"].values.tolist()
                if "cell_type" in adata.obs.columns
                else ["unknown"] * len(adata)
            )
            label = label_encoder.encode(entry["cytokine"])
            cache.append({
                "H":          H,
                "label":      label,
                "cell_types": ct_labels,
                "donor":      donor,
            })
        except Exception as e:
            print(f"  WARN: skipping {Path(entry['path']).name}: {e}", flush=True)

    return cache


@torch.no_grad()
def compute_pbs_centroids(encoder, train_manifest, gene_names, device,
                          val_donors=VAL_DONORS):
    """Compute per-cell-type PBS centroids using the given encoder.

    Only training donors' PBS tubes are used (excludes `val_donors`).

    Returns:
        dict: {cell_type_str: np.ndarray of shape (embed_dim,)}.
    """
    encoder.eval()
    encoder.to(device)

    ct_sums: dict = defaultdict(lambda: None)
    ct_counts: dict = defaultdict(int)

    pbs_tubes = [e for e in train_manifest
                 if e["cytokine"] == "PBS" and e.get("donor", "") not in val_donors]
    print(f"  Computing PBS centroids from {len(pbs_tubes)} PBS tubes...", flush=True)

    for entry in pbs_tubes:
        try:
            adata = anndata.read_h5ad(entry["path"])
            H = _embed_tube(encoder, adata, gene_names, device).numpy()
            ct_labels = (
                adata.obs["cell_type"].values
                if "cell_type" in adata.obs.columns
                else np.array(["unknown"] * len(adata))
            )
            for ct in np.unique(ct_labels):
                mask = ct_labels == ct
                ct_H = H[mask]
                if ct_sums[ct] is None:
                    ct_sums[ct] = ct_H.sum(axis=0)
                else:
                    ct_sums[ct] += ct_H.sum(axis=0)
                ct_counts[ct] += mask.sum()
        except Exception as e:
            print(f"  WARN: PBS tube {Path(entry['path']).name}: {e}", flush=True)

    pbs_ct_means = {ct: ct_sums[ct] / ct_counts[ct] for ct in ct_sums}
    print(f"  PBS centroids: {len(pbs_ct_means)} cell types, "
          f"{sum(ct_counts.values())} total cells", flush=True)
    return pbs_ct_means
