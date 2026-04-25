"""
Method 1: Compute PBS-RC asymmetry matrix at each checkpoint (fast, no permutations).

For each checkpoint epoch:
  - Load model weights
  - Build h_i cache (encoder embeddings + cell_types)
  - Apply PBS-RC transform
  - Compute cytokine centroids
  - Compute per-cell-type directional bias
  - Compute ASYM(A,B) = max_T [bias(A,B,T) - bias(B,A,T)]

Permutation tests are skipped — we only need the raw asymmetry trajectory here.
Significance testing is done at the final epoch using the full geo results.

Output: <run_dir>/geo_trajectory.pkl
  {
    "epochs":           [25, 50, 75, 100],
    "asymmetry_traj":   np.array(n_epochs, K, K),
    "cytokine_names":   list[str],
  }

Usage:
    python scripts/run_geo_trajectory.py --run_dir <dir>
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
import sys; sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model
from cytokine_mil.training.cache import build_cache
from scripts.run_experiment_geo import (
    make_pbs_relative_fn,
    precompute_transform_means,
)

HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


def _load_label_encoder(run_dir: Path) -> CytokineLabel:
    with open(run_dir / "label_encoder.json") as f:
        data = json.load(f)
    le = CytokineLabel()
    le._label_to_idx = {c: i for i, c in enumerate(data["cytokines"])}
    le._idx_to_label = {i: c for i, c in enumerate(data["cytokines"])}
    return le


def _load_model(run_dir: Path, ckpt_path: Path, label_enc, gene_names, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_cell_types = state["encoder.cell_type_head.weight"].shape[0]
    encoder = build_encoder(len(gene_names), n_cell_types=n_cell_types, embed_dim=128)
    model = build_mil_model(encoder, embed_dim=128, attention_hidden_dim=64,
                            n_classes=label_enc.n_classes(), encoder_frozen=True)
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def compute_fast_asymmetry(cache, label_encoder, pbs_fn):
    """
    Compute asymmetry matrix without permutation testing.

    Returns:
        asym_matrix: np.array(K, K)  — ASYM(A,B) = max_T [bias(A,B,T) - bias(B,A,T)]
        cytokine_names: list[str]
    """
    cytokine_names = [label_encoder._idx_to_label[i]
                      for i in range(label_encoder.n_classes())]
    K = len(cytokine_names)

    # Apply PBS-RC transform and collect per-cytokine, per-cell-type embeddings
    # cyt_ct_embeddings[cyt_idx][cell_type] = list of h_i vectors
    cyt_ct_embeddings = defaultdict(lambda: defaultdict(list))
    cyt_embeddings    = defaultdict(list)   # all cells in cyt (any cell type)

    for entry in cache:
        H_np      = entry["H"].numpy().astype(np.float64)
        ct_labels = entry["cell_types"]
        cyt_idx   = entry["label"]

        H_transformed = pbs_fn(H_np, ct_labels)

        for i, ct in enumerate(ct_labels):
            cyt_ct_embeddings[cyt_idx][ct].append(H_transformed[i])
            cyt_embeddings[cyt_idx].append(H_transformed[i])

    # Cytokine centroids (mean of all transformed cells per cytokine)
    mu = {}
    for cyt_idx in range(K):
        if cyt_idx in cyt_embeddings and cyt_embeddings[cyt_idx]:
            mu[cyt_idx] = np.mean(cyt_embeddings[cyt_idx], axis=0)

    # Per-cell-type centroids within each cytokine
    mu_ct = {}  # (cyt_idx, cell_type) -> mean
    for cyt_idx, ct_dict in cyt_ct_embeddings.items():
        for ct, vecs in ct_dict.items():
            mu_ct[(cyt_idx, ct)] = np.mean(vecs, axis=0)

    # All observed cell types
    all_cts = sorted({ct for ct_dict in cyt_ct_embeddings.values() for ct in ct_dict})

    # Directional bias: bias(A, B, T) = (mu_{A,T} - mu_A) · (mu_B - mu_A) / ||mu_B - mu_A||
    # Asymmetry: ASYM(A,B) = max_T [bias(A,B,T) - bias(B,A,T)]
    asym_matrix = np.zeros((K, K), dtype=np.float64)

    for a in range(K):
        if a not in mu:
            continue
        for b in range(K):
            if a == b or b not in mu:
                continue
            direction = mu[b] - mu[a]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction_unit = direction / norm

            max_asym = -np.inf
            for ct in all_cts:
                mu_at = mu_ct.get((a, ct))
                mu_bt = mu_ct.get((b, ct))
                if mu_at is None or mu_bt is None:
                    continue
                bias_ab = np.dot(mu_at - mu[a], direction_unit)
                # bias(B,A,T): uses B's centroid and A as target direction
                direction_ba = mu[a] - mu[b]
                norm_ba = np.linalg.norm(direction_ba)
                if norm_ba < 1e-10:
                    continue
                bias_ba = np.dot(mu_bt - mu[b], direction_ba / norm_ba)
                asym = bias_ab - bias_ba
                if asym > max_asym:
                    max_asym = asym

            if max_asym > -np.inf:
                asym_matrix[a, b] = max_asym

    return asym_matrix, cytokine_names


def main():
    args    = _parse_args()
    run_dir = Path(args.run_dir)
    device  = torch.device(args.device)

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        _log(f"ERROR: No checkpoints at {ckpt_dir}")
        sys.exit(1)

    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    _log(f"Found {len(ckpt_files)} checkpoints.")

    with open(HVG_PATH) as f:
        gene_names = json.load(f)

    label_enc = _load_label_encoder(run_dir)

    dataset = PseudoTubeDataset(
        str(run_dir / "manifest_train.json"),
        label_enc, gene_names=gene_names, preload=False,
    )
    _log(f"Training dataset: {len(dataset)} tubes")

    def epoch_of(p): return int(p.stem.replace("epoch_", ""))
    epochs       = [epoch_of(f) for f in ckpt_files]
    asym_traj    = []
    cytokine_names = None

    for ckpt_path, epoch in zip(ckpt_files, epochs):
        _log(f"\n--- Epoch {epoch} ({ckpt_path.name}) ---")
        model = _load_model(run_dir, ckpt_path, label_enc, gene_names, device)

        _log("  Building cache...")
        cache = build_cache(model, dataset, device)

        _log("  Computing PBS means...")
        _, pbs_ct_means = precompute_transform_means(cache, label_enc)
        pbs_fn = make_pbs_relative_fn(pbs_ct_means)

        _log("  Computing asymmetry matrix (no permutations)...")
        asym_mat, cytokine_names = compute_fast_asymmetry(cache, label_enc, pbs_fn)
        asym_traj.append(asym_mat)
        _log(f"  Done. Matrix shape: {asym_mat.shape}")

        del model, cache

    asym_traj_np = np.stack(asym_traj, axis=0)  # (n_epochs, K, K)

    out = {
        "epochs":         epochs,
        "asymmetry_traj": asym_traj_np,
        "cytokine_names": cytokine_names,
    }

    out_path = run_dir / "geo_trajectory.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    _log(f"\nSaved: {out_path}  shape={asym_traj_np.shape}")


if __name__ == "__main__":
    main()
