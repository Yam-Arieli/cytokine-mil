"""
Geometry engineering fixes: PBS-relative centroids and h_residual.

Two post-hoc transformations applied to frozen encoder embeddings h_i,
designed to remove confounds before computing the latent geometry signal.

Fix 1 — PBS-relative:
    For each cell i: h_i^pbs = h_i - μ_{PBS, cell_type(i)}
    Subtracts the PBS (resting-state) centroid of the *same cell type*.
    What remains is the stimulus-specific deviation from rest.
    Analogous to log2FC(cytokine / PBS) at the embedding level.

Fix 2 — h_residual:
    For each cell i: h_i^resid = h_i - μ_{global, cell_type(i)}
    Subtracts the grand mean across ALL cytokine conditions for cell_type(i).
    Removes cell-type structure shared across all conditions.
    Captures the condition-specific residual.

Both fixes are 100% deterministic given the MIL checkpoint — no decoder,
no random seed.  Run one per MIL seed (3 total: 42, 123, 7).

Results are saved to:
    <run_dir>/experiment_geo_pbs_rel/latent_geometry.pkl
    <run_dir>/experiment_geo_hresid/latent_geometry.pkl

Usage:
    python scripts/run_experiment_geo.py --run_dir <run_dir>
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.latent_geometry import (
    build_tube_data_from_cache,
    compute_asymmetry_matrix,
    compute_cytokine_centroids_from_cache,
    compute_directional_bias_from_arrays,
)
from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model
from cytokine_mil.training.cache import build_cache

HVG_PATH = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir", type=str, required=True,
        help="Run directory containing model_stage2.pt, label_encoder.json, "
             "manifest_train.json.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--n_permutations", type=int, default=1000,
        help="Permutations for bias null distribution (default: 1000).",
    )
    p.add_argument(
        "--hvg_path", type=str, default=None,
        help="Path to hvg_list.json. Defaults to the Oesinghaus HVG list.",
    )
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Helpers (shared with run_experiment3.py)
# ---------------------------------------------------------------------------

def _load_label_encoder(run_dir: Path) -> CytokineLabel:
    with open(run_dir / "label_encoder.json") as f:
        data = json.load(f)
    cytokines_list = data["cytokines"]
    le = CytokineLabel()
    le._label_to_idx = {c: i for i, c in enumerate(cytokines_list)}
    le._idx_to_label = {i: c for i, c in enumerate(cytokines_list)}
    return le


def _load_mil_model(
    run_dir: Path,
    label_enc: CytokineLabel,
    gene_names: list,
    device: torch.device,
):
    state_dict = torch.load(
        run_dir / "model_stage2.pt", map_location="cpu", weights_only=False
    )
    n_cell_types = state_dict["encoder.cell_type_head.weight"].shape[0]
    encoder = build_encoder(
        n_input_genes=len(gene_names), n_cell_types=n_cell_types, embed_dim=128
    )
    model = build_mil_model(
        encoder,
        embed_dim=128,
        attention_hidden_dim=64,
        n_classes=label_enc.n_classes(),
        encoder_frozen=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Transform precomputation
# ---------------------------------------------------------------------------

def precompute_transform_means(cache: list, label_encoder) -> tuple:
    """
    Compute per-cell-type embedding means needed for both transforms.

    Returns:
        global_ct_means: {cell_type -> mean h_i across ALL conditions} (h_residual)
        pbs_ct_means:    {cell_type -> mean h_i in PBS tubes only}     (PBS-relative)
    """
    embed_dim = cache[0]["H"].shape[1]

    ct_sum   = defaultdict(lambda: np.zeros(embed_dim, dtype=np.float64))
    ct_count = defaultdict(float)
    pbs_sum   = defaultdict(lambda: np.zeros(embed_dim, dtype=np.float64))
    pbs_count = defaultdict(float)

    for entry in cache:
        H_np      = entry["H"].numpy().astype(np.float64)   # (N, D)
        ct_labels = entry["cell_types"]                       # list[str], len N
        cytokine  = label_encoder._idx_to_label[entry["label"]]

        for i, ct in enumerate(ct_labels):
            ct_sum[ct]   += H_np[i]
            ct_count[ct] += 1.0
            if cytokine == "PBS":
                pbs_sum[ct]   += H_np[i]
                pbs_count[ct] += 1.0

    global_ct_means = {
        ct: ct_sum[ct] / ct_count[ct]
        for ct in ct_sum
    }
    pbs_ct_means = {
        ct: pbs_sum[ct] / pbs_count[ct]
        for ct in pbs_sum
        if pbs_count[ct] > 0
    }

    return global_ct_means, pbs_ct_means


def make_pbs_relative_fn(pbs_ct_means: dict):
    """
    Returns a transform: h_i^pbs = h_i - μ_{PBS, cell_type(i)}.

    Cells whose cell type has no PBS representation are left unchanged.
    """
    def fn(H_np: np.ndarray, ct_labels: np.ndarray) -> np.ndarray:
        result = H_np.copy()
        for i, ct in enumerate(ct_labels):
            if ct in pbs_ct_means:
                result[i] -= pbs_ct_means[ct]
        return result
    fn.__name__ = "pbs_relative"
    return fn


def make_hresidual_fn(global_ct_means: dict):
    """
    Returns a transform: h_i^resid = h_i - μ_{global, cell_type(i)}.

    Removes the cell-type component shared across all cytokine conditions.
    """
    def fn(H_np: np.ndarray, ct_labels: np.ndarray) -> np.ndarray:
        result = H_np.copy()
        for i, ct in enumerate(ct_labels):
            if ct in global_ct_means:
                result[i] -= global_ct_means[ct]
        return result
    fn.__name__ = "h_residual"
    return fn


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------

def run_one_experiment(
    cache: list,
    label_encoder,
    h_transform_fn,
    transform_label: str,
    n_permutations: int,
    out_dir: Path,
):
    """
    Run full geometry pipeline (centroids → biases → asymmetry) for one transform.
    Saves results to out_dir/latent_geometry.pkl.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"\n  [{transform_label}] Computing centroids...")
    centroids_result = compute_cytokine_centroids_from_cache(
        cache, label_encoder, h_transform_fn=h_transform_fn
    )
    centroids = centroids_result["centroids"]
    n_cyts = len(centroids)
    first_mu = next(iter(centroids.values()))
    _log(f"  [{transform_label}] {n_cyts} centroids, dim={first_mu.shape[0]}")

    _log(f"  [{transform_label}] Building tube data...")
    tube_data = build_tube_data_from_cache(
        cache, label_encoder, centroids, h_transform_fn=h_transform_fn
    )
    _log(f"  [{transform_label}] {len(tube_data)} tubes ready")

    _log(f"  [{transform_label}] Computing directional bias "
         f"({n_permutations} permutations)...")
    bias_result = compute_directional_bias_from_arrays(
        tube_data=tube_data,
        centroids=centroids,
        n_permutations=n_permutations,
        transform_label=transform_label,
    )
    _log(f"  [{transform_label}] {len(bias_result['bias'])} (A,B,T) triples computed")

    _log(f"  [{transform_label}] Computing asymmetry matrix...")
    asym_result = compute_asymmetry_matrix(bias_result["bias"], label_encoder)

    results = {
        "centroids": centroids_result,
        "bias": bias_result,
        "asymmetry": asym_result,
        "config": {
            "transform": transform_label,
            "n_permutations": n_permutations,
        },
    }

    out_path = out_dir / "latent_geometry.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    _log(f"  [{transform_label}] Saved: {out_path}")

    # Quick peek at top pairs
    asym_matrix = asym_result["asymmetry_matrix"]
    cyt_names   = asym_result["cytokine_names"]
    pbs_idx     = next((i for i, n in enumerate(cyt_names) if n == "PBS"), None)
    K = asym_matrix.shape[0]
    pairs = []
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            if pbs_idx is not None and (a == pbs_idx or b == pbs_idx):
                continue
            pairs.append((asym_matrix[a, b], cyt_names[a], cyt_names[b]))
    pairs.sort(reverse=True)
    _log(f"\n  [{transform_label}] Top-10 asymmetry pairs (PBS excluded):")
    for score, a, b in pairs[:10]:
        _log(f"    {a:<22} → {b:<22}  asym={score:.4f}")

    # Pairs of interest
    poi = [
        ("IL-12",    "IFN-gamma",  "positive control"),
        ("IFN-gamma","IL-12",      "reverse"),
        ("IL-6",     "IL-10",      "negative control"),
        ("IL-10",    "IL-6",       "negative control reverse"),
    ]
    n_off = K * (K - 1)
    if pbs_idx is not None:
        n_off -= 2 * (K - 1)  # approximate
    name_to_idx = {n: i for i, n in enumerate(cyt_names)}
    all_vals = [(asym_matrix[a, b], a, b)
                for a in range(K) for b in range(K) if a != b
                and (pbs_idx is None or (a != pbs_idx and b != pbs_idx))]
    all_vals.sort(reverse=True)
    flat_vals = [v for v, _, _ in all_vals]
    _log(f"\n  [{transform_label}] Pairs of interest:")
    for src, tgt, note in poi:
        si = name_to_idx.get(src)
        ti = name_to_idx.get(tgt)
        if si is None or ti is None:
            _log(f"    {src} → {tgt}  NOT FOUND")
            continue
        val = float(asym_matrix[si, ti])
        rank = sum(1 for v in flat_vals if v > val) + 1
        _log(f"    {src:<20} → {tgt:<20}  asym={val:.4f}  "
             f"rank={rank}/{len(flat_vals)}  [{note}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    device  = torch.device(args.device)

    _log("=" * 62)
    _log("Geometry Engineering Fixes: PBS-relative + h_residual")
    _log(f"  run_dir        : {run_dir}")
    _log(f"  device         : {args.device}")
    _log(f"  n_permutations : {args.n_permutations}")
    _log("=" * 62)

    hvg_path = args.hvg_path if args.hvg_path else HVG_PATH
    with open(hvg_path) as f:
        gene_names = json.load(f)

    label_encoder = _load_label_encoder(run_dir)
    mil_model     = _load_mil_model(run_dir, label_encoder, gene_names, device)

    dataset = PseudoTubeDataset(
        str(run_dir / "manifest_train.json"),
        label_encoder,
        gene_names=gene_names,
        preload=False,
    )
    _log(f"Training dataset: {len(dataset)} tubes")

    # ---- Single forward pass — builds cache with H, y_hat, cell_types ----
    _log("\nStep 1: Building h_i cache (single forward pass)...")
    cache = build_cache(mil_model, dataset, device)
    _log(f"  Cached {len(cache)} tubes.")

    # ---- Precompute subtraction means from cache ----
    _log("\nStep 2: Precomputing cell-type means (global + PBS)...")
    global_ct_means, pbs_ct_means = precompute_transform_means(cache, label_encoder)
    _log(f"  Cell types found (global): {sorted(global_ct_means.keys())[:5]} ...")
    _log(f"  Cell types in PBS: {len(pbs_ct_means)}/{len(global_ct_means)}")

    # ---- Fix 1: PBS-relative ----
    _log("\nStep 3: Running PBS-relative experiment...")
    pbs_fn = make_pbs_relative_fn(pbs_ct_means)
    run_one_experiment(
        cache=cache,
        label_encoder=label_encoder,
        h_transform_fn=pbs_fn,
        transform_label="pbs_relative",
        n_permutations=args.n_permutations,
        out_dir=run_dir / "experiment_geo_pbs_rel",
    )

    # ---- Fix 2: h_residual ----
    _log("\nStep 4: Running h_residual experiment...")
    hresid_fn = make_hresidual_fn(global_ct_means)
    run_one_experiment(
        cache=cache,
        label_encoder=label_encoder,
        h_transform_fn=hresid_fn,
        transform_label="h_residual",
        n_permutations=args.n_permutations,
        out_dir=run_dir / "experiment_geo_hresid",
    )

    _log("\nDone.")


if __name__ == "__main__":
    main()
