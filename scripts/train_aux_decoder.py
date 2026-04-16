"""
Experiment 3: Train AuxDecoder and re-run latent geometry experiments.

Triggered after Experiment 0 gate failure (encoder embeds cell-type space).

Trains a small post-hoc MLP on frozen encoder+MIL output:
    h_i ∈ R^128  →  g_i ∈ R^64  →  cytokine_logits ∈ R^91

Supervised by sharpened bag-level softmax (temperature τ):
    p_bag_τ(C | tube) = softmax(y_hat / τ)

Loss (uniform KL — attention-weighted was empirically disqualified, see CLAUDE.md §20.8):
    L = (1/N) sum_i KL( softmax(y_hat/τ) || softmax(decoder(h_i)) )

Key optimization: ALL tube embeddings H and bag logits y_hat are precomputed
in a single forward pass through the frozen model and cached in RAM before any
training begins. Decoder training then iterates purely over in-memory tensors —
zero h5ad file I/O during the training loop. This reduces epoch time from
~3h (9,100 NFS reads/epoch) to seconds.

Trains with τ ∈ {0.3, 0.5, 1.0} and saves one checkpoint per τ.
Then re-runs Experiments 0, 1, 2 on g_i and saves combined results.

Usage:
    python scripts/train_aux_decoder.py \\
        --run_dir results/oesinghaus_full/run_20260412_161758_seed42 \\
        --device cpu

Saves:
    <run_dir>/aux_decoder_tau0.3.pt
    <run_dir>/aux_decoder_tau0.5.pt
    <run_dir>/aux_decoder_tau1.0.pt
    <run_dir>/latent_geometry_decoder_tau0.3.pkl
    <run_dir>/latent_geometry_decoder_tau0.5.pkl
    <run_dir>/latent_geometry_decoder_tau1.0.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.latent_geometry import (
    compute_alignment_scores,
    compute_asymmetry_matrix,
    compute_cytokine_centroids,
    compute_directional_bias,
)
from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model
from cytokine_mil.models.aux_decoder import AuxDecoder

HVG_PATH = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
)
TAU_VALUES = [0.3, 0.5, 1.0]
EMBED_DIM = 64
EPOCHS = 50
LR = 1e-3
N_PERMUTATIONS = 1000


# ---------------------------------------------------------------------------
# Argument parsing
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
        "--tau_values", type=float, nargs="+", default=TAU_VALUES,
        help="Temperature values for bag-level softmax sharpening.",
    )
    p.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="Training epochs per τ value.",
    )
    p.add_argument(
        "--n_permutations", type=int, default=N_PERMUTATIONS,
        help="Permutations for null distribution in geometry experiments.",
    )
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_label_encoder(run_dir: Path) -> CytokineLabel:
    with open(run_dir / "label_encoder.json") as f:
        data = json.load(f)
    cytokines_list = data["cytokines"]
    le = CytokineLabel()
    le._label_to_idx = {cyt: i for i, cyt in enumerate(cytokines_list)}
    le._idx_to_label = {i: cyt for i, cyt in enumerate(cytokines_list)}
    return le


def _load_mil_model(
    run_dir: Path, label_enc: CytokineLabel, gene_names: list, device: torch.device
) -> nn.Module:
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
# Precompute cache — single pass through all tubes
# ---------------------------------------------------------------------------

def _precompute_cache(
    model: nn.Module,
    dataset: PseudoTubeDataset,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[str]]:
    """
    Single forward pass through all tubes. Cache H and y_hat in RAM.

    Returns:
        H_list:        list of (N_i, 128) CPU tensors, one per tube
        yhat_list:     list of (K,) CPU tensors, one per tube
        cytokine_list: list of cytokine name strings, one per tube
        donor_list:    list of donor name strings, one per tube
    """
    _log("Precomputing H and y_hat for all tubes (single forward pass)...")
    H_list: List[torch.Tensor] = []
    yhat_list: List[torch.Tensor] = []
    cytokine_list: List[str] = []
    donor_list: List[str] = []

    model.eval()
    with torch.no_grad():
        for ds_idx in range(len(dataset)):
            X, _label, donor, cytokine = dataset[ds_idx]
            X = X.to(device)
            outputs = model(X)
            if len(outputs) == 3:
                y_hat, _a, H = outputs
            elif len(outputs) == 4:
                y_hat, _a_sa, _a_ca, H = outputs
            else:
                raise ValueError(f"Unexpected model output length: {len(outputs)}")

            if y_hat.dim() == 2:
                y_hat = y_hat.squeeze(0)

            H_list.append(H.cpu())
            yhat_list.append(y_hat.cpu())
            cytokine_list.append(cytokine)
            donor_list.append(donor)

            if (ds_idx + 1) % 500 == 0:
                _log(f"  Precomputed {ds_idx + 1}/{len(dataset)} tubes...")

    _log(f"  Done. Cached {len(H_list)} tubes in RAM.")
    return H_list, yhat_list, cytokine_list, donor_list


# ---------------------------------------------------------------------------
# Cell-type obs loader
# ---------------------------------------------------------------------------

def _build_cell_type_obs(dataset: PseudoTubeDataset) -> Dict[int, List[str]]:
    """
    Load cell_type annotations for each tube from its h5ad file.
    Returns {tube_index -> list of cell_type strings, length N}.
    Tubes without obs["cell_type"] are omitted.
    """
    _log("Loading cell-type annotations from h5ad files...")
    cell_type_obs: Dict[int, List[str]] = {}
    for ds_idx in range(len(dataset)):
        entry = dataset.entries[ds_idx]
        adata = sc.read_h5ad(entry["path"])
        if "cell_type" not in adata.obs.columns:
            continue
        cell_type_obs[ds_idx] = list(adata.obs["cell_type"].values)
    _log(f"  Loaded cell types for {len(cell_type_obs)}/{len(dataset)} tubes.")
    return cell_type_obs


# ---------------------------------------------------------------------------
# Decoder training — purely in-memory
# ---------------------------------------------------------------------------

def _train_decoder(
    H_list: List[torch.Tensor],
    yhat_list: List[torch.Tensor],
    n_classes: int,
    tau: float,
    epochs: int,
    lr: float,
    device: torch.device,
) -> AuxDecoder:
    """
    Train AuxDecoder using precomputed in-memory H and y_hat.

    For each tube:
        p_bag = softmax(y_hat / τ)                         # (K,) supervision
        logits = decoder(H)                                # (N, K)
        loss = mean_i KL( p_bag || softmax(logits_i) )    # uniform KL

    No h5ad I/O — everything is in RAM.

    Loss convention: F.kl_div(log_Q, P) = KL(P || Q) = sum_k P_k * log(P_k/Q_k)
    """
    decoder = AuxDecoder(input_dim=128, embed_dim=EMBED_DIM, n_classes=n_classes)
    decoder.to(device)
    decoder.train()
    optimizer = Adam(decoder.parameters(), lr=lr)

    n_tubes = len(H_list)
    _log(f"  Training decoder: τ={tau}, {epochs} epochs, {n_tubes} tubes in RAM...")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for H_cpu, yhat_cpu in zip(H_list, yhat_list):
            H = H_cpu.to(device)
            y_hat = yhat_cpu.to(device)

            p_bag = F.softmax(y_hat / tau, dim=0)          # (K,)
            logits = decoder(H)                            # (N, K)
            log_q = F.log_softmax(logits, dim=1)           # (N, K)

            # Broadcast p_bag to (N, K): KL(p_bag || q_i) per cell, then mean
            p_bag_exp = p_bag.unsqueeze(0).expand_as(log_q)
            loss = F.kl_div(log_q, p_bag_exp, reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        mean_loss = epoch_loss / n_tubes
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            _log(f"    epoch {epoch:3d}/{epochs}  mean_loss={mean_loss:.6f}")

    decoder.eval()
    return decoder


# ---------------------------------------------------------------------------
# Geometry experiments via CachedDataset wrapper
# ---------------------------------------------------------------------------

class _CachedDataset:
    """
    Minimal duck-type wrapper so compute_* functions from latent_geometry.py
    can be reused with precomputed H and y_hat instead of reading from disk.

    The geometry functions call dataset[i] and expect (X, label, donor, cytokine).
    We return X=H (already encoder output, not raw gene expression) and patch
    _extract_embeddings by passing an identity model wrapper.
    """
    pass


def _compute_centroids_from_cache(
    H_list: List[torch.Tensor],
    cytokine_list: List[str],
    decoder: AuxDecoder,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Compute cytokine centroids μ_C = mean g_i over all C-tubes.
    g_i = decoder.embed(h_i).
    """
    decoder.eval()
    embedding_sums: Dict[str, np.ndarray] = {}
    cell_counts: Dict[str, int] = {}
    embed_dim: int = EMBED_DIM

    with torch.no_grad():
        for H_cpu, cytokine in zip(H_list, cytokine_list):
            H = H_cpu.to(device)
            g = decoder.embed(H).cpu().numpy().astype(np.float64)  # (N, 64)
            if cytokine not in embedding_sums:
                embedding_sums[cytokine] = np.zeros(embed_dim, dtype=np.float64)
                cell_counts[cytokine] = 0
            embedding_sums[cytokine] += g.sum(axis=0)
            cell_counts[cytokine] += g.shape[0]

    return {
        cyt: embedding_sums[cyt] / cell_counts[cyt]
        for cyt in embedding_sums
        if cell_counts[cyt] > 0
    }


def _compute_alignment_from_cache(
    H_list: List[torch.Tensor],
    cytokine_list: List[str],
    centroids: Dict[str, np.ndarray],
    decoder: AuxDecoder,
    device: torch.device,
    n_permutations: int = 1000,
) -> dict:
    """
    Cytokine alignment scores on g_i space, with permutation null.
    """
    from cytokine_mil.analysis.latent_geometry import _bh_correction
    import torch.nn.functional as F_

    decoder.eval()
    centroid_names = list(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in centroid_names]).astype(np.float32)
    centroid_tensor = torch.tensor(centroid_matrix, device=device)

    # Per-tube alignment scores
    tube_scores: List[Tuple[str, float]] = []

    with torch.no_grad():
        for H_cpu, cytokine in zip(H_list, cytokine_list):
            if cytokine not in centroids:
                continue
            g = decoder.embed(H_cpu.to(device))          # (N, 64)
            # L2 distances to all centroids: (N, C)
            g_sq = (g ** 2).sum(dim=1, keepdim=True)
            c_sq = (centroid_tensor ** 2).sum(dim=1, keepdim=True).T
            cross = g @ centroid_tensor.T
            dists = (g_sq + c_sq - 2 * cross).clamp(min=0).sqrt()
            affinities = F_.softmax(-dists, dim=1)       # (N, C)
            cyt_idx = centroid_names.index(cytokine)
            score = float(affinities[:, cyt_idx].mean().item())
            tube_scores.append((cytokine, score))

    # Mean per cytokine
    from collections import defaultdict
    cyt_tube: Dict[str, List[float]] = defaultdict(list)
    for cyt, s in tube_scores:
        cyt_tube[cyt].append(s)
    alignment_scores = {cyt: float(np.mean(vals)) for cyt, vals in cyt_tube.items()}

    # Permutation null
    rng = np.random.default_rng(0)
    scores_arr = np.array([s for _, s in tube_scores])
    null_scores = [float(rng.permutation(scores_arr).mean()) for _ in range(n_permutations)]
    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores))

    p_values = {
        cyt: float(np.mean(np.array(null_scores) >= score))
        for cyt, score in alignment_scores.items()
    }

    return {
        "alignment_scores": alignment_scores,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_values": p_values,
        "metric_description": (
            "mean over cells in A-tubes of softmax(-||g_i - μ_C||_2 for all C)[A], "
            "g_i = decoder.embed(h_i), training donors only"
        ),
    }


def _compute_bias_from_cache(
    H_list: List[torch.Tensor],
    cytokine_list: List[str],
    cell_type_obs: Dict[int, List[str]],
    centroids: Dict[str, np.ndarray],
    decoder: AuxDecoder,
    device: torch.device,
    n_permutations: int = 1000,
) -> dict:
    """
    Directional bias and asymmetry on g_i space, with BH-FDR null.
    """
    from cytokine_mil.analysis.latent_geometry import (
        _compute_all_biases, _compute_bias_null, _z_to_bh_qvalues
    )

    decoder.eval()
    tube_data: List[Tuple[str, np.ndarray, np.ndarray]] = []

    with torch.no_grad():
        for ds_idx, (H_cpu, cytokine) in enumerate(zip(H_list, cytokine_list)):
            if cytokine not in centroids:
                continue
            ct_labels = cell_type_obs.get(ds_idx)
            if ct_labels is None:
                continue
            g = decoder.embed(H_cpu.to(device)).cpu().numpy().astype(np.float64)
            tube_data.append((cytokine, g, np.array(ct_labels)))

    bias_obs = _compute_all_biases(tube_data, centroids)
    bias_null = _compute_bias_null(tube_data, centroids, n_permutations=n_permutations)

    z_scores = {}
    for key, obs in bias_obs.items():
        null_vals = bias_null.get(key, [])
        if len(null_vals) < 2:
            z_scores[key] = 0.0
        else:
            null_arr = np.array(null_vals)
            nm, ns = float(null_arr.mean()), float(null_arr.std())
            z_scores[key] = (obs - nm) / ns if ns > 0 else 0.0

    q_values = _z_to_bh_qvalues(z_scores)

    return {
        "bias": dict(bias_obs),
        "z_scores": dict(z_scores),
        "q_values": q_values,
        "metric_description": (
            "scalar projection of (μ_{A,T} - μ_A) [g_i = decoder.embed(h_i)] "
            "onto unit vector (μ_B - μ_A)/||μ_B - μ_A||_2"
        ),
    }


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _report_exp0(alignment_result: dict, tau: float, label_encoder: CytokineLabel) -> None:
    scores = alignment_result["alignment_scores"]
    p_values = alignment_result["p_values"]
    null_mean = alignment_result["null_mean"]
    null_std = alignment_result["null_std"]

    n_sig = sum(1 for p in p_values.values() if p < 0.05)
    n_total = len(scores)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    gate = n_sig >= n_total * 0.5

    _log()
    _log(f"{'=' * 62}")
    _log(f"EXP 0 ALIGNMENT — g_i space — τ={tau}")
    _log(f"  Null: mean={null_mean:.6f}  std={null_std:.6f}")
    _log(f"  Significant (p < 0.05): {n_sig}/{n_total}")
    _log(f"  Gate (≥50% sig): {'PASS ✓' if gate else 'FAIL ✗'}")
    _log(f"  {'Cytokine':<28} {'Score':>9} {'p':>7}")
    for cyt, score in sorted_scores[:20]:
        p = p_values.get(cyt, 1.0)
        marker = " *" if p < 0.05 else ""
        _log(f"  {cyt:<28} {score:9.6f} {p:7.4f}{marker}")
    if len(sorted_scores) > 20:
        _log(f"  ... ({len(sorted_scores) - 20} more cytokines)")

    _log("\n  Known controls:")
    for cyt in ["IL-12", "IFN-gamma", "IL-6", "IL-10", "IL-2", "TNF-alpha"]:
        if cyt in scores:
            p = p_values.get(cyt, 1.0)
            sig = " *" if p < 0.05 else ""
            _log(f"    {cyt:<25} {scores[cyt]:.6f}  p={p:.4f}{sig}")
    _log(f"{'=' * 62}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)

    _log("=" * 62)
    _log("Experiment 3 — AuxDecoder (in-memory cache)")
    _log(f"  run_dir: {run_dir}")
    _log(f"  tau: {args.tau_values}  epochs: {args.epochs}  device: {args.device}")
    _log("=" * 62)

    # Load shared resources
    with open(HVG_PATH) as f:
        gene_names = json.load(f)
    label_encoder = _load_label_encoder(run_dir)
    model = _load_mil_model(run_dir, label_encoder, gene_names, device)

    train_dataset = PseudoTubeDataset(
        str(run_dir / "manifest_train.json"),
        label_encoder,
        gene_names=gene_names,
        preload=False,
    )
    _log(f"Training dataset: {len(train_dataset)} tubes")

    # ----------------------------------------------------------------
    # Precompute ONCE — H and y_hat for all tubes in RAM
    # ----------------------------------------------------------------
    H_list, yhat_list, cytokine_list, donor_list = _precompute_cache(
        model, train_dataset, device
    )

    # ----------------------------------------------------------------
    # Cell-type annotations — loaded ONCE from h5ad
    # ----------------------------------------------------------------
    cell_type_obs = _build_cell_type_obs(train_dataset)

    # ----------------------------------------------------------------
    # Train one decoder per τ, run geometry experiments
    # ----------------------------------------------------------------
    for tau in args.tau_values:
        _log()
        _log(f"{'─' * 62}")
        _log(f"τ = {tau}")
        _log(f"{'─' * 62}")

        decoder = _train_decoder(
            H_list=H_list,
            yhat_list=yhat_list,
            n_classes=label_encoder.n_classes(),
            tau=tau,
            epochs=args.epochs,
            lr=LR,
            device=device,
        )

        # Save checkpoint
        ckpt_path = run_dir / f"aux_decoder_tau{tau}.pt"
        torch.save(decoder.state_dict(), ckpt_path)
        _log(f"  Saved: {ckpt_path}")

        # Exp 0: centroids + alignment
        _log("  Exp 0: centroids...")
        centroids = _compute_centroids_from_cache(H_list, cytokine_list, decoder, device)

        _log("  Exp 0: alignment scores (permutation test)...")
        alignment_result = _compute_alignment_from_cache(
            H_list, cytokine_list, centroids, decoder, device,
            n_permutations=args.n_permutations,
        )
        _report_exp0(alignment_result, tau, label_encoder)

        # Exp 1: directional bias (BH-FDR, 1000 permutations)
        _log("  Exp 1: directional bias (1000 permutations, may take ~10 min)...")
        bias_result = _compute_bias_from_cache(
            H_list, cytokine_list, cell_type_obs, centroids, decoder, device,
            n_permutations=args.n_permutations,
        )

        # Exp 2: asymmetry matrix
        _log("  Exp 2: asymmetry matrix...")
        asym_result = _compute_asymmetry_matrix_from_cache(bias_result["bias"], label_encoder)

        # Gate decision
        scores = alignment_result["alignment_scores"]
        p_values = alignment_result["p_values"]
        n_sig = sum(1 for p in p_values.values() if p < 0.05)
        gate_pass = n_sig >= len(scores) * 0.5

        results = {
            "tau": tau,
            "gate_pass": gate_pass,
            "exp0": {
                "centroids": {"centroids": centroids},
                "alignment": alignment_result,
            },
            "exp1": bias_result,
            "exp2": asym_result,
        }

        out_path = run_dir / f"latent_geometry_decoder_tau{tau}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        _log(f"  Saved: {out_path}")

    _log()
    _log("Done.")


def _compute_asymmetry_matrix_from_cache(bias: dict, label_encoder) -> dict:
    """Wrapper calling the public function from latent_geometry."""
    from cytokine_mil.analysis.latent_geometry import compute_asymmetry_matrix
    # compute_asymmetry_matrix takes the bias dict directly
    return compute_asymmetry_matrix(bias, label_encoder)


if __name__ == "__main__":
    main()
