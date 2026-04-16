"""
Experiment 3: Train AuxDecoder and re-run latent geometry experiments.

Triggered after Experiment 0 gate failure (encoder embeds cell-type space).

Trains a small post-hoc MLP on frozen encoder+MIL output:
    h_i ∈ R^128  →  g_i ∈ R^64  →  cytokine_logits ∈ R^91

Supervised by sharpened bag-level softmax (temperature τ):
    p_bag_τ(C | tube) = softmax(y_hat / τ)

Loss (uniform KL — attention-weighted was empirically disqualified, see CLAUDE.md §20.8):
    L = (1/N) sum_i KL( softmax(y_hat/τ) || softmax(decoder(h_i)) )

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
    <run_dir>/latent_geometry_decoder_tau{τ}.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List

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
             "manifest_train.json, and optionally manifest_val.json.",
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


def _load_mil_model(run_dir: Path, label_enc: CytokineLabel, gene_names: list, device):
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
# Decoder training
# ---------------------------------------------------------------------------

def _train_decoder(
    model: nn.Module,
    dataset: PseudoTubeDataset,
    label_encoder: CytokineLabel,
    tau: float,
    epochs: int,
    lr: float,
    device: torch.device,
) -> AuxDecoder:
    """
    Train AuxDecoder for one temperature τ.

    For each tube:
        1. Forward the frozen MIL model → y_hat (K,), H (N, 128)
        2. Compute supervision: p_bag = softmax(y_hat / τ)  [K,]
        3. Compute decoder logits: decoder(H)  [N, K]
        4. Per-cell KL: KL( p_bag || softmax(decoder(h_i)) )
        5. Loss = mean over all cells in tube

    Loss (uniform KL — attention-weighted disqualified by empirical check):
        L = (1/N) sum_i KL( p_bag_τ || softmax(decoder(h_i)) )

    The KL convention: KL(P || Q) = sum_k P_k * log(P_k / Q_k)
    Using F.kl_div(Q.log(), P, reduction='batchmean') = KL(P || Q) per PyTorch docs.
    """
    decoder = AuxDecoder(
        input_dim=128, embed_dim=EMBED_DIM, n_classes=label_encoder.n_classes()
    )
    decoder.to(device)
    decoder.train()

    optimizer = Adam(decoder.parameters(), lr=lr)

    _log(f"  Training decoder with τ={tau} for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_losses: List[float] = []

        for ds_idx in range(len(dataset)):
            X, _label, _donor, _cytokine = dataset[ds_idx]
            X = X.to(device)

            # Frozen MIL forward.
            with torch.no_grad():
                outputs = model(X)
                if len(outputs) == 3:
                    y_hat, _a, H = outputs
                elif len(outputs) == 4:
                    y_hat, _a_sa, _a_ca, H = outputs
                else:
                    raise ValueError(
                        f"Unexpected model output length: {len(outputs)}"
                    )
            # y_hat: (K,) or (1, K) — normalize to 1-D
            if y_hat.dim() == 2:
                y_hat = y_hat.squeeze(0)

            # Supervision: sharpened bag-level softmax.
            p_bag = F.softmax(y_hat / tau, dim=0)  # (K,)

            # Decoder forward.
            logits = decoder(H)                            # (N, K)
            log_q = F.log_softmax(logits, dim=1)           # (N, K)

            # Broadcast p_bag to (N, K) and compute KL(p_bag || q_i) per cell.
            p_bag_exp = p_bag.unsqueeze(0).expand_as(log_q)  # (N, K)

            # F.kl_div(input=log_q, target=p_bag_exp, reduction='none')
            # = sum_k p_k * (log p_k - log q_k)   per row (standard KL)
            # We want KL(p_bag || q_i), sum over classes.
            kl_per_cell = F.kl_div(
                log_q, p_bag_exp, reduction="none"
            ).sum(dim=1)                                 # (N,)
            loss = kl_per_cell.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            _log(
                f"    epoch {epoch:3d}/{epochs}  "
                f"mean_loss={np.mean(epoch_losses):.6f}"
            )

    decoder.eval()
    return decoder


# ---------------------------------------------------------------------------
# Cell-type obs loader
# ---------------------------------------------------------------------------

def _build_cell_type_obs(dataset: PseudoTubeDataset) -> dict:
    """
    Load cell_type annotations for each tube from its h5ad file.

    Returns {tube_index -> list of cell_type strings, length N}.
    Tubes without obs["cell_type"] are omitted.
    """
    _log("Loading cell-type annotations from h5ad files...")
    cell_type_obs = {}
    for ds_idx in range(len(dataset)):
        entry = dataset.entries[ds_idx]
        adata = sc.read_h5ad(entry["path"])
        if "cell_type" not in adata.obs.columns:
            continue
        cell_type_obs[ds_idx] = list(adata.obs["cell_type"].values)
    _log(f"  Loaded cell types for {len(cell_type_obs)}/{len(dataset)} tubes.")
    return cell_type_obs


# ---------------------------------------------------------------------------
# Geometry experiments (Exp 0, 1, 2)
# ---------------------------------------------------------------------------

def _run_geometry_experiments(
    model: nn.Module,
    decoder: AuxDecoder,
    train_dataset: PseudoTubeDataset,
    label_encoder: CytokineLabel,
    cell_type_obs: dict,
    n_permutations: int,
    device_str: str,
    tau: float,
) -> dict:
    """
    Run Experiments 0, 1, 2 using g_i = decoder.embed(h_i) as the embedding space.

    Returns dict with keys: 'tau', 'exp0', 'exp1', 'exp2'.
    """
    _log(f"  Running geometry experiments (τ={tau})...")

    # Experiment 0: cytokine centroids and alignment scores.
    _log("    Exp 0: computing centroids...")
    centroid_result = compute_cytokine_centroids(
        model, train_dataset, label_encoder, device=device_str, decoder=decoder
    )
    centroids = centroid_result["centroids"]

    _log("    Exp 0: computing alignment scores (permutation test)...")
    alignment_result = compute_alignment_scores(
        model,
        train_dataset,
        label_encoder,
        centroids=centroids,
        n_permutations=n_permutations,
        device=device_str,
        decoder=decoder,
    )

    # Experiment 1: directional bias.
    _log("    Exp 1: computing directional bias (BH-FDR, may take a while)...")
    bias_result = compute_directional_bias(
        model,
        train_dataset,
        label_encoder,
        centroids=centroids,
        cell_type_obs=cell_type_obs,
        n_permutations=n_permutations,
        device=device_str,
        decoder=decoder,
    )

    # Experiment 2: asymmetry matrix.
    _log("    Exp 2: computing asymmetry matrix...")
    asym_result = compute_asymmetry_matrix(bias_result["bias"], label_encoder)

    return {
        "tau": tau,
        "exp0": {
            "centroids": centroid_result,
            "alignment": alignment_result,
        },
        "exp1": bias_result,
        "exp2": asym_result,
    }


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _report_exp0(alignment_result: dict, tau: float) -> None:
    """Print Exp 0 alignment score summary for this τ."""
    scores = alignment_result["alignment_scores"]
    p_values = alignment_result["p_values"]
    null_mean = alignment_result["null_mean"]
    null_std = alignment_result["null_std"]

    n_sig = sum(1 for p in p_values.values() if p < 0.05)
    n_total = len(scores)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    _log()
    _log(f"{'=' * 60}")
    _log(f"EXP 0 ALIGNMENT — τ={tau}")
    _log(f"  Null: mean={null_mean:.5f}  std={null_std:.5f}")
    _log(f"  Significant (p < 0.05): {n_sig}/{n_total}")
    _log(f"  {'Cytokine':<25} {'Score':>8} {'p':>8}")
    for cyt, score in sorted_scores[:20]:
        p = p_values.get(cyt, 1.0)
        marker = " *" if p < 0.05 else ""
        _log(f"  {cyt:<25} {score:8.5f} {p:8.4f}{marker}")
    if len(sorted_scores) > 20:
        _log(f"  ... ({len(sorted_scores) - 20} more)")
    _log(f"{'=' * 60}")

    # Known controls.
    for pair_a, pair_b in [("IL-12", "IFN-gamma"), ("IL-6", "IL-10")]:
        for cyt in [pair_a, pair_b]:
            if cyt in scores:
                _log(
                    f"  Control — {cyt}: score={scores[cyt]:.5f}  "
                    f"p={p_values.get(cyt, 1.0):.4f}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    device_str = args.device

    _log("=" * 60)
    _log("Experiment 3 — AuxDecoder training + latent geometry")
    _log(f"  run_dir: {run_dir}")
    _log(f"  tau values: {args.tau_values}")
    _log(f"  epochs per τ: {args.epochs}")
    _log(f"  device: {args.device}")
    _log("=" * 60)

    # Load shared resources.
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

    cell_type_obs = _build_cell_type_obs(train_dataset)

    # Train one decoder per τ, then run geometry experiments.
    for tau in args.tau_values:
        _log()
        _log(f"{'─' * 60}")
        _log(f"τ = {tau}")
        _log(f"{'─' * 60}")

        decoder = _train_decoder(
            model=model,
            dataset=train_dataset,
            label_encoder=label_encoder,
            tau=tau,
            epochs=args.epochs,
            lr=LR,
            device=device,
        )

        # Save decoder checkpoint.
        ckpt_path = run_dir / f"aux_decoder_tau{tau}.pt"
        torch.save(decoder.state_dict(), ckpt_path)
        _log(f"  Saved decoder checkpoint: {ckpt_path}")

        # Run geometry experiments with this decoder.
        results = _run_geometry_experiments(
            model=model,
            decoder=decoder,
            train_dataset=train_dataset,
            label_encoder=label_encoder,
            cell_type_obs=cell_type_obs,
            n_permutations=args.n_permutations,
            device_str=device_str,
            tau=tau,
        )

        _report_exp0(results["exp0"]["alignment"], tau)

        # Save results.
        tau_str = str(tau).replace(".", "")
        out_path = run_dir / f"latent_geometry_decoder_tau{tau}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        _log(f"  Saved geometry results: {out_path}")

    _log()
    _log("Done.")


if __name__ == "__main__":
    main()
