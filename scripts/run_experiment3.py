"""
Experiment 3: AuxDecoder for cytokine geometry injection.

Triggered after Experiment 0 gate failure (encoder embeds cell-type space only,
not cytokine space). Trains a small post-hoc MLP on frozen encoder+MIL output
then re-runs latent geometry experiments in decoder embedding space.

Pipeline:
    1. Load trained MIL model checkpoint from <run_dir>/model_stage2.pt
    2. Build in-memory cache (single forward pass, no per-epoch h5ad reads)
    3. Train AuxDecoder with SGD+momentum and MSE loss
    4. Compute cytokine centroids in decoder embedding space (g_i = decoder.embed(h_i))
    5. Evaluate directional bias (Experiment 1) and asymmetry (Experiment 2)
    6. Save results to <run_dir>/experiment3/

Usage:
    python scripts/run_experiment3.py \\
        --run_dir results/oesinghaus_full/run_20260412_161758_seed42 \\
        --device cuda

Saves:
    <run_dir>/experiment3/aux_decoder.pt
    <run_dir>/experiment3/latent_geometry.pkl
    <run_dir>/experiment3/report.txt
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.latent_geometry import (
    compute_asymmetry_matrix,
    compute_cytokine_centroids,
    compute_directional_bias,
)
from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model
from cytokine_mil.models.aux_decoder import AuxDecoder
from cytokine_mil.training.cache import CachedTubeDataset, build_cache
from cytokine_mil.training.train_aux_decoder import train_aux_decoder

HVG_PATH = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
)
EMBED_DIM = 64
EPOCHS = 50
LR = 0.01
MOMENTUM = 0.9


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir", type=str, required=True,
        help="Run directory containing model_stage2.pt, label_encoder.json, "
             "manifest_train.json.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--hidden_dim", type=int, default=EMBED_DIM,
        help="AuxDecoder hidden/embedding dimension (default: 64).",
    )
    p.add_argument(
        "--n_permutations", type=int, default=1000,
        help="Permutations for bias null distribution.",
    )
    p.add_argument(
        "--min_confidence", type=float, default=0.5,
        help="Skip training tubes where max(ŷ) < threshold (default 0.5).",
    )
    p.add_argument(
        "--exp_name", type=str, default="experiment3_v2",
        help="Output subdirectory name under run_dir (default: experiment3_v2).",
    )
    p.add_argument(
        "--encoder_space", action="store_true", default=False,
        help="Approach F: compute geometry in frozen h_i (encoder) space. "
             "Decoder is trained normally but used only as a per-cell cytokine "
             "confidence weight, not as an embedding space. Eliminates rotation "
             "ambiguity by construction. (default: False = original g_i space)",
    )
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


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


def _report_alignment(centroids: dict, label_encoder: CytokineLabel) -> None:
    first_mu = next(iter(centroids["centroids"].values()))
    _log(f"\nCytokine centroids computed: dim={first_mu.shape[0]}, "
         f"n_cytokines={len(centroids['centroids'])}")
    _log(f"  {centroids['metric_description']}")


def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    out_dir = run_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("=" * 62)
    _log("Experiment 3 — AuxDecoder (in-memory cache pipeline)")
    _log(f"  run_dir : {run_dir}")
    _log(f"  out_dir : {out_dir}")
    _log(f"  device  : {args.device}")
    _log(f"  epochs  : {args.epochs}  lr={args.lr}  seed={args.seed}")
    _log(f"  hidden_dim: {args.hidden_dim}  min_conf={args.min_confidence}  exp_name={args.exp_name}")
    _log(f"  encoder_space: {args.encoder_space}  "
         f"({'h_i geometry, decoder as weights' if args.encoder_space else 'g_i geometry (original)'})")
    _log("=" * 62)

    with open(HVG_PATH) as f:
        gene_names = json.load(f)
    label_encoder = _load_label_encoder(run_dir)
    mil_model = _load_mil_model(run_dir, label_encoder, gene_names, device)

    dataset = PseudoTubeDataset(
        str(run_dir / "manifest_train.json"),
        label_encoder,
        gene_names=gene_names,
        preload=False,
    )
    _log(f"Training dataset: {len(dataset)} tubes")

    # ----------------------------------------------------------------
    # Step 1: Build in-memory cache (single forward pass)
    # ----------------------------------------------------------------
    _log("\nStep 1: Building cache...")
    cache = build_cache(mil_model, dataset, device)
    _log(f"  Cached {len(cache)} tubes in RAM.")

    cached_dataset = CachedTubeDataset(cache)
    _log(f"  CachedTubeDataset: {len(cached_dataset)} items")

    # ----------------------------------------------------------------
    # Step 2: Train AuxDecoder
    # ----------------------------------------------------------------
    _log("\nStep 2: Training AuxDecoder (SGD+momentum, MSE)...")
    decoder = AuxDecoder(
        input_dim=128, hidden_dim=args.hidden_dim, n_classes=label_encoder.n_classes()
    )
    decoder = train_aux_decoder(
        model=decoder,
        cache=cache,
        n_epochs=args.epochs,
        lr=args.lr,
        device=device,
        seed=args.seed,
        momentum=MOMENTUM,
        verbose=True,
        min_confidence=args.min_confidence,
    )

    ckpt_path = out_dir / "aux_decoder.pt"
    torch.save(decoder.state_dict(), ckpt_path)
    _log(f"\n  Saved checkpoint: {ckpt_path}")

    # ----------------------------------------------------------------
    # Step 3: Cytokine centroids
    # encoder_space=True  → h_i space, decoder-softmax weighted (Approach F)
    # encoder_space=False → g_i = decoder.embed(h_i) space (original)
    # ----------------------------------------------------------------
    space_label = "h_i (encoder), decoder-weighted" if args.encoder_space \
                  else "g_i = decoder.embed(h_i)"
    _log(f"\nStep 3: Computing cytokine centroids ({space_label})...")
    centroids = compute_cytokine_centroids(
        model=mil_model,
        dataset=dataset,
        label_encoder=label_encoder,
        device=args.device,
        decoder=decoder,
        encoder_space=args.encoder_space,
    )
    _report_alignment(centroids, label_encoder)

    # ----------------------------------------------------------------
    # Step 4: Directional bias (Experiment 1)
    # cell_type_obs reintroduced post-hoc from cache — never used during training
    # ----------------------------------------------------------------
    _log("\nStep 4: Directional bias (Experiment 1)...")
    cell_type_obs = {i: entry["cell_types"] for i, entry in enumerate(cache)}
    bias_result = compute_directional_bias(
        model=mil_model,
        dataset=dataset,
        label_encoder=label_encoder,
        centroids=centroids["centroids"],
        cell_type_obs=cell_type_obs,
        device=args.device,
        decoder=decoder,
        n_permutations=args.n_permutations,
        encoder_space=args.encoder_space,
    )

    # ----------------------------------------------------------------
    # Step 5: Asymmetry matrix (Experiment 2)
    # ----------------------------------------------------------------
    _log("\nStep 5: Asymmetry matrix (Experiment 2)...")
    asym_result = compute_asymmetry_matrix(
        bias_result["bias"], label_encoder
    )

    # ----------------------------------------------------------------
    # Save combined results
    # ----------------------------------------------------------------
    results = {
        "centroids": centroids,
        "bias": bias_result,
        "asymmetry": asym_result,
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "hidden_dim": args.hidden_dim,
            "encoder_space": args.encoder_space,
        },
    }
    out_path = out_dir / "latent_geometry.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    _log(f"\nResults saved: {out_path}")

    # Print top cascade candidates from asymmetry (PBS excluded — no cascade interpretation)
    asym_matrix = asym_result.get("asymmetry_matrix")
    if asym_matrix is not None:
        cytokine_names = asym_result.get("cytokine_names", [])
        _log("\nTop asymmetry scores (evidence for A→B cascade, PBS excluded):")
        pairs = []
        K = asym_matrix.shape[0]
        for a in range(K):
            for b in range(K):
                if a != b and cytokine_names[a] != "PBS" and cytokine_names[b] != "PBS":
                    pairs.append((asym_matrix[a, b], cytokine_names[a], cytokine_names[b]))
        pairs.sort(reverse=True)
        for score, a, b in pairs[:10]:
            _log(f"  {a:<20} → {b:<20}  asym={score:.4f}")

    _log("\nDone.")


if __name__ == "__main__":
    main()
