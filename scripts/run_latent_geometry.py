"""
Latent Space Cytokine Geometry — Experiments 0, 1, 2.

Loads a trained CytokineABMIL checkpoint and runs:
  Exp 0 — cytokine alignment gate (does geometry exist at cell level?)
  Exp 1 — per-cell-type directional bias toward other cytokine centroids
  Exp 2 — asymmetry matrix for cascade direction inference

Centroids computed on training donors only. Cell-type labels loaded
post-hoc from h5ad obs["cell_type"]; never used during training.

Results saved to --output_dir/latent_geometry_seed{N}.pkl

Usage:
    python scripts/run_latent_geometry.py --run_dir results/oesinghaus_full/run_20260412_161758_seed42
    python scripts/run_latent_geometry.py --run_dir results/oesinghaus_full/run_20260412_161803_seed123
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model
from cytokine_mil.analysis.latent_geometry import (
    compute_cytokine_centroids,
    compute_alignment_scores,
    compute_directional_bias,
    compute_asymmetry_matrix,
    build_latent_cascade_graph,
)

HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"

# Known positive control for Exp 1 gate check
POSITIVE_CONTROL = ("IL-12", "IFN-gamma", "NK")  # expected: bias > 0, large z
NEGATIVE_CONTROL = ("IL-6", "IL-10")              # expected: ~symmetric ASYM


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                   help="Path to a trained run directory (contains model_stage2.pt, "
                        "manifest_train.json, label_encoder.json).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to save latent_geometry_results.pkl. "
                        "Defaults to --run_dir.")
    p.add_argument("--n_permutations", type=int, default=1000,
                   help="Permutations for null distributions (default 1000).")
    p.add_argument("--fdr_alpha", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


def _load_label_encoder(run_dir: Path) -> CytokineLabel:
    with open(run_dir / "label_encoder.json") as f:
        data = json.load(f)
    le = CytokineLabel()
    le.cytokines = data["cytokines"]
    le._cytokine_to_idx = {c: i for i, c in enumerate(data["cytokines"])}
    return le


def _infer_n_cell_types(state_dict: dict) -> int:
    """Infer n_cell_types from the encoder classification head weight shape."""
    key = "encoder.classification_head.weight"
    if key in state_dict:
        return state_dict[key].shape[0]
    # Fallback: scan for classification head
    for k, v in state_dict.items():
        if "classification_head" in k and "weight" in k:
            return v.shape[0]
    raise ValueError("Cannot infer n_cell_types from state dict — "
                     "classification_head not found.")


def _load_model(run_dir: Path, label_enc: CytokineLabel,
                gene_names: list, device: str) -> torch.nn.Module:
    state_dict = torch.load(run_dir / "model_stage2.pt", map_location="cpu")
    n_cell_types = _infer_n_cell_types(state_dict)
    _log(f"  Inferred n_cell_types={n_cell_types} from checkpoint.")

    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=n_cell_types,
        embed_dim=128,
    )
    model = build_mil_model(
        encoder,
        embed_dim=128,
        attention_hidden_dim=64,
        n_classes=label_enc.n_classes(),
        encoder_frozen=True,
    )
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    model.eval()
    return model


def _build_cell_type_obs(dataset: PseudoTubeDataset) -> dict:
    """
    Load cell-type labels for each tube from h5ad obs["cell_type"].
    Returns {tube_index: np.array of cell_type strings}.
    Cell-type labels are never passed to the model; this is post-hoc annotation.
    """
    _log("  Loading cell-type labels from h5ad metadata...")
    cell_type_obs = {}
    for idx, entry in enumerate(dataset.entries):
        adata = sc.read_h5ad(entry["path"])
        if "cell_type" in adata.obs.columns:
            cell_type_obs[idx] = adata.obs["cell_type"].values
        else:
            _log(f"    WARNING: tube {idx} ({entry['path']}) has no cell_type column.")
    _log(f"  Cell-type obs loaded for {len(cell_type_obs)}/{len(dataset)} tubes.")
    return cell_type_obs


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _report_exp0(result: dict, label_enc) -> bool:
    """Print Exp 0 results and return True if gate passes."""
    _log()
    _log("=" * 65)
    _log("EXPERIMENT 0 — Cytokine Alignment Gate (GO/NO-GO)")
    _log("=" * 65)
    _log(f"Metric: {result['metric_description']}")
    _log(f"Null mean: {result['null_mean']:.4f}  Null std: {result['null_std']:.4f}")
    _log()

    scores = result["alignment_scores"]
    pvals  = result["p_values"]
    sorted_cyts = sorted(scores, key=lambda c: scores[c], reverse=True)

    _log(f"{'Cytokine':<24} {'Alignment':>10} {'p-value':>10}  {'Sig?':>5}")
    _log("-" * 56)
    for c in sorted_cyts:
        sig = "*" if pvals[c] < 0.05 else ""
        _log(f"  {c:<22} {scores[c]:>10.4f} {pvals[c]:>10.4f}  {sig:>5}")

    n_sig = sum(1 for p in pvals.values() if p < 0.05)
    n_above_null = sum(1 for s in scores.values() if s > result["null_mean"])
    _log()
    _log(f"Significant (p<0.05): {n_sig}/{len(scores)} cytokines")
    _log(f"Above null mean: {n_above_null}/{len(scores)} cytokines")

    gate_pass = n_above_null > len(scores) / 2
    _log()
    if gate_pass:
        _log("GATE: PASS — cytokine geometry exists at cell level. Proceed to Exp 1+2.")
    else:
        _log("GATE: FAIL — alignment ≈ null. Encoder embeds cell-type space only.")
        _log("  → Implement Experiment 3 (auxiliary decoder). See CLAUDE.md Section 20.5.")
    return gate_pass


def _report_exp1(bias_result: dict, asym_result: dict):
    _log()
    _log("=" * 65)
    _log("EXPERIMENT 1 — Per-Cell-Type Directional Bias")
    _log("=" * 65)
    _log(f"Metric: {bias_result['metric_description']}")
    _log()

    z = bias_result["z_scores"]
    q = bias_result["q_values"]

    # Positive control: IL-12 → IFN-gamma
    _log("Positive control: IL-12 → IFN-gamma (expected bias > 0 in NK cells)")
    for ct in ["NK", "CD14_Mono", "CD4_T", "CD8_T"]:
        key = ("IL-12", "IFN-gamma", ct)
        key_rev = ("IFN-gamma", "IL-12", ct)
        z_fwd = z.get(key, float("nan"))
        z_rev = z.get(key_rev, float("nan"))
        q_fwd = q.get(key, float("nan"))
        sig = "*" if q_fwd < 0.05 else ""
        _log(f"  bias(IL-12→IFN-γ, {ct:<12}) z={z_fwd:+.2f}  q={q_fwd:.3f} {sig}"
             f"  |  bias(IFN-γ→IL-12, {ct:<12}) z={z_rev:+.2f}")

    _log()
    _log("Negative control: IL-6 / IL-10 asymmetry (expected ~0)")
    for ct in ["CD14_Mono", "NK", "CD4_T"]:
        key_ab = ("IL-6", "IL-10", ct)
        key_ba = ("IL-10", "IL-6", ct)
        z_ab = z.get(key_ab, float("nan"))
        z_ba = z.get(key_ba, float("nan"))
        _log(f"  bias(IL-6→IL-10, {ct:<12}) z={z_ab:+.2f}  |  "
             f"bias(IL-10→IL-6, {ct:<12}) z={z_ba:+.2f}")

    _log()
    _log("Top 20 most significant directional biases (FDR-corrected):")
    _log(f"{'Source':<14} {'Target':<14} {'CellType':<14} {'z-score':>8} {'q-value':>10}")
    _log("-" * 65)
    sig_triples = sorted(
        [(k, v) for k, v in q.items() if not np.isnan(v)],
        key=lambda x: x[1],
    )[:20]
    for (ca, cb, ct), qval in sig_triples:
        zval = z.get((ca, cb, ct), float("nan"))
        _log(f"  {ca:<14} {cb:<14} {ct:<14} {zval:>8.2f} {qval:>10.4f}")


def _report_exp2(asym_result: dict, cascade_graph):
    _log()
    _log("=" * 65)
    _log("EXPERIMENT 2 — Asymmetry Matrix (Cascade Direction)")
    _log("=" * 65)
    _log(f"Metric: {asym_result['metric_description']}")
    _log()

    A = asym_result["asymmetry_matrix"]
    names = asym_result["cytokine_names"]
    K = len(names)

    # Top directional pairs by |ASYM|
    pairs = []
    for i in range(K):
        for j in range(K):
            if i != j:
                pairs.append((names[i], names[j], float(A[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)

    _log("Top 20 directional pairs (positive ASYM = evidence for A→B):")
    _log(f"{'Source':<20} {'Target':<20} {'ASYM':>8}")
    _log("-" * 52)
    for src, tgt, asym in pairs[:20]:
        _log(f"  {src:<20} {tgt:<20} {asym:>8.4f}")

    _log()
    _log("Positive control asymmetry: IL-12 → IFN-gamma")
    il12_idx = names.index("IL-12") if "IL-12" in names else None
    ifng_idx = names.index("IFN-gamma") if "IFN-gamma" in names else None
    if il12_idx is not None and ifng_idx is not None:
        fwd = A[il12_idx, ifng_idx]
        rev = A[ifng_idx, il12_idx]
        _log(f"  ASYM(IL-12→IFN-γ) = {fwd:+.4f}")
        _log(f"  ASYM(IFN-γ→IL-12) = {rev:+.4f}")
        _log(f"  Direction correct: {fwd > 0} (expected True)")

    _log()
    _log("Negative control: IL-6 / IL-10")
    il6_idx  = names.index("IL-6")  if "IL-6"  in names else None
    il10_idx = names.index("IL-10") if "IL-10" in names else None
    if il6_idx is not None and il10_idx is not None:
        fwd = A[il6_idx, il10_idx]
        rev = A[il10_idx, il6_idx]
        _log(f"  ASYM(IL-6→IL-10)  = {fwd:+.4f}")
        _log(f"  ASYM(IL-10→IL-6)  = {rev:+.4f}")
        _log(f"  Near-symmetric: {abs(fwd - rev) < 0.01} (expected True)")

    if cascade_graph is not None:
        _log()
        _log(f"Cascade graph edges (FDR-significant + ASYM>0): "
             f"{cascade_graph.number_of_edges()} edges")
        edges = sorted(cascade_graph.edges(data=True),
                       key=lambda e: e[2]["asymmetry"], reverse=True)
        for src, tgt, attrs in edges[:15]:
            _log(f"  {src:<20} → {tgt:<20}  "
                 f"ASYM={attrs['asymmetry']:.4f}  max_z={attrs['max_z']:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir

    _log(f"Run directory : {run_dir}")
    _log(f"Output directory: {out_dir}")
    _log(f"Device: {args.device}")
    _log(f"Permutations: {args.n_permutations}")

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    _log("\nLoading gene names...")
    with open(HVG_PATH) as f:
        gene_names = json.load(f)
    _log(f"  {len(gene_names)} HVGs")

    _log("Loading label encoder...")
    label_enc = _load_label_encoder(run_dir)
    _log(f"  {label_enc.n_classes()} classes")

    _log("Loading model...")
    model = _load_model(run_dir, label_enc, gene_names, args.device)
    _log("  Model loaded.")

    _log("Loading training dataset (train manifest)...")
    train_dataset = PseudoTubeDataset(
        str(run_dir / "manifest_train.json"),
        label_enc,
        gene_names=gene_names,
        preload=False,  # sequential forward pass — LRU cache is sufficient
    )
    _log(f"  {len(train_dataset)} tubes in training set.")

    cell_type_obs = _build_cell_type_obs(train_dataset)

    # ------------------------------------------------------------------
    # Experiment 0: Cytokine alignment gate
    # ------------------------------------------------------------------
    _log("\nComputing cytokine centroids (training tubes only)...")
    centroid_result = compute_cytokine_centroids(
        model, train_dataset, label_enc, device=args.device,
    )
    centroids = centroid_result["centroids"]
    _log(f"  Centroids computed for {len(centroids)} cytokines.")

    _log(f"Running alignment score computation ({args.n_permutations} permutations)...")
    alignment_result = compute_alignment_scores(
        model, train_dataset, label_enc, centroids,
        n_permutations=args.n_permutations,
        device=args.device,
    )

    gate_pass = _report_exp0(alignment_result, label_enc)

    # ------------------------------------------------------------------
    # Experiments 1 & 2: Directional bias + asymmetry
    # ------------------------------------------------------------------
    _log(f"\nComputing directional bias ({args.n_permutations} permutations, "
         f"this may take ~30–60 min for 1000 permutations)...")
    bias_result = compute_directional_bias(
        model, train_dataset, label_enc, centroids, cell_type_obs,
        n_permutations=args.n_permutations,
        device=args.device,
    )
    _log(f"  Bias computed for {len(bias_result['bias'])} (A, B, T) triples.")

    _log("Computing asymmetry matrix...")
    asym_result = compute_asymmetry_matrix(bias_result["bias"], label_enc)

    _log("Building cascade graph...")
    try:
        cascade_graph = build_latent_cascade_graph(
            asym_result["asymmetry_matrix"],
            bias_result["z_scores"],
            label_enc,
            fdr_alpha=args.fdr_alpha,
        )
    except ImportError as e:
        _log(f"  WARNING: {e}. Skipping graph construction.")
        cascade_graph = None

    _report_exp1(bias_result, asym_result)
    _report_exp2(asym_result, cascade_graph)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = out_dir / "latent_geometry_results.pkl"
    payload = {
        "centroid_result":   centroid_result,
        "alignment_result":  alignment_result,
        "bias_result":       bias_result,
        "asym_result":       asym_result,
        "gate_pass":         gate_pass,
        "run_dir":           str(run_dir),
        "n_permutations":    args.n_permutations,
        "fdr_alpha":         args.fdr_alpha,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    _log(f"\nResults saved to: {out_path}")
    _log("Done.")


if __name__ == "__main__":
    main()
