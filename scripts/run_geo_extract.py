"""
PBS-RC geometric detection: run on a MIL-finetuned encoder and extract
top top_pct% ordered cytokine pairs as cascade candidates.

Uses the encoder from model_stage3.pt (after MIL fine-tuning), which embeds
cells in a cytokine-discriminative space.  PBS cell-type centroids are
recomputed on the fly from PBS tubes in the training manifest using the
same encoder, ensuring centroid/embedding consistency.

Outputs (to --output_dir, default: --exp_dir):
  latent_geo_results.pkl  - full bias + Wilcoxon significance output
  top_pairs.json          - list of {A, B, relay_cell_type, p_bonf, rank}

Usage:
    python scripts/run_geo_extract.py --exp_dir results/two_stage_pipeline/exp_0_seed42
    python scripts/run_geo_extract.py --exp_dir ... --top_pct 0.05 --direction_mode global
    python scripts/run_geo_extract.py --exp_dir ... --model_file model_stage2.pt
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import anndata
import numpy as np
import torch
from tqdm import tqdm

from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model
from cytokine_mil.analysis.latent_geometry import (
    compute_directional_bias_per_donor,
    test_directional_significance,
)


HVG_PATH    = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
VAL_DONORS  = {"Donor2", "Donor3"}
N_CELL_TYPES = 18   # Oesinghaus dataset constant


def parse_args():
    p = argparse.ArgumentParser(description="PBS-RC geo detection on a MIL-finetuned encoder.")
    p.add_argument("--exp_dir",   required=True,
                   help="Experiment output directory containing model_stage3.pt (or "
                        "--model_file), manifest_train.json, label_encoder.json.")
    p.add_argument("--output_dir", default=None,
                   help="Where to save results (default: same as --exp_dir).")
    p.add_argument("--model_file", default="model_stage3.pt",
                   help="Model checkpoint file within --exp_dir (default: model_stage3.pt). "
                        "Must be a CytokineABMIL state dict.")
    p.add_argument("--top_pct",  type=float, default=0.05,
                   help="Fraction of ordered (A,B) pairs to include in top_pairs.json.")
    p.add_argument("--n_genes",  type=int,   default=4000)
    p.add_argument("--embed_dim", type=int,  default=128)
    p.add_argument("--attn_dim", type=int,   default=64)
    p.add_argument("--direction_mode", default="global",
                   choices=["global", "cell_type"],
                   help="Direction vector for PBS-RC projection.")
    p.add_argument("--alpha",    type=float, default=0.05,
                   help="Significance threshold for cascade calls (Bonferroni per pair).")
    return p.parse_args()


def _load_label_encoder(exp_dir: Path) -> CytokineLabel:
    with open(exp_dir / "label_encoder.json") as f:
        data = json.load(f)
    cytos = data["cytokines"]
    le = CytokineLabel()
    le._label_to_idx = {c: i for i, c in enumerate(cytos)}
    le._idx_to_label = {i: c for i, c in enumerate(cytos)}
    return le


@torch.no_grad()
def _embed_tube(encoder, adata, gene_names, device):
    """Run encoder on a single tube's expression matrix → (N, embed_dim) tensor."""
    if gene_names is not None:
        adata = adata[:, gene_names]
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_t = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)
    return encoder(X_t).cpu()


@torch.no_grad()
def build_cache(encoder, train_manifest, gene_names, label_encoder, device):
    """Run encoder on every training tube; return list of embedding dicts."""
    encoder.eval()
    encoder.to(device)
    cache = []

    for entry in tqdm(train_manifest, desc="Embedding training tubes"):
        donor = entry.get("donor", "unknown")
        if donor in VAL_DONORS:
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
def compute_pbs_centroids(encoder, train_manifest, gene_names, device):
    """Compute per-cell-type PBS centroids using the given encoder.

    Returns dict: {cell_type_str: np.ndarray of shape (embed_dim,)}.
    Only uses PBS tubes from training donors (excludes VAL_DONORS).
    """
    encoder.eval()
    encoder.to(device)

    # Accumulate embeddings per cell type
    ct_sums: dict = defaultdict(lambda: None)
    ct_counts: dict = defaultdict(int)

    pbs_tubes = [e for e in train_manifest
                 if e["cytokine"] == "PBS" and e.get("donor", "") not in VAL_DONORS]
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


def main():
    args    = parse_args()
    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cpu")

    # ── Label encoder ──────────────────────────────────────────────────────
    le = _load_label_encoder(exp_dir)
    print(f"Label encoder: {len(le._label_to_idx)} classes", flush=True)

    # ── HVG gene names ─────────────────────────────────────────────────────
    hvg_path = Path(HVG_PATH)
    gene_names = None
    if hvg_path.exists():
        with open(hvg_path) as f:
            gene_names = json.load(f)
        print(f"HVG list: {len(gene_names)} genes", flush=True)
    else:
        print(f"WARNING: HVG list not found at {HVG_PATH} — using all genes", flush=True)

    # ── Training manifest ──────────────────────────────────────────────────
    with open(exp_dir / "manifest_train.json") as f:
        train_manifest = json.load(f)
    train_donors = sorted({
        e["donor"] for e in train_manifest
        if e["donor"] not in VAL_DONORS
    })
    print(f"Training donors: {train_donors}  ({len(train_donors)} total)", flush=True)
    print(f"Training tubes : {len(train_manifest)}", flush=True)

    # ── Load MIL-finetuned model and extract encoder ─────────────────────
    model_path = exp_dir / args.model_file
    print(f"Loading model: {model_path.name}", flush=True)

    encoder = build_encoder(
        n_input_genes=args.n_genes,
        n_cell_types=N_CELL_TYPES,
        embed_dim=args.embed_dim,
    )
    model = build_mil_model(
        encoder,
        embed_dim=args.embed_dim,
        attention_hidden_dim=args.attn_dim,
        n_classes=le.n_classes(),
        encoder_frozen=False,
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    # Extract the encoder from the full MIL model
    encoder = model.encoder
    encoder.eval()
    print(f"  Encoder extracted from {args.model_file}", flush=True)

    # ── Compute PBS cell-type centroids with this encoder ─────────────────
    print("\nComputing PBS centroids with MIL-finetuned encoder...", flush=True)
    pbs_ct_means = compute_pbs_centroids(encoder, train_manifest, gene_names, device)

    # ── Build embedding cache ─────────────────────────────────────────────
    print("\nBuilding inference cache...", flush=True)
    cache = build_cache(encoder, train_manifest, gene_names, le, device)
    print(f"Cache: {len(cache)} tubes embedded", flush=True)

    # ── Directional bias per donor (PBS-RC space) ─────────────────────────
    print("\nComputing directional bias per donor...", flush=True)
    bias_result = compute_directional_bias_per_donor(
        cache, le, pbs_ct_means,
        train_donors=train_donors,
        direction_mode=args.direction_mode,
    )
    n_donors = len(bias_result["donors"])
    n_cytos  = len(bias_result["centroids"])
    n_ct     = len({ct for (_, ct) in bias_result["b_per_donor"]})
    print(f"  donors={n_donors}, cytokines={n_cytos}, cell_types={n_ct}", flush=True)

    # ── Wilcoxon significance test ─────────────────────────────────────────
    print("\nRunning Wilcoxon significance tests (this may take a few minutes)...", flush=True)
    sig_result = test_directional_significance(
        bias_result, le, alpha=args.alpha,
    )
    n_pairs_tested = len(sig_result["p_pair_fwd"])
    print(f"  Tested {n_pairs_tested} ordered pairs", flush=True)
    n_calls = sum(1 for v in sig_result["cascade_call"].values() if v != "none")
    print(f"  Non-trivial calls (A->B, B->A, shared): {n_calls}", flush=True)

    # ── Save full results ──────────────────────────────────────────────────
    with open(out_dir / "latent_geo_results.pkl", "wb") as f:
        pickle.dump({
            "bias_result":   bias_result,
            "sig_result":    sig_result,
            "pbs_ct_means":  pbs_ct_means,
            "model_file":    args.model_file,
            "args":          vars(args),
        }, f)
    print("\nSaved: latent_geo_results.pkl", flush=True)

    # ── Extract top top_pct% ordered pairs (exclude PBS) ─────────────────
    # Rank by W statistic (max Wilcoxon W across cell types), NOT p-value.
    # With n=10 donors the minimum achievable Bonferroni p saturates at ~0.018,
    # making all top-5% pairs look statistically identical when ranked by p.
    # W is a continuous score in [0, n*(n+1)/2] (= [0, 55] for n=10) that
    # reflects how consistently all donors point in the forward direction.
    p_pair_fwd = sig_result["p_pair_fwd"]   # {(A, B) -> min Bonferroni p}
    W_pair_fwd = sig_result["W_pair_fwd"]   # {(A, B) -> max W across cell types}
    relay_T    = sig_result["relay_T"]       # {(A, B) -> best relay cell type}

    scored = []
    for (a, b), p in p_pair_fwd.items():
        if a == "PBS" or b == "PBS":
            continue
        scored.append({
            "A": a, "B": b,
            "relay_cell_type": relay_T.get((a, b)),
            "p_bonf":          float(p),
            "W_stat":          float(W_pair_fwd.get((a, b), 0.0)),
        })

    # Sort by W descending (higher W = more consistent signal across donors).
    # Secondary sort by p ascending for ties in W.
    scored.sort(key=lambda x: (-x["W_stat"], x["p_bonf"]))

    n_total = len(scored)
    n_top   = max(1, int(round(n_total * args.top_pct)))
    top_pairs = scored[:n_top]
    for rank, entry in enumerate(top_pairs, 1):
        entry["rank"] = rank

    with open(out_dir / "top_pairs.json", "w") as f:
        json.dump(top_pairs, f, indent=2)

    print(f"\nTotal scored pairs  : {n_total}", flush=True)
    print(f"Top {args.top_pct*100:.1f}%           : {n_top} pairs → top_pairs.json", flush=True)
    print(f"Ranking             : W_stat descending (not p-value)", flush=True)

    # Print summary table
    header = f"{'Rank':>4}  {'A':<18}  {'B':<18}  {'relay_T':<22}  {'W_stat':>7}  {'p_bonf':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for entry in top_pairs[:25]:
        rt = str(entry["relay_cell_type"] or "?")
        print(f"{entry['rank']:>4}  {entry['A']:<18}  {entry['B']:<18}  "
              f"{rt:<22}  {entry['W_stat']:>7.1f}  {entry['p_bonf']:>10.4f}", flush=True)
    if n_top > 25:
        print(f"  ... ({n_top - 25} more pairs not shown)")

    print(f"\nResults saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
