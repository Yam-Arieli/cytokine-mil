"""
PBS-RC geometric detection: run on a trained stage1 encoder and extract
top top_pct% ordered cytokine pairs as cascade candidates.

Uses `encoder_stage1.pt` (pure cell-type embedding, before MIL objective
distorts the geometry) and `pbs_ct_means.pkl` (PBS centroids computed with
the same encoder during training).

Outputs (to --output_dir, default: --exp_dir):
  latent_geo_results.pkl  - full bias + Wilcoxon significance output
  top_pairs.json          - list of {A, B, relay_cell_type, p_bonf, rank}

Usage:
    python scripts/run_geo_extract.py --exp_dir results/two_stage_pipeline/exp_0_seed42
    python scripts/run_geo_extract.py --exp_dir ... --top_pct 0.05 --direction_mode global
"""

import argparse
import json
import pickle
from pathlib import Path

import anndata
import numpy as np
import torch
from tqdm import tqdm

from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder
from cytokine_mil.analysis.latent_geometry import (
    compute_directional_bias_per_donor,
    test_directional_significance,
)


HVG_PATH    = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
VAL_DONORS  = {"Donor2", "Donor3"}
N_CELL_TYPES = 18   # Oesinghaus dataset constant


def parse_args():
    p = argparse.ArgumentParser(description="PBS-RC geo detection on a trained stage1 encoder.")
    p.add_argument("--exp_dir",   required=True,
                   help="Experiment output directory containing encoder_stage1.pt, "
                        "manifest_train.json, label_encoder.json, pbs_ct_means.pkl.")
    p.add_argument("--output_dir", default=None,
                   help="Where to save results (default: same as --exp_dir).")
    p.add_argument("--top_pct",  type=float, default=0.05,
                   help="Fraction of ordered (A,B) pairs to include in top_pairs.json.")
    p.add_argument("--n_genes",  type=int,   default=4000)
    p.add_argument("--embed_dim", type=int,  default=128)
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
def build_cache(encoder, train_manifest, gene_names, label_encoder, device):
    """Run encoder_stage1 on every training tube; return list of embedding dicts."""
    encoder.eval()
    encoder.to(device)
    cache = []

    for entry in tqdm(train_manifest, desc="Embedding training tubes"):
        donor = entry.get("donor", "unknown")
        if donor in VAL_DONORS:
            continue
        try:
            adata = anndata.read_h5ad(entry["path"])
            if gene_names is not None:
                adata = adata[:, gene_names]
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            X_t = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)
            H = encoder(X_t).cpu()          # (N, embed_dim)
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

    # ── PBS cell-type centroids (computed with stage1 encoder during training) ─
    pbs_path = exp_dir / "pbs_ct_means.pkl"
    with open(pbs_path, "rb") as f:
        pbs_ct_means = pickle.load(f)
    print(f"pbs_ct_means: {len(pbs_ct_means)} cell types", flush=True)

    # ── Stage 1 encoder ────────────────────────────────────────────────────
    encoder = build_encoder(
        n_input_genes=args.n_genes,
        n_cell_types=N_CELL_TYPES,
        embed_dim=args.embed_dim,
    )
    state = torch.load(exp_dir / "encoder_stage1.pt", map_location="cpu")
    encoder.load_state_dict(state)
    encoder.eval()
    print("Loaded encoder_stage1.pt", flush=True)

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
            "bias_result": bias_result,
            "sig_result":  sig_result,
            "args":        vars(args),
        }, f)
    print("\nSaved: latent_geo_results.pkl", flush=True)

    # ── Extract top top_pct% ordered pairs (exclude PBS) ─────────────────
    p_pair_fwd = sig_result["p_pair_fwd"]   # {(A, B) -> min Bonferroni p}
    relay_T    = sig_result["relay_T"]       # {(A, B) -> best relay cell type}

    scored = []
    for (a, b), p in p_pair_fwd.items():
        if a == "PBS" or b == "PBS":
            continue
        scored.append({
            "A": a, "B": b,
            "relay_cell_type": relay_T.get((a, b)),
            "p_bonf":          float(p),
        })

    scored.sort(key=lambda x: x["p_bonf"])

    n_total = len(scored)
    n_top   = max(1, int(round(n_total * args.top_pct)))
    top_pairs = scored[:n_top]
    for rank, entry in enumerate(top_pairs, 1):
        entry["rank"] = rank

    with open(out_dir / "top_pairs.json", "w") as f:
        json.dump(top_pairs, f, indent=2)

    print(f"\nTotal scored pairs  : {n_total}", flush=True)
    print(f"Top {args.top_pct*100:.1f}%           : {n_top} pairs → top_pairs.json", flush=True)

    # Print summary table
    header = f"{'Rank':>4}  {'A':<18}  {'B':<18}  {'relay_T':<22}  {'p_bonf':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for entry in top_pairs[:25]:
        rt = str(entry["relay_cell_type"] or "?")
        print(f"{entry['rank']:>4}  {entry['A']:<18}  {entry['B']:<18}  "
              f"{rt:<22}  {entry['p_bonf']:>10.4f}", flush=True)
    if n_top > 25:
        print(f"  ... ({n_top - 25} more pairs not shown)")

    print(f"\nResults saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
