"""
Alignment-based cytokine pair detection on a MIL-finetuned encoder.

For each unordered pair (A, B) with A < B (and A, B ≠ PBS), per cell type T,
per training donor d:
  - Compute PBS-RC centroid v_{C,T,d} = µ_{C,T,d} - µ_{PBS,T} for C in {A, B}.
  - Score the pair under each metric (cosine, raw inner product) at multiple
    dimensionalities (PCA-6D, PCA-24D, full 128D no-PCA).
Aggregate across donors (mean), take max over cell types for pair-level score.

Outputs (to --output_dir, default: --exp_dir):
  inner_product_results.pkl                  - full nested results dict
  top_pairs_{metric}_{dim}D.json (6 files)   - per-variant ranked top pairs

Each top-pairs entry: {A, B, relay_cell_type, score, rank}.

Usage:
    python scripts/run_inner_product_pairs.py \\
        --exp_dir results/two_stage_pipeline/exp_0_seed42 \\
        --top_pct 0.10 \\
        --pca_dims 6 24
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from cytokine_mil.experiment_setup import build_encoder, build_mil_model
from cytokine_mil.analysis.encoder_cache import (
    VAL_DONORS,
    _load_label_encoder,
    build_cache,
    compute_pbs_centroids,
)
from cytokine_mil.analysis.pair_alignment import (
    METRIC_DESCRIPTIONS,
    compute_per_atd_centroids,
    fit_pca_projections,
    compute_pair_scores,
    rank_and_format_top_pairs,
)


HVG_PATH    = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
N_CELL_TYPES = 18   # Oesinghaus dataset constant
METRICS = ("cosine", "inner_product")


def parse_args():
    p = argparse.ArgumentParser(
        description="Alignment-based pair detection on a MIL-finetuned encoder."
    )
    p.add_argument("--exp_dir", required=True,
                   help="Experiment dir with model_stage3.pt, manifest_train.json, "
                        "label_encoder.json.")
    p.add_argument("--output_dir", default=None,
                   help="Where to write results (default: same as --exp_dir).")
    p.add_argument("--model_file", default="model_stage3.pt",
                   help="Checkpoint file within --exp_dir.")
    p.add_argument("--top_pct", type=float, default=0.10,
                   help="Fraction of pairs to save in top_pairs_*.json (default 0.10).")
    p.add_argument("--pca_dims", type=int, nargs="*", default=[6, 24],
                   help="PCA dimensionalities to compute (default: 6 24). "
                        "Full no-PCA dim is always included as baseline.")
    p.add_argument("--n_genes",  type=int, default=4000)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--attn_dim",  type=int, default=64)
    p.add_argument("--seed", type=int, default=0,
                   help="PCA random_state.")
    return p.parse_args()


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
    encoder = model.encoder
    encoder.eval()
    print(f"  Encoder extracted from {args.model_file}", flush=True)

    # ── PBS centroids ─────────────────────────────────────────────────────
    print("\nComputing PBS centroids with MIL-finetuned encoder...", flush=True)
    pbs_ct_means = compute_pbs_centroids(encoder, train_manifest, gene_names, device)

    # ── Embedding cache ───────────────────────────────────────────────────
    print("\nBuilding embedding cache...", flush=True)
    cache = build_cache(encoder, train_manifest, gene_names, le, device)
    print(f"Cache: {len(cache)} tubes embedded", flush=True)

    # ── Per-(cyto, ct, donor) PBS-RC centroids ───────────────────────────
    print("\nComputing per-(cytokine, cell_type, donor) PBS-RC centroids...", flush=True)
    atd_centroids = compute_per_atd_centroids(
        cache, le, pbs_ct_means, train_donors=train_donors,
    )
    n_cytos = len({c for (c, _, _) in atd_centroids})
    n_cts   = len({t for (_, t, _) in atd_centroids})
    n_dons  = len({d for (_, _, d) in atd_centroids})
    print(f"  centroids: {len(atd_centroids)} triples "
          f"(cytos={n_cytos}, cell_types={n_cts}, donors={n_dons})",
          flush=True)

    # ── PCA projections ───────────────────────────────────────────────────
    print(f"\nFitting PCA at dims {args.pca_dims} (plus full {args.embed_dim}D)...",
          flush=True)
    projections = fit_pca_projections(
        atd_centroids,
        n_components_list=args.pca_dims,
        random_state=args.seed,
    )
    dims = sorted(projections.keys(), reverse=True)
    print(f"  available dims: {dims}", flush=True)

    # ── Pair scoring across all (metric, dim) variants ───────────────────
    print(f"\nScoring pairs across metrics={METRICS} × dims={dims}...", flush=True)
    all_pair_scores = {}     # {(metric, dim): {(A, B) -> score}}
    all_relay_T     = {}     # {(metric, dim): {(A, B) -> ct}}
    all_full_table  = {}     # {(metric, dim): {(A, B, T) -> {...}}}
    all_top_pairs   = {}     # {(metric, dim): list of dicts}

    for d in dims:
        proj = projections[d]
        pair_scores, relay_T, full_table = compute_pair_scores(proj, metrics=METRICS)
        for metric in METRICS:
            key = (metric, d)
            all_pair_scores[key] = pair_scores[metric]
            all_relay_T[key]     = relay_T[metric]
            all_full_table[key]  = full_table[metric]
            top = rank_and_format_top_pairs(
                pair_scores[metric], relay_T[metric], top_pct=args.top_pct,
            )
            all_top_pairs[key] = top
            print(f"  [{metric:>13}, {d:>3}D] scored {len(pair_scores[metric])} pairs, "
                  f"top {len(top)} saved", flush=True)

    # ── Save results ──────────────────────────────────────────────────────
    with open(out_dir / "inner_product_results.pkl", "wb") as f:
        pickle.dump({
            "atd_centroids":   atd_centroids,
            "projections":     projections,
            "pair_scores":     all_pair_scores,
            "relay_T":         all_relay_T,
            "full_table":      all_full_table,
            "metrics":         list(METRICS),
            "dims":            dims,
            "pbs_ct_means":    pbs_ct_means,
            "model_file":      args.model_file,
            "args":            vars(args),
            "metric_descriptions": METRIC_DESCRIPTIONS,
        }, f)
    print(f"\nSaved: inner_product_results.pkl", flush=True)

    for (metric, d), top in all_top_pairs.items():
        fname = f"top_pairs_{metric}_{d}D.json"
        with open(out_dir / fname, "w") as f:
            json.dump(top, f, indent=2)
        print(f"Saved: {fname}  ({len(top)} pairs)", flush=True)

    # ── Preview tables ────────────────────────────────────────────────────
    for (metric, d), top in all_top_pairs.items():
        print(f"\n=== Top 20 — metric={metric}, dim={d}D ===")
        header = f"{'Rank':>4}  {'A':<18}  {'B':<18}  {'relay_T':<22}  {'score':>10}"
        print(header)
        print("-" * len(header))
        for entry in top[:20]:
            rt = str(entry["relay_cell_type"] or "?")
            print(f"{entry['rank']:>4}  {entry['A']:<18}  {entry['B']:<18}  "
                  f"{rt:<22}  {entry['score']:>10.4f}", flush=True)
        if len(top) > 20:
            print(f"  ... ({len(top) - 20} more pairs not shown)")

    print(f"\nResults saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
