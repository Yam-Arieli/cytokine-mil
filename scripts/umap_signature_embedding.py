#!/usr/bin/env python
"""
Compute 3 UMAP embeddings of the SAME cells (one cytokine vs PBS), each
restricted to a different gene set, to test whether a discovered signature
S_X carries condition-discriminating information beyond what's visible in
the full HVG panel or in an arbitrary same-size gene set.

Purpose (thesis Results figure): there is no ground truth for "is S_X a good
signature" (it's defined instrumentally, as the top genes by IG through a
classifier trained on exactly this discrimination). What IS checkable without
circularity: does restricting to S_X reveal condition structure that (a) all
4000 HVGs together and (b) a random 50-gene set of the same size do NOT show?

Panels (same cells, same PCA/UMAP hyperparameters, only the gene subset differs):
  1. all_hvg   -- every HVG (4000) -- the current appendix UMAP's baseline
  2. signature -- S_X: top `top_n` genes by rank_ig for `--cytokine`, read from
                  a binary_ig.parquet (cytokine, gene, ig, mean_expression, rank_ig)
  3. random    -- `top_n` genes sampled from the HVG list, excluding S_X
                  (seeded) -- the honest, size-matched control

Data: Oesinghaus pseudo-tubes, ALL entries where cytokine in {--cytokine, PBS},
capped at --max_tubes_per_condition each (deterministic: sorted by donor then
tube_idx). Tubes are already normalized + log1p + HVG-subset.

Outputs (under --out_dir):
  umap_coords.parquet   -- long format: [variant, umap_x, umap_y, cell_type,
                            donor, condition] (one row per cell per variant)
  stats.json            -- per-variant: n_cells, n_genes, silhouette (PCA space,
                            condition labels) -- the quantitative separability
                            statistic paired with the visual panels.
  signature_genes.json  -- the exact S_X and random gene lists used (provenance)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest_path",
        default="/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json",
    )
    p.add_argument(
        "--hvg_path",
        default="/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json",
    )
    p.add_argument(
        "--binary_ig_parquet",
        default="results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet",
        help="Long-format (cytokine, gene, ig, mean_expression, rank_ig) parquet "
        "from scripts/run_binary_ig_probe.py / merge_binary_ig_parquets.py.",
    )
    p.add_argument("--cytokine", default="IFN-gamma", help="Cytokine to test (vs PBS).")
    p.add_argument("--control_label", default="PBS")
    p.add_argument("--top_n", type=int, default=50, help="|S_X| -- must match the thesis definition.")
    p.add_argument(
        "--max_tubes_per_condition",
        type=int,
        default=40,
        help="Cap tubes per condition (deterministic: sorted by donor, tube_idx).",
    )
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--random_seed", type=int, default=0, help="Seed for the random gene-set control.")
    p.add_argument("--out_dir", default="results/umap_signature")
    return p.parse_args()


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def select_entries(manifest: List[dict], cytokine: str, control_label: str, cap: int) -> List[dict]:
    by_cond: Dict[str, List[dict]] = defaultdict(list)
    for e in manifest:
        if e.get("cytokine") in (cytokine, control_label):
            by_cond[e["cytokine"]].append(e)
    for cond in (cytokine, control_label):
        if not by_cond.get(cond):
            raise ValueError(f"No manifest entries found for condition {cond!r}.")
        by_cond[cond].sort(key=lambda e: (e.get("donor", ""), int(e.get("tube_idx", 0))))
        by_cond[cond] = by_cond[cond][:cap]
    return by_cond[cytokine] + by_cond[control_label]


def load_and_concat(entries: List[dict], cytokine: str, control_label: str) -> ad.AnnData:
    adatas = []
    for e in entries:
        a = ad.read_h5ad(e["path"])
        a.obs["condition"] = "cytokine" if e["cytokine"] == cytokine else "PBS"
        a.obs["donor"] = e.get("donor", "unknown")
        if "cell_type" not in a.obs.columns:
            raise KeyError(f"'cell_type' missing in obs of {e['path']}.")
        adatas.append(a)
    combined = ad.concat(adatas, join="inner", index_unique="-")
    return combined


def build_gene_sets(
    binary_ig_parquet: str, cytokine: str, hvg_genes: List[str], top_n: int, seed: int,
) -> Dict[str, List[str]]:
    ig_df = pd.read_parquet(binary_ig_parquet)
    ig_df = ig_df[ig_df["cytokine"] == cytokine]
    if ig_df.empty:
        raise ValueError(
            f"Cytokine {cytokine!r} not found in {binary_ig_parquet}. "
            f"Available: {sorted(pd.read_parquet(binary_ig_parquet)['cytokine'].unique())}"
        )
    ig_df = ig_df.sort_values("rank_ig").head(top_n)
    signature = ig_df["gene"].tolist()
    if len(signature) < top_n:
        raise ValueError(f"Only {len(signature)} genes for {cytokine!r}, expected {top_n}.")

    rng = np.random.default_rng(seed)
    pool = [g for g in hvg_genes if g not in set(signature)]
    random_genes = rng.choice(pool, size=top_n, replace=False).tolist()

    return {"all_hvg": list(hvg_genes), "signature": signature, "random": random_genes}


def embed_variant(adata: ad.AnnData, genes: List[str], n_pcs: int) -> tuple[np.ndarray, float]:
    """Subset to `genes`, run PCA -> neighbors -> UMAP; return (umap_xy, silhouette)."""
    import scanpy as sc  # local import: scanpy is heavy
    from sklearn.metrics import silhouette_score

    avail = [g for g in genes if g in adata.var_names]
    sub = adata[:, avail].copy()

    n_comps = int(min(n_pcs, sub.n_vars - 1, sub.n_obs - 1))
    sc.pp.pca(sub, n_comps=n_comps)
    sc.pp.neighbors(sub)
    sc.tl.umap(sub)

    coords = np.asarray(sub.obsm["X_umap"], dtype=float)
    # Separability statistic computed in PCA space (not UMAP space -- UMAP
    # distances are for visualization, PCA space is the more principled
    # metric space for a silhouette score).
    labels = sub.obs["condition"].to_numpy()
    try:
        sil = float(silhouette_score(sub.obsm["X_pca"], labels))
    except ValueError:
        sil = float("nan")
    return coords, sil, len(avail), sub.n_obs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hvg_genes = load_json(args.hvg_path)
    print(f"HVG count: {len(hvg_genes)}")

    manifest = load_json(args.manifest_path)
    entries = select_entries(manifest, args.cytokine, args.control_label, args.max_tubes_per_condition)
    print(f"Selected {len(entries)} tubes for {args.cytokine!r} vs {args.control_label!r}.")

    adata = load_and_concat(entries, args.cytokine, args.control_label)
    print(f"Concatenated: {adata.n_obs} cells x {adata.n_vars} genes.")
    print(adata.obs["condition"].value_counts().to_string())

    gene_sets = build_gene_sets(
        args.binary_ig_parquet, args.cytokine, hvg_genes, args.top_n, args.random_seed,
    )
    with open(out_dir / "signature_genes.json", "w") as fh:
        json.dump(
            {
                "cytokine": args.cytokine,
                "top_n": args.top_n,
                "random_seed": args.random_seed,
                **gene_sets,
            },
            fh,
            indent=2,
        )

    all_rows = []
    stats = {}
    for variant in ["all_hvg", "signature", "random"]:
        genes = gene_sets[variant]
        print(f"\n>>> variant={variant} ({len(genes)} genes) <<<")
        coords, sil, n_genes_used, n_cells = embed_variant(adata, genes, args.n_pcs)
        stats[variant] = {"n_cells": n_cells, "n_genes": n_genes_used, "silhouette": sil}
        print(f"  n_genes_used={n_genes_used}  silhouette(condition, PCA space)={sil:+.4f}")

        df = pd.DataFrame(
            {
                "variant": variant,
                "umap_x": coords[:, 0],
                "umap_y": coords[:, 1],
                "cell_type": adata.obs["cell_type"].astype(str).to_numpy(),
                "donor": adata.obs["donor"].astype(str).to_numpy(),
                "condition": adata.obs["condition"].astype(str).to_numpy(),
            }
        )
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    coords_path = out_dir / "umap_coords.parquet"
    combined.to_parquet(coords_path)
    print(f"\nWrote {coords_path} ({len(combined)} rows).")

    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as fh:
        json.dump({"cytokine": args.cytokine, "top_n": args.top_n, "variants": stats}, fh, indent=2)
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
