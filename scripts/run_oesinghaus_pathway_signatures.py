#!/usr/bin/env python3
"""
Pathway-signature cascade analysis on Oesinghaus 24h PBMC (91 cytokines, 4000 HVGs).

Mirrors `run_sheu_pathway_signatures.py` but with:
  - HUMAN gene symbols (UPPERCASE)
  - Oesinghaus 91-cytokine manifest, subsampled to train donors × 1 tube
    per (donor, cytokine) to keep memory bounded (~910 tubes vs 9100)
  - IFN-cytokine positives auto-detected by substring matching (the dataset
    uses arbitrary cytokine naming conventions)
  - Cascade-negatives = all cytokines not in the IFN positive set

Pre-registered test:
  IFNAR_induced toward IFN-α/β:
    positives = {IFN-α/β/γ/λ cytokines auto-detected}
    negatives = all other cytokines (~85)
    AUC of penetration values, ranked.

Outputs (under --out_dir):
    subset_manifest.json            the subsampled manifest used
    resolved_pathways.json
    penetration_long.parquet
    cytokine_penetration_ranking.csv  ranked list of all 91 cytokines
    ifnar_binary_summary.csv          per-cell-type AUC
    plots/
        ifnar_ranking_bar.pdf
        ifnar_strip_per_cell_type.pdf
        pathway_strip_*.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.eda_pair_benchmark import load_phase1_cells
from cytokine_mil.analysis.pathway_signatures import (
    PATHWAY_SIGNATURES_HUMAN,
    OESINGHAUS_VAL_DONORS_DEFAULT,
    resolve_all_pathways,
    pick_control_genes,
    compute_pathway_score,
    compute_penetration,
    subsample_oesinghaus_manifest,
    match_cytokines_by_patterns,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json",
    )
    parser.add_argument(
        "--out_dir",
        default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_pathway",
    )
    parser.add_argument("--val_donors", nargs="+", default=OESINGHAUS_VAL_DONORS_DEFAULT,
                        help="Donor names to drop from the analysis (default: Donor2, Donor3).")
    parser.add_argument("--tubes_per_pair", type=int, default=1,
                        help="Number of tubes to keep per (donor, cytokine).")
    parser.add_argument("--min_cells", type=int, default=10)
    parser.add_argument("--n_ctrl", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load + subsample manifest
    print(f"[load] full manifest: {args.manifest}", flush=True)
    with open(args.manifest) as f:
        full_manifest = json.load(f)
    print(f"[load] {len(full_manifest)} entries in full manifest", flush=True)

    subset = subsample_oesinghaus_manifest(
        full_manifest,
        val_donors=args.val_donors,
        tubes_per_pair=args.tubes_per_pair,
    )
    print(f"[subset] {len(subset)} entries after dropping val donors "
          f"{args.val_donors} and keeping {args.tubes_per_pair} tube(s) per pair",
          flush=True)
    subset_path = out_dir / "subset_manifest.json"
    with open(subset_path, "w") as f:
        json.dump(subset, f)
    print(f"[subset] wrote subset to {subset_path}", flush=True)

    # 2) Load cells
    print("[load] loading cells from subset manifest ...", flush=True)
    cells_by_pair, gene_names = load_phase1_cells(str(subset_path))
    print(f"[load] {len(cells_by_pair)} (cytokine, cell_type) buckets, "
          f"{len(gene_names)} genes", flush=True)

    # 3) Resolve pathways against HUMAN library
    print("\n[resolve] human pathway genes vs Oesinghaus 4000-HVG panel:", flush=True)
    resolved = resolve_all_pathways(gene_names, signatures=PATHWAY_SIGNATURES_HUMAN)
    for p, r in resolved.items():
        n_curated = len(PATHWAY_SIGNATURES_HUMAN[p]["up"])
        print(f"  {p:18s} ok={r['ok']}  found={len(r['found'])}/{n_curated}  "
              f"missing={r['missing']}", flush=True)
    with open(out_dir / "resolved_pathways.json", "w") as f:
        json.dump(
            {p: {"ok": r["ok"], "found": r["found"], "missing": r["missing"]}
             for p, r in resolved.items()},
            f, indent=2,
        )

    if not any(r["ok"] for r in resolved.values()):
        print("[fail] no pathway resolved — aborting"); return

    # 4) Detect IFN cytokines
    all_cytokines = sorted({k[0] for k in cells_by_pair.keys()})
    print(f"\n[cyt] {len(all_cytokines)} cytokines in subset: {all_cytokines}", flush=True)

    pathway = "IFNAR_induced"
    sig = PATHWAY_SIGNATURES_HUMAN[pathway]
    primary_match = match_cytokines_by_patterns(all_cytokines, sig["primary_patterns"])
    extended_match = match_cytokines_by_patterns(all_cytokines, sig.get("extended_positive_patterns", []))
    positives = sorted(set(primary_match) | set(extended_match))
    negatives = [c for c in all_cytokines if c not in positives and c != "PBS"]

    print(f"\n[ifn] PRIMARY-pattern matches (predicted highest penetration): {primary_match}")
    print(f"[ifn] EXTENDED-positive matches (also IFN family): {extended_match}")
    print(f"[ifn] positives ({len(positives)}): {positives}")
    print(f"[ifn] negatives ({len(negatives)}): first 10 = {negatives[:10]}...")

    if not primary_match:
        print("\n[warn] no PRIMARY IFN-α/β cytokine matched. Trying first extended match as primary.")
        if extended_match:
            primary_match = [extended_match[0]]
            print(f"  fallback primary = {primary_match}")
        else:
            print("[fail] no IFN cytokine of any kind matched — aborting"); return

    primary_stim = primary_match[0]
    print(f"\n[ifn] using primary_stim = '{primary_stim}'", flush=True)

    # 5) Compute IFNAR penetration per cell type, per cytokine
    rng = np.random.default_rng(args.seed)
    idx_map = {p: r["idx"] for p, r in resolved.items() if r["ok"]}
    control_idx = pick_control_genes(args.n_ctrl, idx_map, len(gene_names), rng=rng)
    ifnar_idx = resolved[pathway]["idx"]

    cell_types = sorted({ct for (_, ct) in cells_by_pair.keys()})
    print(f"\n[ct] {len(cell_types)} cell types: {cell_types}", flush=True)

    rows = []
    for T in cell_types:
        if (primary_stim, T) not in cells_by_pair:
            continue
        if ("PBS", T) not in cells_by_pair:
            continue
        cB = cells_by_pair[(primary_stim, T)]
        cP = cells_by_pair[("PBS", T)]
        if len(cB) < args.min_cells or len(cP) < args.min_cells:
            continue
        sB = compute_pathway_score(cB, ifnar_idx, control_idx)
        sP = compute_pathway_score(cP, ifnar_idx, control_idx)
        for A in all_cytokines:
            if (A, T) not in cells_by_pair:
                continue
            cA = cells_by_pair[(A, T)]
            if len(cA) < args.min_cells:
                continue
            sA = compute_pathway_score(cA, ifnar_idx, control_idx)
            pen = compute_penetration(sA, sB, sP)
            rows.append({
                "pathway": pathway,
                "primary_stim": primary_stim,
                "A": A,
                "cell_type": T,
                "n_A": int(len(cA)),
                "n_B": int(len(cB)),
                "n_PBS": int(len(cP)),
                "mean_score_A": float(np.mean(sA)),
                "mean_score_B": float(np.mean(sB)),
                "mean_score_PBS": float(np.mean(sP)),
                "penetration": float(pen),
                "is_positive": (A in positives),
                "is_primary_match": (A in primary_match),
                "is_extended_positive": (A in extended_match),
            })

    pen_df = pd.DataFrame(rows)
    pen_df.to_parquet(out_dir / "penetration_long.parquet", index=False)
    print(f"\n[compute] {len(pen_df)} penetration rows -> penetration_long.parquet", flush=True)

    # 6) Per-cell-type AUC + ranking summary
    auc_rows = []
    for T, sub in pen_df.groupby("cell_type"):
        pos_v = sub[sub["is_positive"]]["penetration"].dropna().values
        neg_v = sub[~sub["is_positive"] & (sub["A"] != "PBS")]["penetration"].dropna().values
        if len(pos_v) == 0 or len(neg_v) == 0:
            continue
        pairs = pos_v[:, None] - neg_v[None, :]
        auc = (np.sign(pairs) + 1).sum() / (2.0 * pairs.size)
        auc_rows.append({
            "cell_type": T,
            "n_pos": int(len(pos_v)),
            "n_neg": int(len(neg_v)),
            "auc": float(auc),
            "mean_pen_pos": float(np.mean(pos_v)),
            "mean_pen_neg": float(np.mean(neg_v)),
            "max_pen_pos": float(np.max(pos_v)),
            "min_pen_pos": float(np.min(pos_v)),
            "max_pen_neg": float(np.max(neg_v)),
            "sep_clean": bool(np.min(pos_v) > np.max(neg_v)),
        })
    auc_df = pd.DataFrame(auc_rows).sort_values("auc", ascending=False)
    auc_df.to_csv(out_dir / "ifnar_binary_summary.csv", index=False)
    print("\n[auc] per-cell-type AUC of IFN-positive vs others:")
    if not auc_df.empty:
        print(auc_df.to_string(index=False))

    # Ranking across all cell types (mean penetration per cytokine)
    ranking = (
        pen_df.groupby("A")
        .agg(mean_penetration=("penetration", "mean"),
             max_penetration=("penetration", "max"),
             n_cell_types=("penetration", "count"),
             is_positive=("is_positive", "first"),
             is_primary_match=("is_primary_match", "first"))
        .reset_index()
        .sort_values("mean_penetration", ascending=False)
    )
    ranking.to_csv(out_dir / "cytokine_penetration_ranking.csv", index=False)
    print("\n[rank] top 20 cytokines by mean IFNAR penetration:")
    print(ranking.head(20).to_string(index=False))
    print("\n[rank] bottom 10:")
    print(ranking.tail(10).to_string(index=False))

    # 7) Plots
    print("\n[plot] generating plots ...", flush=True)

    # 7a) Ranking bar chart
    fig, ax = plt.subplots(figsize=(8, 0.18 * len(ranking) + 2))
    colors = ["#1F77B4" if p else "#999999" for p in ranking["is_positive"].values]
    # Highlight primary IFN-α/β matches in red
    for i, (a, prim) in enumerate(zip(ranking["A"].values, ranking["is_primary_match"].values)):
        if prim:
            colors[i] = "#D62728"
    ax.barh(np.arange(len(ranking)), ranking["mean_penetration"].values, color=colors)
    ax.set_yticks(np.arange(len(ranking)))
    ax.set_yticklabels(ranking["A"].values, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.4, linestyle="--")
    ax.axvline(1, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
    ax.set_xlabel("mean penetration of IFNAR_induced (across cell types)")
    ax.set_title(f"Oesinghaus 24h PBMC: cytokines ranked by IFNAR penetration "
                 f"(toward {primary_stim})")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "ifnar_ranking_bar.pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 7b) Per-cell-type strip plot of penetration positives vs negatives
    fig, axes = plt.subplots(1, len(cell_types), figsize=(3 * len(cell_types) + 2, 4), sharey=True)
    if len(cell_types) == 1:
        axes = [axes]
    for ax, T in zip(axes, cell_types):
        sub_T = pen_df[pen_df["cell_type"] == T]
        pos = sub_T[sub_T["is_positive"]]
        neg = sub_T[~sub_T["is_positive"] & (sub_T["A"] != "PBS")]
        rng2 = np.random.default_rng(0)
        ax.scatter(rng2.normal(0, 0.04, len(pos)), pos["penetration"],
                   color="#D62728", s=24, alpha=0.85, edgecolor="black", linewidth=0.4,
                   label=f"positives (n={len(pos)})")
        ax.scatter(rng2.normal(1, 0.04, len(neg)), neg["penetration"],
                   color="#1F77B4", s=14, alpha=0.55,
                   label=f"negatives (n={len(neg)})")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["IFN+", "other"])
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.axhline(1, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
        ax.set_title(T, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("penetration of IFNAR_induced")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle(f"IFNAR penetration: predicted positives vs all other cytokines  "
                 f"(toward {primary_stim})", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "ifnar_strip_per_cell_type.pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
