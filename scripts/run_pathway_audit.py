#!/usr/bin/env python3
"""
Adversarial audit of the pathway-signature cascade methodology (§23).

Runs four independent validations on Sheu 3hr BMDM data:
  (1) Cytokine-label permutation null for the binary IFNAR test
  (2) Random-pathway null (curated genes vs random gene sets of same size)
  (3) Per-donor (not per-cell) inference for the NF-κB magnitude tests
  (4) Directional asymmetry test for the cascade pairs

The goal is to detect: cherry-picking, overfit p-values from pseudo-replication,
tautological gene-list matching, and the conflation of pathway-engagement with
cascade-direction.

Outputs (under --out_dir):
    audit_summary.md                human-readable peer-review-style report
    audit_1_permutation_null.csv    per-cell-type AUC + null distribution stats
    audit_2_random_pathway_null.csv per-cell-type random-pathway AUC stats
    audit_3_per_donor_magnitude.csv donor-level Wilcoxon signed-rank tests
    audit_4_directional_asymmetry.csv  directional cascade evidence per pair
    plots/
        audit1_null_distribution.pdf
        audit2_random_vs_curated.pdf
        audit3_per_donor_means.pdf
        audit4_directional_quadrants.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
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
    PATHWAY_SIGNATURES,
    IFNAR_POSITIVE_STIMULI,
    IFNAR_NEGATIVE_STIMULI,
    resolve_all_pathways,
    compute_all_penetrations,
)
from cytokine_mil.analysis.pathway_audit import (
    load_cells_by_donor,
    permutation_null_binary_auc,
    random_pathway_null,
    per_donor_magnitude_test,
    directional_asymmetry_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json",
    )
    parser.add_argument(
        "--out_dir",
        default="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_pathway_audit",
    )
    parser.add_argument("--min_cells", type=int, default=10)
    parser.add_argument("--n_permutations", type=int, default=2000)
    parser.add_argument("--n_random_pathways", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Setup: load cells, compute penetrations (same as §23 main run)
    # =====================================================================
    print(f"[load] manifest = {args.manifest}", flush=True)
    cells_by_pair, gene_names = load_phase1_cells(args.manifest)
    print(f"[load] {len(cells_by_pair)} buckets, {len(gene_names)} genes", flush=True)

    resolved = resolve_all_pathways(gene_names)
    print("[resolve] pathways:")
    for p, r in resolved.items():
        print(f"  {p:18s} ok={r['ok']}  found={len(r['found'])}")

    print("[setup] computing penetration table ...", flush=True)
    penetration_df, _ = compute_all_penetrations(
        cells_by_pair, gene_names, n_ctrl=20,
        min_cells=args.min_cells, seed=args.seed,
    )
    penetration_df.to_parquet(out_dir / "penetration_long.parquet", index=False)
    print(f"[setup] {len(penetration_df)} penetration rows", flush=True)

    # =====================================================================
    # Audit 1: permutation null on binary IFNAR test
    # =====================================================================
    print("\n=== AUDIT 1: permutation null on cytokine labels (binary IFNAR test) ===")
    aud1 = permutation_null_binary_auc(
        penetration_df,
        pos_stims=IFNAR_POSITIVE_STIMULI,
        neg_stims=IFNAR_NEGATIVE_STIMULI,
        primary_stim="IFNb",
        pathway="IFNAR_induced",
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    aud1["observed"].to_csv(out_dir / "audit_1_permutation_null.csv", index=False)
    print(aud1["observed"].to_string(index=False))

    # Plot null distributions per cell type
    if not aud1["null_summary"].empty:
        cell_types = sorted(aud1["null_summary"]["cell_type"].unique())
        n = len(cell_types)
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.2), sharey=True)
        if n == 1:
            axes = [axes]
        obs_by_ct = dict(zip(aud1["observed"]["cell_type"],
                              aud1["observed"]["observed_auc"]))
        for ax, T in zip(axes, cell_types):
            null_T = aud1["null_summary"][aud1["null_summary"]["cell_type"] == T]["auc"].values
            ax.hist(null_T, bins=30, color="#999999", alpha=0.7,
                    label=f"null (n={len(null_T)})")
            obs = obs_by_ct.get(T, np.nan)
            if not np.isnan(obs):
                ax.axvline(obs, color="#D62728", linewidth=2.0,
                           label=f"observed AUC={obs:.3f}")
                # p-value annotation
                p = float((null_T >= obs).mean())
                ax.text(0.02, 0.95, f"p_emp = {p:.4f}",
                        transform=ax.transAxes, fontsize=9, va="top")
            ax.set_xlabel("AUC")
            ax.set_title(T, fontsize=10)
            ax.legend(loc="upper right", fontsize=7)
        axes[0].set_ylabel("count")
        fig.suptitle("Audit 1: cytokine-label permutation null vs observed AUC", fontsize=11)
        plt.tight_layout()
        plt.savefig(str(plots_dir / "audit1_null_distribution.pdf"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    # =====================================================================
    # Audit 2: random-pathway null
    # =====================================================================
    print("\n=== AUDIT 2: random-pathway null (curated vs random gene sets) ===")
    # Excluded indices: ALL genes used by any curated pathway
    all_curated_idx = []
    for p, r in resolved.items():
        if r["ok"]:
            all_curated_idx.extend(r["idx"].tolist())
    excluded_idx = set(all_curated_idx)

    ifnar_idx = resolved["IFNAR_induced"]["idx"]
    aud2 = random_pathway_null(
        cells_by_pair, gene_names, ifnar_idx,
        pos_stims=IFNAR_POSITIVE_STIMULI,
        neg_stims=IFNAR_NEGATIVE_STIMULI,
        primary_stim="IFNb",
        excluded_gene_indices=excluded_idx,
        n_random_pathways=args.n_random_pathways,
        seed=args.seed,
    )
    aud2["summary"].to_csv(out_dir / "audit_2_random_pathway_null.csv", index=False)
    print(aud2["summary"].to_string(index=False))

    # Plot: random-pathway AUC distribution vs curated observed AUC per cell type
    if not aud2["random_aucs"].empty:
        cell_types = sorted(aud2["random_aucs"]["cell_type"].unique())
        n = len(cell_types)
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.2), sharey=True)
        if n == 1:
            axes = [axes]
        obs_by_ct = dict(zip(aud1["observed"]["cell_type"],
                              aud1["observed"]["observed_auc"]))
        for ax, T in zip(axes, cell_types):
            null_T = aud2["random_aucs"][aud2["random_aucs"]["cell_type"] == T]["auc"].values
            ax.hist(null_T, bins=30, color="#1F77B4", alpha=0.7,
                    label=f"random pathways (n={len(null_T)})")
            obs = obs_by_ct.get(T, np.nan)
            if not np.isnan(obs):
                ax.axvline(obs, color="#D62728", linewidth=2.0,
                           label=f"curated AUC={obs:.3f}")
                p = float((null_T >= obs).mean())
                ax.text(0.02, 0.95, f"p_random ≤ obs = {p:.4f}",
                        transform=ax.transAxes, fontsize=9, va="top")
            ax.set_xlabel("AUC")
            ax.set_title(T, fontsize=10)
            ax.legend(loc="upper right", fontsize=7)
        axes[0].set_ylabel("count")
        fig.suptitle("Audit 2: random gene-set AUC vs curated IFNAR_induced", fontsize=11)
        plt.tight_layout()
        plt.savefig(str(plots_dir / "audit2_random_vs_curated.pdf"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    # =====================================================================
    # Audit 3: per-donor inference
    # =====================================================================
    print("\n=== AUDIT 3: per-donor Wilcoxon signed-rank inference ===")
    print("[load] loading cells with per-donor stratification ...", flush=True)
    cells_by_triple, _ = load_cells_by_donor(args.manifest)
    print(f"[load] {len(cells_by_triple)} (cyt, ct, donor) triples", flush=True)

    aud3_rows = []
    cascade_pairs = [
        ("NFkB_canonical", "LPS",   "TNF"),
        ("NFkB_canonical", "LPSlo", "TNF"),
        ("NFkB_canonical", "P3CSK", "TNF"),
        ("NFkB_canonical", "CpG",   "TNF"),
    ]
    for pathway_name, A, B in cascade_pairs:
        if pathway_name not in resolved or not resolved[pathway_name]["ok"]:
            continue
        df = per_donor_magnitude_test(
            cells_by_triple, resolved[pathway_name]["idx"], A, B,
            min_cells=args.min_cells,
        )
        df["pathway"] = pathway_name
        aud3_rows.append(df)

    if aud3_rows:
        aud3 = pd.concat(aud3_rows, ignore_index=True)
        aud3.to_csv(out_dir / "audit_3_per_donor_magnitude.csv", index=False)
        print(aud3[["pathway", "A_upstream", "B_downstream", "cell_type",
                    "n_donors_paired_AB", "mean_A", "mean_B", "mean_PBS",
                    "wilcoxon_p_A_gt_B", "sign_p_A_gt_B",
                    "wilcoxon_p_B_gt_PBS", "pass_ordering"]].to_string(index=False))

        # Plot: per-cell-type bar of mean_A vs mean_B (means + paired donor diffs)
        cell_types = sorted(aud3["cell_type"].unique())
        n = len(cell_types)
        if n > 0:
            fig, axes = plt.subplots(
                1, n, figsize=(3.5 * n + 1, 4.0), sharey=True,
            )
            if n == 1:
                axes = [axes]
            for ax, T in zip(axes, cell_types):
                sub_T = aud3[aud3["cell_type"] == T]
                xs = np.arange(len(sub_T))
                ax.bar(xs - 0.2, sub_T["mean_A"].values, width=0.4,
                       label="A_upstream", color="#D62728")
                ax.bar(xs + 0.2, sub_T["mean_B"].values, width=0.4,
                       label="TNF (B)", color="#1F77B4")
                ax.set_xticks(xs)
                ax.set_xticklabels(sub_T["A_upstream"].values, rotation=45, ha="right")
                ax.set_title(T, fontsize=10)
                if ax is axes[0]:
                    ax.set_ylabel("mean s_NFkB (donor-averaged)")
                ax.legend(fontsize=7)
            fig.suptitle("Audit 3: donor-level mean s_NFkB per A vs TNF", fontsize=11)
            plt.tight_layout()
            plt.savefig(str(plots_dir / "audit3_per_donor_means.pdf"),
                        dpi=200, bbox_inches="tight")
            plt.close(fig)

    # =====================================================================
    # Audit 4: directional asymmetry test
    # =====================================================================
    print("\n=== AUDIT 4: directional asymmetry — true cascade direction test ===")
    aud4_rows = []
    pathway_idx_map = {p: resolved[p]["idx"] for p in resolved if resolved[p]["ok"]}
    directional_pairs = [
        # (A, B, P_A, P_B) — A is the predicted UPSTREAM stim in cascade A→B
        ("PIC", "IFNb", "IRF3_direct", "IFNAR_induced"),
        ("LPS", "IFNb", "IRF3_direct", "IFNAR_induced"),
        # For LPS→TNF, A engages NFkB directly + drives autocrine TNF.
        # We use TNFR_autocrine as the "downstream" signature.
        ("LPS",   "TNF", "NFkB_canonical", "TNFR_autocrine"),
        ("LPSlo", "TNF", "NFkB_canonical", "TNFR_autocrine"),
        ("P3CSK", "TNF", "NFkB_canonical", "TNFR_autocrine"),
        ("CpG",   "TNF", "NFkB_canonical", "TNFR_autocrine"),
    ]
    for A, B, P_A, P_B in directional_pairs:
        if P_A not in pathway_idx_map or P_B not in pathway_idx_map:
            print(f"  skipping ({A}, {B}): pathway not resolved")
            continue
        df = directional_asymmetry_test(
            cells_by_pair, pathway_idx_map, A, B, P_A, P_B,
            min_cells=args.min_cells,
        )
        if not df.empty:
            aud4_rows.append(df)

    if aud4_rows:
        aud4 = pd.concat(aud4_rows, ignore_index=True)
        aud4.to_csv(out_dir / "audit_4_directional_asymmetry.csv", index=False)
        print(aud4[["A", "B", "P_A", "P_B", "cell_type",
                    "sA_PA_norm", "sB_PA_norm", "sA_PB_norm", "sB_PB_norm",
                    "asym_PA", "asym_PB", "directional_score",
                    "interpretation"]].to_string(index=False))

        # Quadrant plot: x = asym_PA (A engages P_A more than B?),
        #                y = asym_PB (A engages P_B more than B?)
        # True cascade A→B should sit in the upper-right (asym_PA > 0, asym_PB ≈ 0)
        fig, ax = plt.subplots(figsize=(7, 6))
        for (A, B, P_A, P_B), sub in aud4.groupby(["A", "B", "P_A", "P_B"]):
            label = f"{A}→{B}"
            ax.scatter(sub["asym_PA"], sub["asym_PB"], s=110,
                       alpha=0.85, edgecolor="black", linewidth=0.6,
                       label=label)
            for _, r in sub.iterrows():
                ax.annotate(r["cell_type"], (r["asym_PA"], r["asym_PB"]),
                            fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.4, linestyle="--")
        # Diagonal: asym_PA = asym_PB (symmetric — A and B differ equally on both pathways)
        lim = float(max(abs(aud4[["asym_PA", "asym_PB"]].values).max(), 1.0))
        ax.plot([-lim, lim], [-lim, lim], color="black", linewidth=0.3,
                linestyle=":", label="diagonal (no asymmetry)")
        ax.set_xlim(-lim * 1.1, lim * 1.1)
        ax.set_ylim(-lim * 1.1, lim * 1.1)
        ax.set_xlabel("asym_PA: (s_A − s_B) on the UPSTREAM-specific pathway P_A")
        ax.set_ylabel("asym_PB: (s_A − s_B) on the DOWNSTREAM pathway P_B")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_title("Audit 4: directional asymmetry — true cascade A→B sits"
                     "\nin upper-right (positive asym_PA, ≈0 asym_PB)", fontsize=10)
        plt.tight_layout()
        plt.savefig(str(plots_dir / "audit4_directional_quadrants.pdf"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    # =====================================================================
    # Write peer-review-style summary
    # =====================================================================
    write_audit_summary(out_dir, aud1, aud2, aud3 if aud3_rows else None,
                         aud4 if aud4_rows else None)
    print(f"\n[done] outputs in {out_dir}")


def write_audit_summary(out_dir, aud1, aud2, aud3, aud4):
    """Generate a markdown summary report."""
    lines = []
    lines.append("# Pathway-Signature Cascade Methodology: Adversarial Audit")
    lines.append("")
    lines.append("Four independent validations of the §23 cascade-direction methodology.")
    lines.append("Each is designed to falsify a specific concern raised in peer review.")
    lines.append("")

    # Audit 1
    lines.append("## Audit 1: Cytokine-label permutation null on binary IFNAR test")
    lines.append("")
    lines.append("**Concern addressed:** Is the observed AUC structurally inflated by"
                 " small-N selection effects, or is it distinguishable from null?")
    lines.append("")
    if aud1["observed"].empty:
        lines.append("_(no data)_")
    else:
        lines.append("Per cell type:")
        lines.append("")
        lines.append("| cell_type | observed AUC | null mean | null Q95 | p_emp |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in aud1["observed"].iterrows():
            lines.append(f"| {r['cell_type']} | {r['observed_auc']:.3f} | "
                         f"{r['null_mean']:.3f} | {r['null_q95']:.3f} | "
                         f"{r['p_emp']:.4f} |")
        n_pass = int((aud1["observed"]["p_emp"] < 0.05).sum())
        lines.append("")
        lines.append(f"**{n_pass} of {len(aud1['observed'])} cell types have "
                     f"observed AUC above the permutation null at p<0.05.**")
    lines.append("")

    # Audit 2
    lines.append("## Audit 2: Random-pathway null (curated genes vs random gene sets)")
    lines.append("")
    lines.append("**Concern addressed:** Is the curated gene list carrying specific"
                 " information, or do random gene sets of the same size give equally"
                 " high AUCs (which would mean we are reading overall activation,"
                 " not pathway specificity)?")
    lines.append("")
    if aud2 is None or aud2["summary"].empty:
        lines.append("_(no data)_")
    else:
        lines.append("Per cell type, the AUC distribution from 200 random gene sets"
                     " of the same size as the curated IFNAR_induced signature:")
        lines.append("")
        lines.append("| cell_type | random mean | random Q95 | random Q99 |")
        lines.append("|---|---:|---:|---:|")
        for _, r in aud2["summary"].iterrows():
            lines.append(f"| {r['cell_type']} | {r['mean']:.3f} | "
                         f"{r['q95']:.3f} | {r['q99']:.3f} |")
        lines.append("")
        lines.append("Compare to observed curated AUC in `audit_1_permutation_null.csv`."
                     " If curated AUC > random Q95, the gene-list specificity matters.")
    lines.append("")

    # Audit 3
    lines.append("## Audit 3: Per-donor (not per-cell) magnitude tests")
    lines.append("")
    lines.append("**Concern addressed:** The original Mann-Whitney p-values"
                 " (`p ≈ 10⁻¹³¹`) were per-cell, with hundreds of cells per tube"
                 " sharing donors. These are fiction. The unit of replication is"
                 " the donor (~3-4 per condition). Honest p-values via Wilcoxon"
                 " signed-rank on donor-paired means.")
    lines.append("")
    if aud3 is None or aud3.empty:
        lines.append("_(no data)_")
    else:
        lines.append("| pathway | A | cell_type | n_donors | mean_A | mean_TNF | mean_PBS | Wilcoxon p (A>TNF) | sign-test p | order ok |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for _, r in aud3.iterrows():
            lines.append(
                f"| {r['pathway']} | {r['A_upstream']} | {r['cell_type']} | "
                f"{r['n_donors_paired_AB']} | {r['mean_A']:.3f} | "
                f"{r['mean_B']:.3f} | {r['mean_PBS']:.3f} | "
                f"{r['wilcoxon_p_A_gt_B']:.4f} | {r['sign_p_A_gt_B']:.4f} | "
                f"{r['pass_ordering']} |"
            )
        n_pass = int((aud3["pass_ordering"] & (aud3["wilcoxon_p_A_gt_B"] < 0.05)).sum())
        lines.append("")
        lines.append(f"**{n_pass} of {len(aud3)} (pathway × cell_type × pair) tests pass "
                     f"donor-level α=0.05 with the predicted ordering.**")
        lines.append("")
        lines.append("With only ~3 donors per (A, B) pair, the minimum achievable"
                     " Wilcoxon signed-rank p is ~0.125. So 'pass' here means: the"
                     " donor-level means consistently follow the predicted ordering"
                     " (which is the substantive claim), even if formal significance"
                     " is unreachable at this donor count.")
    lines.append("")

    # Audit 4
    lines.append("## Audit 4: Directional asymmetry (the test originally skipped)")
    lines.append("")
    lines.append("**Concern addressed:** High penetration of pathway P by stimulus A"
                 " proves A engages P. It does NOT prove A → B cascade through P_B."
                 " A genuine cascade A→B requires: A engages P_A (A's own pathway)"
                 " AND A engages P_B (B's pathway, via autocrine cascade) AND"
                 " B engages P_B but NOT P_A. This audit computes the four quantities"
                 " and the directional asymmetry score.")
    lines.append("")
    lines.append("**Predicted pattern for true cascade A→B:**")
    lines.append("- `asym_PA` (= s_A_norm − s_B_norm on P_A) **positive** (A engages P_A; B doesn't)")
    lines.append("- `asym_PB` (= s_A_norm − s_B_norm on P_B) **≈ 0** (both engage P_B)")
    lines.append("- `directional_score` = asym_PA − asym_PB → **positive** for true A→B")
    lines.append("")
    if aud4 is None or aud4.empty:
        lines.append("_(no data)_")
    else:
        lines.append("| A | B | P_A | P_B | cell_type | asym_PA | asym_PB | directional_score | interpretation |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---|")
        for _, r in aud4.iterrows():
            lines.append(
                f"| {r['A']} | {r['B']} | {r['P_A']} | {r['P_B']} | {r['cell_type']} | "
                f"{r['asym_PA']:.3f} | {r['asym_PB']:.3f} | "
                f"{r['directional_score']:.3f} | {r['interpretation']} |"
            )
        n_positive_score = int((aud4["directional_score"] > 0).sum())
        lines.append("")
        lines.append(f"**{n_positive_score} of {len(aud4)} (pair, cell_type) "
                     f"observations show positive directional_score (consistent with "
                     f"A→B cascade direction).**")
    lines.append("")

    # Honest overall reading
    lines.append("---")
    lines.append("")
    lines.append("## Honest reading")
    lines.append("")
    lines.append("All four audits are diagnostics, not validations. Specifically:")
    lines.append("")
    lines.append("- Audit 1: tells us whether random label assignment can reproduce"
                 " the observed AUC. Strong if p_emp < 0.01.")
    lines.append("- Audit 2: tells us whether the curated gene list is doing work,"
                 " or whether any random gene set would yield similar discrimination.")
    lines.append("- Audit 3: replaces the fictional per-cell p-values with honest"
                 " donor-level statistics. With only ~3 donors per condition this is"
                 " low-powered but rigorous.")
    lines.append("- Audit 4: tests whether the data shows directional cascade"
                 " evidence (vs pure pathway engagement). Positive directional_score"
                 " is necessary but not sufficient evidence for cascade direction;"
                 " interventional follow-up (e.g., IFNAR-KO) would close the loop.")
    lines.append("")
    lines.append("Combining the four: if Audit 1 and Audit 2 both reject null at"
                 " p<0.01 AND Audit 3 shows consistent donor-level ordering AND"
                 " Audit 4 shows positive directional_score, the result is robust"
                 " against the most likely peer-review concerns. If any one fails,"
                 " that's the specific weakness to address before publication.")

    (out_dir / "audit_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
