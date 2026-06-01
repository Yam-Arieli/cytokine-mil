"""
§25 cascade-direction sweep on the Immune Dictionary (Cui Nature 2024).

Runs `directional_asymmetry_test` (§24 methodology, unchanged) on the 10
pre-registered cascades from `IMMUNE_DICTIONARY_PREREGISTERED_CASCADES`,
per-mouse, and applies the GREEN / AMBER / RED verdict criteria locked
in `reports/immune_dictionary/PRE_REGISTRATION.md`.

Outputs (under --out_dir, default `results/immune_dictionary_pathway/`):
  - cascade_results.parquet      long-form per (cascade, mouse, cell_type)
  - per_cascade_summary.csv      one row per cascade with observed outcome
  - verdict.json                 GREEN / AMBER / RED + per-category counts
  - pathway_overlap_matrix.csv   confirmatory snapshot at run time
  - resolved_pathways.json       which curated genes resolved against panel
  - plots/cascade_<A>_<B>.pdf    per-cascade strip plot

Usage (cluster):
    /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \\
        scripts/run_immune_dictionary_pathway_audit.py \\
        --manifest /cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cytokine_mil.analysis.eda_pair_benchmark import load_phase1_cells
from cytokine_mil.analysis.pathway_audit import directional_asymmetry_test
from cytokine_mil.analysis.pathway_signatures import (
    IMMUNE_DICTIONARY_PREREGISTERED_CASCADES,
    PATHWAY_SIGNATURES,
    compute_pathway_overlap_matrix,
    resolve_all_pathways,
)


def _resolve_observed_outcome(
    sub: pd.DataFrame, predicted: str, must_pass_threshold: float, must_fail_threshold: float
) -> str:
    """Apply the §25.3 verdict criteria to one cascade's observations."""
    if sub.empty:
        return "MISSING"
    if predicted == "MUST_PASS":
        # PASS if any cell-type observation has directional_score > threshold (default +1.0)
        return "PASS" if (sub["directional_score"] > must_pass_threshold).any() else "FAIL"
    if predicted == "MUST_FAIL":
        # PASS if all observations have |directional_score| < threshold (default 0.5)
        return "PASS" if sub["directional_score"].abs().max() < must_fail_threshold else "FAIL"
    # NEG_CONTROL
    return "PASS" if sub["directional_score"].mean() <= 0 else "FAIL"


def _compose_verdict(summary: pd.DataFrame) -> dict:
    """§25.3 GREEN / AMBER / RED."""
    must_pass = summary[summary["predicted_outcome"] == "MUST_PASS"]
    must_fail = summary[summary["predicted_outcome"] == "MUST_FAIL"]
    neg_ctrl = summary[summary["predicted_outcome"] == "NEG_CONTROL"]

    mp_pass = int((must_pass["observed_outcome"] == "PASS").sum())
    mf_pass = int((must_fail["observed_outcome"] == "PASS").sum())
    nc_pass = int((neg_ctrl["observed_outcome"] == "PASS").sum())

    mp_total = int(len(must_pass))
    mf_total = int(len(must_fail))
    nc_total = int(len(neg_ctrl))

    if mp_pass >= 4 and mf_pass >= 2 and nc_pass == 2:
        verdict = "GREEN"
    elif mp_pass >= 2 or mf_pass >= 1:
        verdict = "AMBER"
    else:
        verdict = "RED"

    return {
        "verdict": verdict,
        "must_pass": f"{mp_pass}/{mp_total}",
        "must_fail": f"{mf_pass}/{mf_total}",
        "neg_control": f"{nc_pass}/{nc_total}",
    }


def _make_plot(out_path: Path, sub: pd.DataFrame, A: str, B: str, predicted: str, observed: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axhline(1.0, color="green", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(-0.5, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(0.5, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    cell_types = sorted(sub["cell_type"].unique())
    ct_to_x = {ct: i for i, ct in enumerate(cell_types)}
    mice = sorted(sub["mouse"].unique())
    cmap = plt.get_cmap("tab10")
    for i, mouse in enumerate(mice):
        m = sub[sub["mouse"] == mouse]
        xs = [ct_to_x[ct] + 0.05 * (i - (len(mice) - 1) / 2) for ct in m["cell_type"]]
        ax.scatter(xs, m["directional_score"], label=mouse, alpha=0.85, color=cmap(i))
    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(cell_types, rotation=45, ha="right")
    ax.set_xlabel("Cell type")
    ax.set_ylabel("directional_score = asym_PA − asym_PB")
    ax.set_title(f"{A} → {B}  (predicted: {predicted}, observed: {observed})")
    ax.legend(title="mouse", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="§25 cascade sweep on Immune Dictionary.")
    ap.add_argument("--manifest", required=True, help="Path to ID pseudotubes manifest.json")
    ap.add_argument("--out_dir", default="results/immune_dictionary_pathway")
    ap.add_argument("--min_cells", type=int, default=10,
                    help="Minimum cells per (cytokine, cell_type, mouse) triple required to score a cascade")
    ap.add_argument("--must_pass_threshold", type=float, default=1.0,
                    help="A cell-type directional_score above this counts as MUST-PASS evidence")
    ap.add_argument("--must_fail_threshold", type=float, default=0.5,
                    help="|directional_score| below this counts as MUST-FAIL overlap-failure evidence")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print(f"=== §25 Immune Dictionary cascade sweep ===")
    print(f"manifest: {args.manifest}")
    print(f"out_dir:  {out_dir}")
    print(f"min_cells: {args.min_cells}")
    print(f"thresholds: MUST_PASS > {args.must_pass_threshold}, MUST_FAIL < {args.must_fail_threshold}")
    print()

    # Discover mice from manifest
    with open(args.manifest) as f:
        entries = json.load(f)
    mice = sorted({e["donor"] for e in entries})
    cytokines_present = sorted({e["cytokine"] for e in entries})
    print(f"manifest has {len(entries)} entries, {len(mice)} mice: {mice}")
    print(f"cytokines: {cytokines_present[:10]}{'...' if len(cytokines_present) > 10 else ''}")
    print()

    # Probe one mouse to get gene names
    print("[load] probing gene panel from first mouse...")
    probe_cells, gene_names = load_phase1_cells(args.manifest, donors=[mice[0]])
    print(f"  {len(gene_names)} genes; {len(probe_cells)} (cytokine, cell_type) keys for {mice[0]}")
    print()

    # Resolve all curated pathways against the panel
    resolved = resolve_all_pathways(gene_names, min_hits=3)
    pathway_idx_map = {p: r["idx"] for p, r in resolved.items() if r["ok"]}
    with open(out_dir / "resolved_pathways.json", "w") as f:
        json.dump(
            {p: {"n_found": len(r["found"]), "found": r["found"],
                 "missing": r["missing"], "ok": bool(r["ok"])}
             for p, r in resolved.items()},
            f, indent=2,
        )
    print("[resolve] curated pathway gene coverage:")
    for p, r in resolved.items():
        flag = "OK" if r["ok"] else "SKIP"
        total = len(r["found"]) + len(r["missing"])
        print(f"  [{flag}] {p}: {len(r['found'])}/{total} genes resolved")
    print()

    # Pre-registered cascades (no post-hoc edits)
    cascades = IMMUNE_DICTIONARY_PREREGISTERED_CASCADES
    print(f"[cascades] {len(cascades)} pre-registered:")
    for A, B, PA, PB, outcome in cascades:
        print(f"  {outcome:<12} {A:>6} -> {B:<6}  P_A={PA}, P_B={PB}")
    print()

    # Per-mouse, per-cascade asymmetry test
    all_rows = []
    for mouse in mice:
        print(f"[run] {mouse} ...", flush=True)
        cells_by_pair, _ = load_phase1_cells(args.manifest, donors=[mouse])
        stims_present = {s for s, _ in cells_by_pair.keys()}
        for A, B, P_A, P_B, outcome in cascades:
            if P_A not in pathway_idx_map or P_B not in pathway_idx_map:
                continue
            if A not in stims_present or B not in stims_present or "PBS" not in stims_present:
                continue
            df = directional_asymmetry_test(
                cells_by_pair, pathway_idx_map, A, B, P_A, P_B,
                min_cells=args.min_cells,
            )
            if df.empty:
                continue
            df["mouse"] = mouse
            df["predicted_outcome"] = outcome
            all_rows.append(df)

    if not all_rows:
        print("ERROR: no cascades produced rows. Check manifest, stimulus names, and pathway resolution.", file=sys.stderr)
        sys.exit(1)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_parquet(out_dir / "cascade_results.parquet")
    print(f"[write] {len(full)} rows -> cascade_results.parquet")
    print()

    # Aggregate per cascade
    summary_rows = []
    for A, B, P_A, P_B, predicted in cascades:
        sub = full[(full["A"] == A) & (full["B"] == B)]
        n_obs = int(len(sub))
        mean_score = float(sub["directional_score"].mean()) if n_obs else float("nan")
        max_abs = float(sub["directional_score"].abs().max()) if n_obs else float("nan")
        observed = _resolve_observed_outcome(
            sub, predicted,
            must_pass_threshold=args.must_pass_threshold,
            must_fail_threshold=args.must_fail_threshold,
        )
        summary_rows.append({
            "A": A, "B": B, "P_A": P_A, "P_B": P_B,
            "predicted_outcome": predicted,
            "observed_outcome": observed,
            "n_observations": n_obs,
            "mean_directional_score": mean_score,
            "max_abs_directional_score": max_abs,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "per_cascade_summary.csv", index=False)
    print("=== Per-cascade summary ===")
    print(summary.to_string(index=False))
    print()

    # GREEN / AMBER / RED verdict
    verdict = _compose_verdict(summary)
    verdict["n_mice"] = len(mice)
    verdict["mice"] = mice
    verdict["n_cascades_run"] = int((summary["n_observations"] > 0).sum())
    verdict["n_total_observations"] = int(full.shape[0])
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("=== §25.3 VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print()

    # Confirmatory pathway overlap matrix (must match PRE_REGISTRATION.md)
    overlap = compute_pathway_overlap_matrix()
    overlap.to_csv(out_dir / "pathway_overlap_matrix.csv")

    # Per-cascade plots
    print("[plot] writing per-cascade strip plots...")
    for A, B, P_A, P_B, predicted in cascades:
        sub = full[(full["A"] == A) & (full["B"] == B)]
        if sub.empty:
            continue
        observed = summary[(summary["A"] == A) & (summary["B"] == B)]["observed_outcome"].iloc[0]
        out_path = out_dir / "plots" / f"cascade_{A.replace('-', '_')}_to_{B.replace('-', '_')}.pdf"
        _make_plot(out_path, sub, A, B, predicted, observed)

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
