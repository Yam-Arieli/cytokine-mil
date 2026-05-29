"""
Post-processor for Probe A: reads ranking_summary.csv produced by
scripts/run_static_ig_probe.py and writes verdict.md.

The IG probe script crashed at its final to_markdown() call because the venv
lacks `tabulate`. The IG computation itself succeeded — static_ig.parquet
and ranking_summary.csv are on disk. This script:

  1. Reads ranking_summary.csv
  2. Re-computes the three pre-registered Phase 0 pass criteria
  3. Writes verdict.md (manual markdown — no tabulate dependency)
  4. Also prints to stdout so the user sees the answer immediately

Usage:
  python scripts/finalize_static_ig_verdict.py \\
      --probe_dir results/gene_dynamics_phase0/static_ig_seed42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--probe_dir", required=True,
        help="Directory containing ranking_summary.csv from run_static_ig_probe.py",
    )
    return p.parse_args()


def _md_table(df: pd.DataFrame) -> str:
    """Manual GitHub-flavored markdown table (no tabulate dep)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                if np.isnan(val):
                    cells.append("NaN")
                else:
                    cells.append(f"{val:.4f}")
            elif isinstance(val, bool):
                cells.append("True" if val else "False")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def main():
    args = _parse_args()
    probe_dir = Path(args.probe_dir)
    csv_path = probe_dir / "ranking_summary.csv"
    if not csv_path.exists():
        print(f"ABORT: {csv_path} not found", flush=True)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    n_markers = len(df)

    n_correct = int(df["correct_in_top3"].sum())
    median_gap = float(df["magnitude_gap"].median(skipna=True))
    median_rho = float(df["rho_ig_vs_expression"].median(skipna=True))

    n_pass_target = max(1, int(round(0.70 * n_markers)))

    crit_top3 = n_correct >= n_pass_target
    crit_gap = (not np.isnan(median_gap)) and (median_gap >= 0.20)
    crit_rho = (not np.isnan(median_rho)) and (median_rho < 0.70)
    overall = "PASS" if (crit_top3 and crit_gap and crit_rho) else "FAIL"

    # --- Console summary ---
    print()
    print("=" * 70)
    print(f"Probe A — Static-IG routing sanity check ({probe_dir.name})")
    print("=" * 70)
    print(f"Markers evaluated: {n_markers}")
    print(f"  (1) top-3 hit rate:    "
          f"{n_correct} / {n_markers}  (need ≥ {n_pass_target})  "
          f"{'PASS' if crit_top3 else 'FAIL'}")
    print(f"  (2) median gap:        "
          f"{median_gap:+.3f}             (need ≥ 0.20)            "
          f"{'PASS' if crit_gap else 'FAIL'}")
    print(f"  (3) median rho(IG, X): "
          f"{median_rho:+.3f}             (need < 0.70)            "
          f"{'PASS' if crit_rho else 'FAIL'}")
    print("-" * 70)
    print(f"OVERALL: {overall}")
    print()

    # Per-marker brief
    print("Per-marker top-3 (sorted by marker_gene):")
    print()
    brief_cols = ["marker_gene", "pathway", "winner", "top3", "correct_in_top3",
                  "magnitude_gap", "rho_ig_vs_expression"]
    brief = df.sort_values("marker_gene")[brief_cols]
    for _, row in brief.iterrows():
        flag = "✓" if row["correct_in_top3"] else "✗"
        print(
            f"  {flag} {row['marker_gene']:<8s} "
            f"({row['pathway']:<20s})  "
            f"top3=[{row['top3']}]  "
            f"gap={row['magnitude_gap']:+.3f}  "
            f"rho={row['rho_ig_vs_expression']:+.3f}"
        )

    # --- verdict.md ---
    lines = [
        "# Probe A — Static-IG routing sanity check",
        "",
        f"**Probe dir:** `{probe_dir}`",
        "",
        f"## Overall verdict: **{overall}**",
        "",
        "## Pre-registered pass criteria",
        "",
        "| Criterion | Threshold | Observed | Pass? |",
        "|---|---|---|---|",
        f"| Markers with biological inducer in top-3 | "
        f"≥ {n_pass_target} / {n_markers} | "
        f"{n_correct} / {n_markers} | "
        f"{'PASS' if crit_top3 else 'FAIL'} |",
        f"| Median magnitude gap (winner − runner-up) / |winner| | ≥ 0.20 | "
        f"{median_gap:+.3f} | {'PASS' if crit_gap else 'FAIL'} |",
        f"| Median Pearson ρ(IG, mean expression) across markers | < 0.70 | "
        f"{median_rho:+.3f} | {'PASS' if crit_rho else 'FAIL'} |",
        "",
        "## Per-marker results",
        "",
        _md_table(df.sort_values(["pathway", "marker_gene"])),
        "",
        "## Notes",
        "",
        "- Generated post-hoc from `ranking_summary.csv`. The original "
        "`run_static_ig_probe.py` crashed at its final `to_markdown()` call "
        "(missing `tabulate` in venv); the IG computation itself completed "
        "for all cytokines.",
        "- IG baseline = per-gene mean expression across PBS tubes.",
        "- Pooled across cell types (Option B). Per-cell IG averaged within a "
        "tube; per-tube IG averaged across X-tubes (max 10 tubes/cytokine).",
        "- MX1 and MX2 were dropped at load time (not in the HVG list — my "
        "earlier substring grep had false positives).",
    ]

    verdict_path = probe_dir / "verdict.md"
    verdict_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {verdict_path}")


if __name__ == "__main__":
    main()
