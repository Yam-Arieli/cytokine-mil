"""
Cross-time-point comparison — the BOTTOM LINE of the Sheu single-frame
cascade-direction experiment.

For each time point T (each fully self-contained — no cross-time leakage in the
method), the Path B driver produced `per_celltype.csv` with a `cross_asym`
column. This script reads them, joins the crystal-clear Sheu labels, and asks:

  * Directional accuracy by class (IFN_MUST, IFN_SHOULD, NFKB_SHOULD) vs time
  * Negative-pair |cross_asym| (specificity) vs time
  * The per-pair cross_asym kinetic curve for the MUST IFN cascades
  * Which single time frame best supports single-frame cascade detection

Output: reports/sheu_cascade/timepoint_comparison.md

Imports: argparse, csv, sys, pathlib, numpy, pandas. No banned deps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = REPO / "reports" / "sheu_cascade" / "sheu_axes_labeled.csv"
DEFAULT_OUT = REPO / "reports" / "sheu_cascade" / "timepoint_comparison.md"


def _median_sign(df_axis: pd.DataFrame, col: str = "cross_asym") -> float:
    return float(np.median(df_axis[col].to_numpy()))


def load_timepoint(per_celltype_csv: Path) -> pd.DataFrame:
    """Per-axis median cross_asym for one time point."""
    df = pd.read_csv(per_celltype_csv)
    if "cross_asym" not in df.columns:
        df["cross_asym"] = df["sA_PB_norm"] - df["sB_PA_norm"]
    rows = []
    for (a, b), g in df.groupby(["axis_a", "axis_b"]):
        s = g["cross_asym"].to_numpy()
        med = float(np.median(s))
        n = len(s)
        cons = (np.sum(s > 0) if med > 0 else np.sum(s < 0)) / n if n else float("nan")
        rows.append({"axis_a": a, "axis_b": b, "cross_med": med, "cross_consensus": cons})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default=str(REPO / "results" / "sheu_cascade"),
                   help="Dir containing <T>/pathB/per_celltype.csv per time point.")
    p.add_argument("--timepoints", nargs="+", default=["1hr", "3hr", "5hr"])
    p.add_argument("--labels_csv", default=str(DEFAULT_LABELS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    labels = pd.read_csv(args.labels_csv)
    base = Path(args.base_dir)

    # Load each available time point
    per_T = {}
    for T in args.timepoints:
        pc = base / T / "pathB" / "per_celltype.csv"
        if not pc.exists():
            print(f"WARNING: missing {pc}; skipping {T}", file=sys.stderr)
            continue
        per_T[T] = load_timepoint(pc).merge(labels, on=["axis_a", "axis_b"], how="inner")

    if not per_T:
        print("FATAL: no time-point outputs found.", file=sys.stderr)
        sys.exit(2)

    lines = ["# Sheu single-frame cascade-direction — cross-time comparison", ""]
    lines.append("Primary metric: **cross_asym** (antisymmetric cross-engagement; "
                 "sign encodes direction). Each time point is a self-contained "
                 "single-frame detection — no cross-time data used in the method.")
    lines.append("")

    # ---- Directional accuracy by class, per time point ----
    lines.append("## Directional accuracy by class (sign vs expected)")
    lines.append("")
    classes = ["IFN_MUST", "IFN_SHOULD", "NFKB_SHOULD"]
    header = "| time | " + " | ".join(classes) + " | all directional |"
    lines.append(header)
    lines.append("|" + "---|" * (len(classes) + 2))
    for T, df in per_T.items():
        bench = df[df["counts_in_benchmark"].astype(str).str.lower() == "true"].copy()
        bench["exp"] = bench["expected_sign"].astype(float).astype(int)
        bench["correct"] = np.sign(bench["cross_med"]).astype(int) == bench["exp"]
        cells = []
        for c in classes:
            sub = bench[bench["benchmark_class"] == c]
            cells.append(f"{int(sub['correct'].sum())}/{len(sub)}" if len(sub) else "—")
        allc = f"{int(bench['correct'].sum())}/{len(bench)}"
        lines.append(f"| {T} | " + " | ".join(cells) + f" | {allc} |")
    lines.append("")

    # ---- Negative specificity ----
    lines.append("## Negative-pair specificity (|cross_asym|, lower ⇒ cleaner null)")
    lines.append("")
    lines.append("| time | median \\|cross_asym\\| NEGATIVE | median \\|cross_asym\\| IFN benchmark |")
    lines.append("|---|---|---|")
    for T, df in per_T.items():
        neg = df[df["benchmark_class"] == "NEGATIVE"]
        ifn = df[df["benchmark_class"].isin(["IFN_MUST", "IFN_SHOULD"])]
        nm = neg["cross_med"].abs().median() if len(neg) else float("nan")
        im = ifn["cross_med"].abs().median() if len(ifn) else float("nan")
        lines.append(f"| {T} | {nm:.4f} | {im:.4f} |")
    lines.append("")

    # ---- Kinetic curve for the MUST IFN cascades ----
    lines.append("## Per-pair cross_asym kinetics (signed; correct sign in parens)")
    lines.append("")
    must = labels[labels["benchmark_class"] == "IFN_MUST"][["axis_a", "axis_b", "expected_sign"]]
    times = list(per_T.keys())
    lines.append("| pair | expected | " + " | ".join(times) + " |")
    lines.append("|" + "---|" * (len(times) + 2))
    for _, r in must.iterrows():
        a, b, exp = r["axis_a"], r["axis_b"], int(float(r["expected_sign"]))
        cells = []
        for T in times:
            df = per_T[T]
            row = df[(df["axis_a"] == a) & (df["axis_b"] == b)]
            if len(row) == 0:
                cells.append("—")
            else:
                v = float(row["cross_med"].iloc[0])
                ok = "✓" if np.sign(v) == exp else "✗"
                cells.append(f"{v:+.4f} ({ok})")
        exps = "+" if exp > 0 else "−"
        lines.append(f"| {a}/{b} | {exps} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Headline ----
    lines.append("## Headline")
    lines.append("")
    best_T, best_acc = None, -1.0
    for T, df in per_T.items():
        bench = df[df["counts_in_benchmark"].astype(str).str.lower() == "true"].copy()
        bench["exp"] = bench["expected_sign"].astype(float).astype(int)
        acc = (np.sign(bench["cross_med"]).astype(int) == bench["exp"]).mean()
        if acc > best_acc:
            best_acc, best_T = acc, T
    lines.append(f"- Best single time frame for directional accuracy: **{best_T}** "
                 f"({best_acc:.0%} on the {int((per_T[best_T]['counts_in_benchmark'].astype(str).str.lower()=='true').sum())} directional pairs).")
    lines.append("- IFN cascades (MUST) are the clean test (distinct pathways, §24 "
                 "precondition holds). NF-κB cascades (SHOULD) are expected weaker "
                 "(pathway overlap). Negatives should stay near zero.")
    lines.append("- Reminder: cross_asym (not the symmetric directional_score) is the "
                 "direction-bearing quantity — on Oesinghaus 24h it scored 88% vs 47%.")
    lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}")
    print(f"Best time frame: {best_T} ({best_acc:.0%})")


if __name__ == "__main__":
    main()
