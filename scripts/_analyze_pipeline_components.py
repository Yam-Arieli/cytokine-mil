"""
Local analysis on per_celltype.csv from the full-19 pipeline run.

Three questions:
  1. Do (asym_PA, asym_PB) cluster differently across literature_direction
     classes, even though their scalar difference (directional_score) doesn't?
  2. For the top "wrong" calls by |directional_score|, is the wrong sign
     uniform across cell types or driven by 1-2 outlier cell types
     (i.e., is median aggregation hiding a cascade that lives in specific
     cell types)?
  3. Which 5 axes have the biggest "wrong-direction" calls? (caller to
     gut-check against literature)

Pure pandas/numpy, no plots. Writes a markdown summary to stdout.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CSV_PATH = Path("results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv")


def _median_consensus(series: pd.Series) -> tuple[float, float, int]:
    """Median + signed-consensus fraction. Matches pipeline aggregation."""
    arr = series.to_numpy()
    med = float(np.median(arr))
    n = len(arr)
    if med > 0:
        consensus_n = int(np.sum(arr > 0))
    elif med < 0:
        consensus_n = int(np.sum(arr < 0))
    else:
        consensus_n = 0
    return med, consensus_n / n, n


def analyze_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per axis: median asym_PA, median asym_PB, median directional_score,
    consensus on directional_score sign, literature_direction.
    """
    rows = []
    for (a, b), g in df.groupby(["axis_a", "axis_b"], sort=False):
        med_PA, _, _ = _median_consensus(g["asym_PA"])
        med_PB, _, _ = _median_consensus(g["asym_PB"])
        med_dir, cons_dir, n = _median_consensus(g["directional_score"])
        rows.append({
            "axis_a": a,
            "axis_b": b,
            "lit": g["literature_direction"].iloc[0],
            "med_asym_PA": med_PA,
            "med_asym_PB": med_PB,
            "med_dir_score": med_dir,
            "consensus_dir": cons_dir,
            "n_celltypes": n,
        })
    return pd.DataFrame(rows)


def summarize_by_lit_class(per_axis: pd.DataFrame) -> str:
    """For each literature_direction class, report distribution of components."""
    lines = ["## 1. Component pattern by literature class", ""]
    lines.append(
        "By the §24 algebra (A=axis_a, B=axis_b, P_A=S_A, P_B=S_B):"
    )
    lines.append("")
    lines.append("- A→B cascade  → asym_PA > 0, asym_PB ~ 0     (B engages B's signature, A engages B's signature partially via cascade)")
    lines.append("- B→A cascade  → asym_PA ~ 0, asym_PB < 0     (symmetric)")
    lines.append("- No cascade   → asym_PA > 0, asym_PB < 0     (each cytokine engages its own signature exclusively)")
    lines.append("- Coregulated  → asym_PA ~ 0, asym_PB ~ 0     (signatures overlap)")
    lines.append("")
    lines.append("| lit_class | n | median asym_PA | median asym_PB | median dir_score | mean dir_score |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for lit_cls, g in per_axis.groupby("lit"):
        if g.empty:
            continue
        lines.append(
            f"| {lit_cls} | {len(g)} | "
            f"{g['med_asym_PA'].median():+.4f} | "
            f"{g['med_asym_PB'].median():+.4f} | "
            f"{g['med_dir_score'].median():+.4f} | "
            f"{g['med_dir_score'].mean():+.4f} |"
        )
    lines.append("")

    # Cluster signature check: how often does each class fall into each algebraic regime?
    lines.append("### Algebraic regime hits (per axis)")
    lines.append("")
    lines.append("Regime decision per axis:")
    lines.append("- A→B-like   : asym_PA > 0.005 AND |asym_PB| < 0.005")
    lines.append("- B→A-like   : asym_PB < -0.005 AND |asym_PA| < 0.005")
    lines.append("- nocas-like : asym_PA > 0.005 AND asym_PB < -0.005")
    lines.append("- corr-like  : |asym_PA| < 0.005 AND |asym_PB| < 0.005")
    lines.append("- mixed      : everything else")
    lines.append("")

    def _regime(r):
        pa, pb = r["med_asym_PA"], r["med_asym_PB"]
        eps = 0.005
        if pa > eps and abs(pb) < eps:
            return "A→B-like"
        if pb < -eps and abs(pa) < eps:
            return "B→A-like"
        if pa > eps and pb < -eps:
            return "nocas-like"
        if abs(pa) < eps and abs(pb) < eps:
            return "corr-like"
        return "mixed"

    per_axis = per_axis.copy()
    per_axis["regime"] = per_axis.apply(_regime, axis=1)

    pivot = per_axis.pivot_table(
        index="lit", columns="regime", values="axis_a",
        aggfunc="count", fill_value=0
    )
    lines.append(pivot.to_markdown())
    lines.append("")
    return "\n".join(lines), per_axis


def find_wrong_calls(per_axis: pd.DataFrame, top_k: int = 5) -> tuple[str, pd.DataFrame]:
    """Identify axes where the call disagrees with literature."""
    # Only a_to_b and b_to_a have a clean expected sign
    clean_gt = per_axis[per_axis["lit"].isin(["a_to_b", "b_to_a"])].copy()

    def _expected_sign(lit: str) -> int:
        return +1 if lit == "a_to_b" else -1

    clean_gt["expected_sign"] = clean_gt["lit"].apply(_expected_sign)
    clean_gt["observed_sign"] = np.sign(clean_gt["med_dir_score"]).astype(int)
    clean_gt["is_correct"] = (clean_gt["expected_sign"] == clean_gt["observed_sign"])
    clean_gt["abs_score"] = clean_gt["med_dir_score"].abs()

    # Apply AMBIGUOUS filter: |median| < 0.01 was the pipeline's threshold
    clean_gt["is_ambiguous"] = clean_gt["abs_score"] < 0.01

    correct = clean_gt[clean_gt["is_correct"] & ~clean_gt["is_ambiguous"]]
    wrong = clean_gt[~clean_gt["is_correct"] & ~clean_gt["is_ambiguous"]]
    ambig = clean_gt[clean_gt["is_ambiguous"]]

    lines = ["## 2. Right-vs-wrong-vs-ambiguous tally on literature-tagged axes", ""]
    lines.append(f"- a_to_b + b_to_a axes total: **{len(clean_gt)}**")
    lines.append(f"- Correct sign (non-ambiguous): **{len(correct)}**  ({len(correct)/len(clean_gt):.0%})")
    lines.append(f"- Wrong   sign (non-ambiguous): **{len(wrong)}**  ({len(wrong)/len(clean_gt):.0%})")
    lines.append(f"- AMBIGUOUS (|median| < 0.01): **{len(ambig)}**  ({len(ambig)/len(clean_gt):.0%})")
    lines.append("")
    lines.append("Among AMBIGUOUS, would-be sign rate:")
    if len(ambig) > 0:
        ambig_sign_match = (np.sign(ambig["med_dir_score"]).astype(int) == ambig["expected_sign"]).sum()
        lines.append(f"  - would-be correct: {ambig_sign_match}/{len(ambig)} ({ambig_sign_match/len(ambig):.0%})")
    lines.append("")

    lines.append(f"### Top {top_k} wrong calls by |median directional_score|")
    lines.append("")
    lines.append("| axis_a | axis_b | lit | expected | observed | median_score | med_asym_PA | med_asym_PB |")
    lines.append("|---|---|---|:---:|:---:|---:|---:|---:|")
    top_wrong = wrong.sort_values("abs_score", ascending=False).head(top_k)
    for _, r in top_wrong.iterrows():
        expected = "+" if r["expected_sign"] > 0 else "-"
        observed = "+" if r["observed_sign"] > 0 else "-"
        lines.append(
            f"| {r['axis_a']} | {r['axis_b']} | {r['lit']} | "
            f"{expected} | {observed} | "
            f"{r['med_dir_score']:+.4f} | "
            f"{r['med_asym_PA']:+.4f} | "
            f"{r['med_asym_PB']:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines), top_wrong


def per_celltype_breakdown(df: pd.DataFrame, top_wrong: pd.DataFrame) -> str:
    """For each of top-5 wrong calls, show per-cell-type directional_score."""
    lines = ["## 3. Per-cell-type breakdown of top wrong calls", ""]
    lines.append("If the cascade signal lives in 1-2 specific cell types but gets diluted by")
    lines.append("the median across all cell types, the cell-type stratification of §24 is")
    lines.append("being undermined by the aggregation choice (not by the method itself).")
    lines.append("")

    for _, row in top_wrong.iterrows():
        a, b, lit = row["axis_a"], row["axis_b"], row["lit"]
        expected_sign = "+" if lit == "a_to_b" else "-"
        sub = df[(df["axis_a"] == a) & (df["axis_b"] == b)].copy()
        sub = sub.sort_values("directional_score")
        lines.append(f"### {a} → {b}  ({lit}, expected sign {expected_sign})")
        lines.append("")
        n_pos = (sub["directional_score"] > 0).sum()
        n_neg = (sub["directional_score"] < 0).sum()
        lines.append(f"- {n_pos} cell types positive, {n_neg} negative, median {sub['directional_score'].median():+.4f}")
        if expected_sign == "+":
            expected_dir_n = n_pos
        else:
            expected_dir_n = n_neg
        lines.append(f"- Cell types in expected direction ({expected_sign}): **{expected_dir_n} / {len(sub)}**")
        lines.append("")
        lines.append("| cell_type | directional_score | asym_PA | asym_PB |")
        lines.append("|---|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['cell_type']} | "
                f"{r['directional_score']:+.4f} | "
                f"{r['asym_PA']:+.4f} | "
                f"{r['asym_PB']:+.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    df = pd.read_csv(CSV_PATH)
    per_axis = analyze_components(df)

    out = [f"# Pipeline component analysis ({CSV_PATH})", ""]
    out.append(f"- {df['cell_type'].nunique()} cell types × {len(per_axis)} axes")
    out.append("")

    s1, per_axis = summarize_by_lit_class(per_axis)
    out.append(s1)

    s2, top_wrong = find_wrong_calls(per_axis, top_k=5)
    out.append(s2)

    out.append(per_celltype_breakdown(df, top_wrong))

    report = "\n".join(out)
    print(report)


if __name__ == "__main__":
    main()
