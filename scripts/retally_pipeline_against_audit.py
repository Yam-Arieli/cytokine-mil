"""
Re-tally pipeline accuracy against the strict audited labels.

Inputs:
  - results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv
    (per-axis × per-cell-type directional_score; can override path)
  - reports/cascade_pairs/cytokine_axes_audited.csv (audited labels)

Output:
  - reports/cascade_pairs/pipeline_accuracy_audited.md

For each axis we compute:
  - median(directional_score) across cell types  (matches the pipeline's
    median+consensus aggregator)
  - sign of the median
  - consensus fraction (how many cell types share the median sign)

We compare the *sign* against `expected_sign` from the audit:
  - `counts_in_benchmark = True` rows are the headline accuracy denominator.
  - `WEAK_*` and `DIRECTIONAL_*_NOISY` are reported separately as
    "weak benchmark" — sign is meaningful but evidence weaker.
  - `PARTIAL_INHIBITORY`, `LOW_CONFIDENCE`, `UNKNOWN`, `BIDIRECTIONAL`
    are reported by sign distribution only (not graded).

Also reports the **delta vs original-tag accuracy** to quantify the audit's
effect on the conclusion.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE = REPO / "results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv"
DEFAULT_AUDIT = REPO / "reports/cascade_pairs/cytokine_axes_audited.csv"
DEFAULT_OUT = REPO / "reports/cascade_pairs/pipeline_accuracy_audited.md"


def aggregate_per_axis(df: pd.DataFrame, metric: str = "cross_asym") -> pd.DataFrame:
    """Median+consensus aggregation matching the pipeline.

    metric defaults to `cross_asym` (the antisymmetric, direction-bearing
    quantity). If the column is absent (older per_celltype.csv), it is derived
    from `sA_PB_norm - sB_PA_norm`. `directional_score` may be passed to
    reproduce the old (symmetric, chance-level) call for comparison.
    """
    df = df.copy()
    if metric == "cross_asym" and "cross_asym" not in df.columns:
        df["cross_asym"] = df["sA_PB_norm"] - df["sB_PA_norm"]
    rows = []
    for (a, b), g in df.groupby(["axis_a", "axis_b"], sort=False):
        scores = g[metric].to_numpy()
        med = float(np.median(scores))
        n = len(scores)
        if med > 0:
            cons = int(np.sum(scores > 0)) / n
        elif med < 0:
            cons = int(np.sum(scores < 0)) / n
        else:
            cons = 0.0
        observed_sign = int(np.sign(med)) if med != 0 else 0
        rows.append({
            "axis_a": a, "axis_b": b,
            "median_dir": med,
            "consensus": cons,
            "observed_sign": observed_sign,
            "n_celltypes": n,
        })
    return pd.DataFrame(rows)


def _safe_int(x) -> int | None:
    try:
        if x == "" or x is None:
            return None
        return int(x)
    except (ValueError, TypeError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline_csv", type=Path, default=DEFAULT_PIPELINE)
    p.add_argument("--audit_csv", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--metric", default="cross_asym",
                   choices=["cross_asym", "directional_score"],
                   help="Per-cell-type quantity to aggregate. cross_asym is "
                        "the direction-bearing default; directional_score "
                        "reproduces the old symmetric (chance-level) call.")
    args = p.parse_args()

    for path in (args.pipeline_csv, args.audit_csv):
        if not path.exists():
            print(f"FATAL: missing input: {path}", file=sys.stderr)
            sys.exit(2)

    pipeline_df = pd.read_csv(args.pipeline_csv)
    audit_df = pd.read_csv(args.audit_csv)

    agg = aggregate_per_axis(pipeline_df, metric=args.metric)
    # Join on (axis_a, axis_b). Both sides are already canonical.
    merged = agg.merge(audit_df, on=["axis_a", "axis_b"], how="inner")
    assert len(merged) == len(audit_df), \
        f"axes don't all join: agg={len(agg)} audit={len(audit_df)} merged={len(merged)}"

    # ------------------------------------------------------------ buckets
    # Headline (strict): counts_in_benchmark = True → DIRECTIONAL_a_to_b or DIRECTIONAL_b_to_a
    strict = merged[merged["counts_in_benchmark"].astype(str).str.lower() == "true"].copy()
    strict["expected_sign_int"] = strict["expected_sign"].apply(_safe_int)
    strict["sign_correct"] = strict["observed_sign"] == strict["expected_sign_int"]

    # Weak (NOISY + WEAK_*): expected_sign present but excluded from headline
    weak_mask = merged["pair_status"].isin(
        ["WEAK_a_to_b", "WEAK_b_to_a",
         "DIRECTIONAL_a_to_b_NOISY", "DIRECTIONAL_b_to_a_NOISY"]
    )
    weak = merged[weak_mask].copy()
    weak["expected_sign_int"] = weak["expected_sign"].apply(_safe_int)
    weak["sign_correct"] = weak["observed_sign"] == weak["expected_sign_int"]

    # Other (PARTIAL_INHIBITORY, LOW_CONFIDENCE, UNKNOWN, BIDIRECTIONAL):
    # no expected sign — just report sign distribution
    other = merged[~merged["counts_in_benchmark"].astype(str).str.lower().eq("true")
                   & ~weak_mask].copy()

    # ------------------------------------------------------------ original-tag accuracy (delta vs audited)
    # For each axis with original_direction in {a_to_b, b_to_a}, compute expected sign per old tag
    orig_dir_map = {"a_to_b": +1, "b_to_a": -1}
    orig = merged[merged["original_direction"].isin(orig_dir_map.keys())].copy()
    orig["orig_expected_sign"] = orig["original_direction"].map(orig_dir_map)
    orig["orig_sign_correct"] = orig["observed_sign"] == orig["orig_expected_sign"]
    # AMBIGUOUS filter — the pipeline's |score| < 0.01 rule
    orig["orig_ambig"] = orig["median_dir"].abs() < 0.01

    # ------------------------------------------------------------ write report
    lines = []
    lines.append("# Pipeline accuracy re-tallied against strict audited labels")
    lines.append("")
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO))
        except ValueError:
            return str(p)
    lines.append(f"- Pipeline output: `{_rel(args.pipeline_csv)}`")
    lines.append(f"- Audited labels: `{_rel(args.audit_csv)}`")
    lines.append(f"- Metric: `{args.metric}`")
    lines.append("")
    lines.append("## Per-axis aggregation rule")
    lines.append(
        f"Median of `{args.metric}` across cell types, plus sign-consensus fraction. "
        "Matches the pipeline's median+consensus aggregator."
    )
    lines.append("")
    lines.append("## Headline accuracy (strict audited benchmark)")
    lines.append("")
    lines.append(f"- Axes in benchmark: **{len(strict)}** (DIRECTIONAL_a_to_b + DIRECTIONAL_b_to_a)")
    n_correct = int(strict["sign_correct"].sum())
    pct = n_correct / len(strict) if len(strict) else float("nan")
    lines.append(f"- Sign-correct calls: **{n_correct} / {len(strict)} = {pct:.0%}**")
    lines.append("")
    lines.append("### Per status sub-tally")
    lines.append("")
    lines.append("| pair_status | n | n_correct | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for st, g in strict.groupby("pair_status"):
        n = len(g)
        nc = int(g["sign_correct"].sum())
        a = nc / n if n else float("nan")
        lines.append(f"| {st} | {n} | {nc} | {a:.0%} |")
    lines.append("")
    lines.append("### Per-axis detail (strict benchmark)")
    lines.append("")
    lines.append("| axis | pair_status | expected | observed_sign | median | consensus | tag_changed | correct |")
    lines.append("|---|---|:-:|:-:|---:|---:|:-:|:-:|")
    for _, r in strict.sort_values(["pair_status", "axis_a", "axis_b"]).iterrows():
        exp = "+" if r["expected_sign_int"] == +1 else ("−" if r["expected_sign_int"] == -1 else "?")
        obs = "+" if r["observed_sign"] == +1 else ("−" if r["observed_sign"] == -1 else "0")
        tc = "✓" if str(r["tag_changed"]).lower() == "true" else ""
        ok = "✓" if r["sign_correct"] else "✗"
        lines.append(
            f"| {r['axis_a']} / {r['axis_b']} | {r['pair_status']} | "
            f"{exp} | {obs} | {r['median_dir']:+.4f} | {r['consensus']:.2f} | "
            f"{tc} | {ok} |"
        )
    lines.append("")

    # Weak bucket
    lines.append("## Weak benchmark (reported separately)")
    lines.append("")
    lines.append(f"Axes with WEAK_* or DIRECTIONAL_*_NOISY status. Sign has direction but evidence is weak.")
    lines.append(f"")
    lines.append(f"- Axes in weak benchmark: **{len(weak)}**")
    if len(weak) > 0:
        n_correct_w = int(weak["sign_correct"].sum())
        lines.append(f"- Sign-correct: **{n_correct_w} / {len(weak)} = {n_correct_w/len(weak):.0%}**")
    lines.append("")
    if len(weak) > 0:
        lines.append("| axis | pair_status | expected | observed_sign | median | tag_changed | correct |")
        lines.append("|---|---|:-:|:-:|---:|:-:|:-:|")
        for _, r in weak.sort_values(["pair_status", "axis_a", "axis_b"]).iterrows():
            exp = "+" if r["expected_sign_int"] == +1 else "−"
            obs = "+" if r["observed_sign"] == +1 else ("−" if r["observed_sign"] == -1 else "0")
            tc = "✓" if str(r["tag_changed"]).lower() == "true" else ""
            ok = "✓" if r["sign_correct"] else "✗"
            lines.append(
                f"| {r['axis_a']} / {r['axis_b']} | {r['pair_status']} | "
                f"{exp} | {obs} | {r['median_dir']:+.4f} | {tc} | {ok} |"
            )
        lines.append("")

    # Other bucket
    lines.append("## Excluded from accuracy (no graded sign)")
    lines.append("")
    lines.append("| pair_status | n | n_positive_score | n_negative_score | n_zero |")
    lines.append("|---|---:|---:|---:|---:|")
    for st, g in other.groupby("pair_status"):
        n = len(g)
        npos = int((g["observed_sign"] == +1).sum())
        nneg = int((g["observed_sign"] == -1).sum())
        nzero = int((g["observed_sign"] == 0).sum())
        lines.append(f"| {st} | {n} | {npos} | {nneg} | {nzero} |")
    lines.append("")

    # Delta vs original
    lines.append("## Delta vs original keyword-parsed tags")
    lines.append("")
    n_orig = len(orig)
    n_orig_correct = int(orig["orig_sign_correct"].sum())
    n_orig_ambig = int(orig["orig_ambig"].sum())
    n_orig_non_ambig = n_orig - n_orig_ambig
    n_orig_correct_non_ambig = int(orig.loc[~orig["orig_ambig"], "orig_sign_correct"].sum())
    lines.append("Original cytokine_axes.csv tags (any axis with original_direction ∈ {a_to_b, b_to_a}):")
    lines.append(f"- Axes graded by original tag: **{n_orig}**")
    lines.append(f"- Sign-correct vs original tag (all): **{n_orig_correct} / {n_orig} = {n_orig_correct/n_orig:.0%}**")
    lines.append(f"- Of those AMBIGUOUS (|median| < 0.01): **{n_orig_ambig}**")
    lines.append(f"- Sign-correct among non-ambiguous: **{n_orig_correct_non_ambig} / {n_orig_non_ambig} = "
                 f"{(n_orig_correct_non_ambig/n_orig_non_ambig if n_orig_non_ambig else float('nan')):.0%}**")
    lines.append("")
    n_tag_flipped = int(orig["tag_changed"].astype(str).str.lower().eq("true").sum())
    lines.append(f"- Original tags flipped by the audit: **{n_tag_flipped}** (of {n_orig} graded)")
    lines.append("")
    lines.append(
        "Interpretation: the audit reassigns ~"
        f"{n_tag_flipped} of the {n_orig} original tags. The strict benchmark on audited labels "
        f"({n_correct} / {len(strict)} = {pct:.0%}) is the headline number — the original-tag accuracy "
        f"is biased by mislabels and not directly comparable."
    )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}", flush=True)

    # Console summary
    print()
    print(f"Headline (strict audited benchmark): {n_correct}/{len(strict)} = {pct:.0%}")
    print(f"Weak bucket: {int(weak['sign_correct'].sum()) if len(weak) else 0}/{len(weak)}")
    print(f"Original-tag accuracy (all): {n_orig_correct}/{n_orig} = {n_orig_correct/n_orig:.0%}")
    print(f"Original-tag accuracy (non-ambig only): "
          f"{n_orig_correct_non_ambig}/{n_orig_non_ambig}")
    print(f"Tags flipped by audit: {n_tag_flipped}/{n_orig}")


if __name__ == "__main__":
    main()
