"""
Group-U direction FDR + verdict (CLAUDE.md §27.3).

Reads the per-axis summary produced by `run_pipeline_a_bridge_b.py`
(`--n_direction_perms 1000`) over all 121 Path A axes, joins the audited
labels, partitions:

  * LABELED  : audited `counts_in_benchmark == True` (directional ground truth)
  * GROUP U  : every other evaluable axis (Path-A-coupled, NO directional prior)

Then quantifies the unknown directional calls WITHOUT ground truth:

  * BH-FDR over Group-U `dir_p_emp`     -> n significant at q in {0.05, 0.10}
  * Storey pi0 (lambda=0.5) over Group U -> est. fraction with NO reliable dir
  * 1 - pi0                              -> est. fraction WITH reliable direction
  * Calibration P1/P2 on the labeled set, regression P4

and emits the ranked confident Group-U hypothesis list + GROUP_U_RESULTS.md.

Verdict against the locked pre-registration
(`reports/cascade_pairs/GROUP_U_PREREGISTRATION.md`):
  P1 power     : >= 80% labeled non-AMBIGUOUS positives pass the direction null
  P2 spec      : near-zero / miss pairs do NOT pass (descriptive)
  P3 headline  : Group-U pi0 < 0.9  => discovery-capable (else confirmation-only)
  P4 regression: labeled accuracy unchanged vs the §26 15/17

Allowed imports: argparse, sys, pathlib, numpy, pandas, and the numpy-only
helpers in cytokine_mil.analysis.direction_null. NO scipy/matplotlib/tabulate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.direction_null import bh_fdr, storey_pi0  # noqa: E402


DEFAULT_PER_AXIS = (
    REPO_ROOT / "results/group_u/pipeline_full121/per_axis_summary.csv"
)
DEFAULT_AUDIT = REPO_ROOT / "reports/cascade_pairs/cytokine_axes_audited.csv"
DEFAULT_OUT = REPO_ROOT / "reports/cascade_pairs/GROUP_U_RESULTS.md"

# §26 headline labeled accuracy, for the P4 regression check.
REF_LABELED_ACC = "15/17 (§26 audited benchmark)"

ALPHA = 0.05
P1_POWER_FLOOR = 0.80
P3_PI0_CEILING = 0.90
CONSENSUS_FLOOR = 0.70
TOP_K = 10


def _md_table(df: pd.DataFrame, columns) -> str:
    cols = list(columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append("NaN" if np.isnan(v) else f"{v:+.4f}")
            elif isinstance(v, bool):
                cells.append("True" if v else "False")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per_axis_csv", type=Path, default=DEFAULT_PER_AXIS)
    p.add_argument("--audit_csv", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    for path in (args.per_axis_csv, args.audit_csv):
        if not path.exists():
            print(f"FATAL: missing input: {path}", file=sys.stderr)
            sys.exit(2)

    pa = pd.read_csv(args.per_axis_csv)
    aud = pd.read_csv(args.audit_csv)

    if "dir_p_emp" not in pa.columns:
        print("FATAL: per_axis_summary has no dir_p_emp column. Re-run the "
              "pipeline with --n_direction_perms > 0.", file=sys.stderr)
        sys.exit(2)

    # ---- join audited labels (left join: keep all evaluated axes) ----
    aud["counts_in_benchmark"] = (
        aud["counts_in_benchmark"].astype(str).str.lower() == "true"
    )
    aud_keep = aud[["axis_a", "axis_b", "counts_in_benchmark", "pair_status"]].copy()
    aud_keep = aud_keep.rename(columns={"pair_status": "audit_pair_status"})
    m = pa.merge(aud_keep, on=["axis_a", "axis_b"], how="left")
    m["counts_in_benchmark"] = m["counts_in_benchmark"].fillna(False)
    m["audit_pair_status"] = m["audit_pair_status"].fillna("UNAUDITED_NO_PRIOR")
    m["abs_cross"] = m["cross_median"].abs()

    labeled = m[m["counts_in_benchmark"]].copy()
    group_u = m[~m["counts_in_benchmark"]].copy()

    n_eval = len(m)
    n_labeled = len(labeled)
    n_group_u = len(group_u)

    # ---- BH-FDR ----
    # Over all evaluable axes (for P1 on labeled) and over Group U (headline P3).
    m["dir_q_all"] = bh_fdr(m["dir_p_emp"].to_numpy())
    group_u = group_u.copy()
    group_u["dir_q_gu"] = bh_fdr(group_u["dir_p_emp"].to_numpy())
    # map q_all back onto subsets
    q_all_map = dict(zip(zip(m["axis_a"], m["axis_b"]), m["dir_q_all"]))
    labeled = labeled.copy()
    labeled["dir_q_all"] = [
        q_all_map[(a, b)] for a, b in zip(labeled["axis_a"], labeled["axis_b"])
    ]

    # ---- P1: power on labeled non-AMBIGUOUS ----
    lab_na = labeled[labeled["classification"] != "AMBIGUOUS"]
    n_lab_na = len(lab_na)
    lab_pass_raw = int((lab_na["dir_p_emp"] < ALPHA).sum())
    lab_pass_q = int((lab_na["dir_q_all"] <= 0.10).sum())
    p1_rate = (lab_pass_raw / n_lab_na) if n_lab_na else float("nan")
    p1_pass = (not np.isnan(p1_rate)) and (p1_rate >= P1_POWER_FLOOR)

    # ---- P3: Group-U FDR + pi0 ----
    gu_p = group_u["dir_p_emp"].to_numpy()
    pi0 = storey_pi0(gu_p, lam=0.5)
    n_gu_valid = int((~np.isnan(gu_p)).sum())
    est_true = (1.0 - pi0) * n_gu_valid if not np.isnan(pi0) else float("nan")
    n_gu_q05 = int((group_u["dir_q_gu"] <= 0.05).sum())
    n_gu_q10 = int((group_u["dir_q_gu"] <= 0.10).sum())
    p3_pass = (not np.isnan(pi0)) and (pi0 < P3_PI0_CEILING)

    # ---- confident hypothesis bar (P25 of labeled-positive |cross|) ----
    if len(labeled):
        cross_floor = float(np.nanpercentile(labeled["abs_cross"].to_numpy(), 25))
    else:
        cross_floor = float("nan")
    confident = group_u[
        (group_u["dir_q_gu"] <= 0.10)
        & (group_u["cross_consensus"] >= CONSENSUS_FLOOR)
        & (group_u["abs_cross"] >= cross_floor)
    ].copy()
    confident = confident.sort_values(
        ["dir_q_gu", "abs_cross"], ascending=[True, False]
    )

    # ---- P4: labeled regression ----
    lab_gt = labeled[labeled["classification"] != "AMBIGUOUS"]
    n_lab_correct = int(lab_gt["cross_sign_correct"].astype(bool).sum())
    n_lab_gt = len(lab_gt)

    # direction call string for the ranked list
    def _dir_str(row) -> str:
        if row["cross_median"] > 0:
            return f"{row['axis_a']} -> {row['axis_b']}"
        if row["cross_median"] < 0:
            return f"{row['axis_b']} -> {row['axis_a']}"
        return "ambiguous"

    confident["direction_call"] = confident.apply(_dir_str, axis=1)

    # ---------------------------------------------------------------- report
    L = []
    L.append("# Group-U direction FDR — full Path A -> Path B (Oesinghaus)")
    L.append("")
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p)
    L.append(f"- Per-axis input: `{_rel(args.per_axis_csv)}`")
    L.append(f"- Audited labels: `{_rel(args.audit_csv)}`")
    L.append(f"- Pre-registration: `reports/cascade_pairs/GROUP_U_PREREGISTRATION.md`")
    L.append("")
    L.append("Direction null = §27.2 permutation null (hold S_a,S_b fixed, permute "
             "a/b cell labels within each cell type, recentred two-sided p). It tests "
             "whether the **direction is reliable**, NOT whether the pair is a cascade "
             "(existence = Path A) and NOT causation (wet-lab).")
    L.append("")
    L.append("## Partition")
    L.append("")
    L.append(f"- Evaluable axes (direction computed): **{n_eval}**")
    L.append(f"- LABELED (audited counts_in_benchmark): **{n_labeled}**")
    L.append(f"- GROUP U (coupled by Path A, no directional prior): **{n_group_u}**")
    L.append("")
    L.append("## Verdict vs pre-registration")
    L.append("")
    L.append(f"- **P1 (power):** {lab_pass_raw}/{n_lab_na} labeled non-AMBIGUOUS "
             f"positives pass the direction null at raw p<{ALPHA} "
             f"(= {p1_rate:.0%}); {lab_pass_q}/{n_lab_na} at BH-q<=0.10. "
             f"Floor {P1_POWER_FLOOR:.0%} -> **{'PASS' if p1_pass else 'FAIL'}**.")
    L.append(f"- **P3 (headline — discovery-capable?):** Group-U Storey "
             f"pi0 = **{pi0:.3f}** (ceiling {P3_PI0_CEILING}). "
             f"Est. **{est_true:.0f} / {n_gu_valid}** Group-U axes carry a reliable "
             f"directional asymmetry (1-pi0 = {1.0-pi0:.0%}). BH-significant Group-U "
             f"axes: {n_gu_q05} at q<=0.05, {n_gu_q10} at q<=0.10. "
             f"-> **{'DISCOVERY-CAPABLE' if p3_pass else 'CONFIRMATION-ONLY'}**.")
    L.append(f"- **P4 (regression):** labeled cross_asym sign accuracy "
             f"(non-AMBIGUOUS) = **{n_lab_correct}/{n_lab_gt}** "
             f"(ref {REF_LABELED_ACC}).")
    L.append(f"- **P2 (specificity):** see the labeled + descriptive tables below "
             f"(near-zero / miss pairs should not pass; overlap pairs with a real "
             f"magnitude asymmetry may, by design).")
    L.append("")
    L.append(f"Confident-hypothesis bar (locked): dir BH-q(GroupU) <= 0.10 AND "
             f"cross_consensus >= {CONSENSUS_FLOOR} AND |cross_median| >= "
             f"P25(labeled-positive) = {cross_floor:.4f}. "
             f"**{len(confident)}** Group-U axes clear it.")
    L.append("")
    L.append(f"## Top-{TOP_K} confident Group-U directional hypotheses")
    L.append("")
    if len(confident):
        top = confident.head(TOP_K).copy()
        L.append(_md_table(
            top,
            columns=["direction_call", "literature_status", "cross_median",
                     "cross_consensus", "dir_p_emp", "dir_q_gu", "audit_pair_status"],
        ))
    else:
        L.append("_No Group-U axis cleared the confident bar._")
    L.append("")
    L.append("## Labeled set (P1/P4 detail)")
    L.append("")
    L.append(_md_table(
        labeled.sort_values("dir_p_emp"),
        columns=["axis_a", "axis_b", "expected_sign", "cross_median",
                 "cross_consensus", "classification", "cross_sign_correct",
                 "dir_p_emp", "dir_q_all"],
    ))
    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- pi0 (Storey, lambda=0.5) estimates the fraction of Group-U axes "
             "with NO reliable directional asymmetry; 1-pi0 is the est. true-signal "
             "fraction. Small Group-U n makes pi0 noisy — report with the BH counts.")
    L.append("- This characterises Group U as a ranked hypothesis pile, not a set of "
             "verdicts. Confirming any individual hypothesis needs wet-lab (or, for a "
             "Sheu-overlapping pair, the kinetic precedence cross-check).")
    L.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L))
    print(f"Wrote {args.out}", flush=True)

    # console summary
    print()
    print(f"Evaluable={n_eval}  labeled={n_labeled}  group_u={n_group_u}")
    print(f"P1 power: {lab_pass_raw}/{n_lab_na} ({p1_rate:.0%}) pass raw p<{ALPHA} "
          f"-> {'PASS' if p1_pass else 'FAIL'}")
    print(f"P3 headline: Group-U pi0={pi0:.3f}  est_true={est_true:.0f}/{n_gu_valid}  "
          f"BH q<=0.10: {n_gu_q10}  -> "
          f"{'DISCOVERY-CAPABLE' if p3_pass else 'CONFIRMATION-ONLY'}")
    print(f"P4 regression: labeled acc {n_lab_correct}/{n_lab_gt} (ref {REF_LABELED_ACC})")
    print(f"Confident Group-U hypotheses: {len(confident)} (top-{TOP_K} in report)")


if __name__ == "__main__":
    main()
