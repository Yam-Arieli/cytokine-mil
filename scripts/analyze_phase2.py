"""Phase 1+2 stage C — score the four arms and apply the pre-registered rule.

Reads every `arm_*/signatures.parquet`, scores each with `analyze_encsweep.diversity`
(verbatim — the function behind every number in CLAUDE.md §38.3/§38.4 and §40, so these
land in the same ladder), and reports:

  * the four arm means with their across-seed spread;
  * P1 — does the gap reproduce?  mean(cd_cd) - mean(cm_cm) >= 0.08 with non-overlapping
    seed ranges;
  * P2 — the seed control. Within-arm spread must be < 1/3 of the between-arm gap, or the
    comparison is measuring run-to-run noise and NOTHING else here means anything;
  * the Phase 2 bisection: which of Stage 1 / Stage 2 the transplants follow.

The falsification branch is live and is reported as prominently as a confirmation: if
|mean(cd_cd) - mean(cm_cm)| < 0.03 the code-path hypothesis is REJECTED and the cause of
the collapse is again unknown.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _phase2_config as C  # noqa: E402
from analyze_encsweep import diversity  # noqa: E402

GAP_REPRODUCES = 0.08
FALSIFY_BELOW = 0.03
SEED_SPREAD_FRACTION = 1.0 / 3.0


def _closer_to(x: float, a: float, b: float) -> str:
    return "cm" if abs(x - a) <= abs(x - b) else "cd"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--top_n", type=int, default=C.TOP_N)
    args = ap.parse_args()

    out = Path(args.out_dir)
    rows = []
    for enc_p, s2_p in C.ARMS:
        for seed in C.SEEDS:
            p = C.arm_dir(enc_p, s2_p, seed) if out == C.OUT_DIR else (
                out / f"arm_{C.arm_name(enc_p, s2_p)}_seed{seed}")
            f = p / "signatures.parquet"
            if not f.exists():
                print(f"[skip] missing {f}")
                continue
            df = pd.read_parquet(f)
            rec = {"arm": C.arm_name(enc_p, s2_p), "encoder_path": enc_p,
                   "stage2_path": s2_p, "seed": seed}
            rec.update(diversity(df, args.top_n))
            rows.append(rec)
    if not rows:
        raise SystemExit("no arm signatures found — nothing to score")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "phase2_per_seed.csv", index=False)

    agg = (per_seed.groupby("arm")
           .agg(n_seeds=("seed", "size"),
                mean_jaccard=("mean_jaccard", "mean"),
                sd=("mean_jaccard", "std"),
                lo=("mean_jaccard", "min"),
                hi=("mean_jaccard", "max"),
                distinct_genes=("distinct_genes", "mean"))
           .reset_index())
    agg.to_csv(out / "phase2_arm_summary.csv", index=False)

    print("\n" + "=" * 78)
    print(f"PER-SEED diversity at top-{args.top_n}")
    print(per_seed[["arm", "seed", "mean_jaccard", "top5_pool", "top5_worst_gene",
                    "distinct_genes"]].to_string(index=False,
                                                 float_format=lambda x: f"{x:.3f}"))
    print("\nARM SUMMARY (mean over seeds)")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 78)

    have = set(agg.arm)
    if not {"cm_cm", "cd_cd"} <= have:
        print("\n[verdict] both reference arms (cm_cm, cd_cd) are required — incomplete run.")
        return 0

    m = agg.set_index("arm")
    p_mean, c_mean = float(m.loc["cm_cm", "mean_jaccard"]), float(m.loc["cd_cd", "mean_jaccard"])
    gap = c_mean - p_mean
    spreads = {a: float(m.loc[a, "hi"] - m.loc[a, "lo"]) for a in have}
    max_spread = max(spreads.values())
    overlap = not (m.loc["cm_cm", "hi"] < m.loc["cd_cd", "lo"]
                   or m.loc["cd_cd", "hi"] < m.loc["cm_cm", "lo"])

    lines = []
    lines.append(f"cm_cm (pure cytokine_mil) = {p_mean:.3f}   "
                 f"cd_cd (pure cascadir) = {c_mean:.3f}   gap = {gap:+.3f}")
    lines.append(f"max within-arm seed spread = {max_spread:.3f}  "
                 f"({', '.join(f'{a}:{s:.3f}' for a, s in sorted(spreads.items()))})")
    lines.append("")

    # P2 first: if seed noise swamps the gap, nothing else is interpretable.
    p2 = max_spread < SEED_SPREAD_FRACTION * abs(gap) if gap else False
    lines.append(f"P2 seed control : {'PASS' if p2 else 'FAIL'} — spread {max_spread:.3f} "
                 f"vs 1/3 of gap {SEED_SPREAD_FRACTION*abs(gap):.3f}")
    if not p2:
        lines.append("   ** within-arm seed variance is NOT small against the between-arm")
        lines.append("      difference. These fits are run-to-run noisy at this scale, and")
        lines.append("      no arm comparison below should be read as a path effect.")

    if abs(gap) < FALSIFY_BELOW:
        lines.append("")
        lines.append(f"P1 gap reproduces: **NO** (|gap| {abs(gap):.3f} < {FALSIFY_BELOW})")
        lines.append("   ** CODE-PATH HYPOTHESIS REJECTED. Under identical cells, tubes,")
        lines.append("      hyperparameters and attribution, the two paths agree. §38.5's")
        lines.append("      correlation was confounded, and the cause of the Oesinghaus-90")
        lines.append("      signature collapse is once again UNKNOWN.")
    elif gap >= GAP_REPRODUCES and not overlap:
        lines.append("")
        lines.append(f"P1 gap reproduces: YES (gap {gap:+.3f} >= {GAP_REPRODUCES}, "
                     "seed ranges disjoint)")
        if {"cm_cd", "cd_cm"} <= have:
            t1 = float(m.loc["cm_cd", "mean_jaccard"])   # cm encoder + cd stage2
            t2 = float(m.loc["cd_cm", "mean_jaccard"])   # cd encoder + cm stage2
            lines.append("")
            lines.append(f"BISECTION  cm_cd (cm encoder, cd stage2) = {t1:.3f} -> looks "
                         f"{_closer_to(t1, p_mean, c_mean)}")
            lines.append(f"           cd_cm (cd encoder, cm stage2) = {t2:.3f} -> looks "
                         f"{_closer_to(t2, p_mean, c_mean)}")
            a, b = _closer_to(t1, p_mean, c_mean), _closer_to(t2, p_mean, c_mean)
            if a == "cm" and b == "cd":
                lines.append("   => the arms follow their ENCODER: the difference is in STAGE 1.")
            elif a == "cd" and b == "cm":
                lines.append("   => the arms follow their HEAD trainer: it is in STAGE 2.")
            else:
                lines.append("   => the transplants do not follow either component cleanly —")
                lines.append("      the effect is distributed or an interaction; bisect within.")
        else:
            lines.append("   (transplant arms absent — bisection not available)")
    else:
        lines.append("")
        lines.append(f"P1 gap reproduces: PARTIAL (gap {gap:+.3f}, seed ranges "
                     f"{'overlap' if overlap else 'disjoint'})")
        lines.append("   A gap is present but does not meet the pre-registered bar. Report")
        lines.append("   the magnitude and the seed spread; do not call it confirmed.")

    verdict = "\n".join(lines)
    print("\n" + verdict + "\n")
    (out / "PHASE2_VERDICT.md").write_text(
        "# Phase 1+2 — controlled code-path comparison\n\n"
        f"Panel: published-24, top-{args.top_n}. Seeds: {C.SEEDS}.\n\n"
        "```\n" + verdict + "\n```\n\n"
        "## Arm summary\n\n" + agg.to_markdown(index=False) +
        "\n\n## Per seed\n\n" + per_seed[
            ["arm", "seed", "mean_jaccard", "distinct_genes"]].to_markdown(index=False) + "\n"
    )
    print(f"[write] {out/'PHASE2_VERDICT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
