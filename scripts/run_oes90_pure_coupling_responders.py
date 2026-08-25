"""Track 3 — re-run PURE coupling over the RESPONDER subset only.

Why a re-run and not a filter
-----------------------------
The degree (hub) correction double-centres the coupling matrix, so every entry depends on
the row and column means of whatever matrix was passed in. Dropping conditions after the
fact would leave the degree terms computed over the contaminated 90-condition matrix.
The BH family size changes too. So the subset has to go back through the orchestrator.

`cross_asym` is pairwise — it depends only on (a, b, PBS, S_a, S_b) — so the direction
numbers are IDENTICAL on the subset and are deliberately not recomputed here. Read them
from the existing `direction_table.csv`.

What "responder" means here
---------------------------
A condition is a RESPONDER when its own cells score ABOVE the PBS baseline on its own
signature — i.e. self-engagement s(x, S_x) > 0 — in at least (1 - max_frac_negative) of
the cell types where it is measured. s(x, S_x) is read from `sA_PA_norm` / `sB_PB_norm`
in `engagement_per_celltype.parquet`, which cascadir's `directional_asymmetry_test`
emitted; it is a pure function of (condition, cell type), so it is looked up by dedup,
never recomputed. A condition failing this test has a signature that is anti-correlated
with its own response, which makes every `s(., S_x)` term built from it read backwards.

This is a QC gate on which conditions enter the analysis, not a change to any statistic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_pure_config as C  # noqa: E402
import _oes90_pure_estimator as E  # noqa: E402
from run_oes90_pure_coupling import bh_qvalues  # noqa: E402


def self_engagement_table(engagement_parquet) -> pd.DataFrame:
    """Per (condition, cell type) self-engagement s(x, S_x), by lookup.

    `sA_PA_norm` is s(a, S_a) - s(PBS, S_a) and does not depend on the partner b, so the
    same value is repeated across every pair a participates in. Deduplicating recovers one
    row per (condition, cell type) without arithmetic.
    """
    e = pd.read_parquet(engagement_parquet)
    a = e[["condition_a", "cell_type", "sA_PA_norm"]].rename(
        columns={"condition_a": "cytokine", "sA_PA_norm": "self_engagement"}
    )
    b = e[["condition_b", "cell_type", "sB_PB_norm"]].rename(
        columns={"condition_b": "cytokine", "sB_PB_norm": "self_engagement"}
    )
    both = pd.concat([a, b], ignore_index=True)
    n_distinct = both.groupby(["cytokine", "cell_type"])["self_engagement"].nunique()
    if int(n_distinct.max()) != 1:
        raise AssertionError(
            "self-engagement is not constant per (condition, cell type) — the lookup "
            "assumption is violated; refusing to derive a responder set from it."
        )
    return both.drop_duplicates(["cytokine", "cell_type"]).reset_index(drop=True)


def responder_set(se: pd.DataFrame, max_frac_negative: float):
    g = se.assign(neg=se.self_engagement < 0).groupby("cytokine")
    summary = pd.DataFrame({
        "frac_celltypes_negative": g["neg"].mean(),
        "median_self_engagement": g["self_engagement"].median(),
        "n_cell_types": g["neg"].size(),
    }).reset_index()
    summary["responder"] = summary.frac_celltypes_negative <= max_frac_negative
    return summary


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--alpha", type=float, default=C.COUPLING_ALPHA)
    ap.add_argument("--max_frac_negative", type=float, default=0.25,
                    help="max fraction of cell types with s(x,S_x) < 0 for a responder")
    ap.add_argument("--result_name", default="coupling_donor_degree_responders.csv")
    args = ap.parse_args()

    out = Path(args.out_dir)
    se = self_engagement_table(out / "engagement_per_celltype.parquet")
    summary = responder_set(se, args.max_frac_negative)
    summary.to_csv(out / "responder_status.csv", index=False)

    keep = sorted(summary.loc[summary.responder, "cytokine"])
    dropped = sorted(summary.loc[~summary.responder, "cytokine"])
    C.log(f"[gate] responders: {len(keep)}   non-responders dropped: {len(dropped)}")
    C.log(f"[gate] dropped: {', '.join(dropped)}")
    if len(keep) < 10:
        raise SystemExit(f"only {len(keep)} responders — refusing to run a degenerate fit.")

    est, prov = E.build_estimator(out, conditions=keep)
    n_donors = len(prov["donors"])
    C.log(f"[gate] donor_level=True (n_donors={n_donors}), degree_correct=True")

    t0 = time.time()
    coupling = est.signature_coupling(
        donor_level=True, coupling_alpha=args.alpha, degree_correct=True
    )
    C.log(f"[coupling] {len(coupling)} pairs in {(time.time()-t0)/60:.1f} min")

    if "donor_sign_p" not in coupling.columns:
        raise AssertionError(
            "coupling table has no `donor_sign_p` — the donor-level gate did not run "
            "(CLAUDE.md §28.2). Refusing to write."
        )

    coupling["q_donor"] = bh_qvalues(coupling["donor_sign_p"].to_numpy())
    for q in C.FDR_QS:
        tag = f"coupled_q{int(round(q*100)):02d}"
        coupling[tag] = (coupling["q_donor"] <= q) & (coupling["donor_coupling_mean"] > 0)

    dest = out / args.result_name
    coupling.to_csv(dest, index=False)

    n = len(coupling)
    C.log(f"\n[gates] over the {n}-pair responder-only family:")
    for q in C.FDR_QS:
        tag = f"coupled_q{int(round(q*100)):02d}"
        k = int(coupling[tag].sum())
        C.log(f"  BH q<={q:.2f}: {k} coupled ({k/n:.1%})")

    C.write_json(out / "coupling_responders_meta.json", {
        **prov,
        "n_pairs": n,
        "max_frac_negative": args.max_frac_negative,
        "n_responders": len(keep),
        "responders": keep,
        "dropped": dropped,
        "donor_level": True,
        "degree_correct": True,
        **{
            f"n_coupled_q{int(round(q*100)):02d}": int(
                coupling[f"coupled_q{int(round(q*100)):02d}"].sum()
            )
            for q in C.FDR_QS
        },
        "elapsed_s": round(time.time() - t0, 1),
    })
    C.mark_done(out, "coupling_responders")
    C.log(f"\n[done] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
