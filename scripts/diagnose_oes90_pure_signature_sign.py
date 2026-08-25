"""Track 1 — is each PURE-run signature an UP-program or a DOWN-program?

Diagnosis only. Nothing here changes how a signature is derived, and nothing here
computes coupling or direction — those stay on the cascadir orchestrator.

Why this exists
---------------
`cross_asym(a, b) = s(a, S_b) - s(b, S_a)` scores a gene set as *mean expression minus
the PBS mean*, which silently assumes S_X is a set of genes that go UP under X. But
`derive_signature` ranks by signed Integrated Gradients, and IG attributes *evidence for
the class*: a gene whose expression is BELOW the PBS baseline gets positive attribution
whenever the classifier weights it negatively. Down-regulated genes can therefore enter
the top-N legitimately, and the engagement score then reads them backwards.

The PURE fit shows exactly that symptom -- 36 of 90 conditions have negative median
self-engagement s(x, S_x), i.e. their own cells score BELOW PBS on their own signature.
This script resolves that to the individual gene: for every condition, how many of its
top-N genes actually go up, and does IG rank track the direction of change at all.

Reuses `run_signature_ablation.de_matrix_from_cells` (the repo's existing DE-vs-PBS
helper) rather than introducing a second differential-expression implementation.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _oes90_pure_config as C  # noqa: E402
import _oes90_pure_estimator as E  # noqa: E402
from run_signature_ablation import de_matrix_from_cells  # noqa: E402


def cells_by_condition_celltype(tube_set) -> dict:
    """Regroup a PseudoTubeSet into {(condition, cell_type) -> (n_cells, n_genes)}.

    Pure reshaping — the same contract `de_matrix_from_cells` expects.
    """
    buckets = defaultdict(list)
    for t in tube_set.tubes:
        ct = np.asarray(t.cell_types)
        for cell_type in np.unique(ct):
            rows = t.X[ct == cell_type]
            if len(rows):
                buckets[(str(t.condition), str(cell_type))].append(rows)
    return {k: np.vstack(v) for k, v in buckets.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--signatures", default="signatures_main.parquet")
    ap.add_argument("--result_name", default="signature_sign_diagnosis.csv")
    args = ap.parse_args()

    C.assert_agnostic()
    out = Path(args.out_dir)

    tube_set, _ = E.load_tubes(out, which="main")
    signatures = E.load_signatures(out / args.signatures)
    genes = list(tube_set.gene_names)

    cells = cells_by_condition_celltype(tube_set)
    C.log(f"[cells] {len(cells)} (condition, cell type) groups")

    conds = sorted(c for c in signatures if c != C.CONTROL)
    de = de_matrix_from_cells(cells, conds, genes, pbs_label=C.CONTROL)
    C.log(f"[de] delta-expression matrix {de.shape} (condition x gene), pooled over cell types")

    rows = []
    for cond in conds:
        sig = signatures[cond]
        present = [g for g in sig.genes if g in de.columns]
        d = de.loc[cond, present].to_numpy(dtype=float)
        ig = np.asarray(sig.ig_scores[: len(present)], dtype=float)
        n_up = int((d > 0).sum())
        # does IG rank order the genes by how much they actually move?
        rank_ig = np.arange(len(present))
        rho = (
            float(pd.Series(rank_ig).corr(pd.Series(d), method="spearman"))
            if len(present) > 2
            else np.nan
        )
        rows.append(
            {
                "cytokine": cond,
                "n_genes": len(present),
                "n_up": n_up,
                "frac_up": n_up / len(present) if present else np.nan,
                "mean_delta": float(d.mean()),
                "median_delta": float(np.median(d)),
                "mean_delta_up_only": float(d[d > 0].mean()) if n_up else np.nan,
                "spearman_igrank_vs_delta": rho,
                "mean_ig": float(ig.mean()),
                # how many genes a "keep only up-regulated" filter would leave
                "n_survive_up_filter": n_up,
            }
        )

    df = pd.DataFrame(rows).sort_values("frac_up")
    dest = out / args.result_name
    df.to_csv(dest, index=False)
    C.log(f"[write] {dest}  ({len(df)} conditions)")

    n_mostly_down = int((df.frac_up < 0.5).sum())
    C.log(
        f"[summary] conditions whose signature is majority DOWN-regulated: "
        f"{n_mostly_down}/{len(df)}"
    )
    C.log(
        f"[summary] median frac_up = {df.frac_up.median():.3f}; "
        f"median genes surviving an up-only filter = {df.n_survive_up_filter.median():.0f} "
        f"of {C.TOP_N}"
    )
    C.log("[summary] 10 most DOWN-dominated signatures:")
    for _, r in df.head(10).iterrows():
        C.log(f"    {r.cytokine:16s} frac_up={r.frac_up:.2f}  mean_delta={r.mean_delta:+.5f}")
    C.mark_done(out, "signature_sign")


if __name__ == "__main__":
    main()
