"""
End-to-end Path A → Bridge → Path B pipeline driver.

Chains the three methods built in this project:

  Path A  : latent geometry → unordered cytokine pairs
            (input  : reports/cascade_pairs/cytokine_axes.csv)

  Bridge  : binary AB-MIL IG → per-cytokine discovered gene set S_X^binary
            (input  : results/.../binary_ig/binary_ig.parquet)

  Path B  : §24 directional asymmetry test
            (reuses cytokine_mil.analysis.pathway_audit.directional_asymmetry_test;
             pathway_idx_dict is keyed by cytokine name and indexes top-N IG
             gene columns)

For each evaluable axis (A, B) we call directional_asymmetry_test twice
(forward A→B and reverse B→A) per cell type. We compare the sign of the
aggregated directional_score against the literature direction column in
the axes CSV.

Allowed dependencies (must match the plan's defensive-imports list):
  argparse, json, sys, time, traceback, pathlib, collections, typing,
  numpy, pandas, pyarrow (via pandas.read_parquet), torch (transitive),
  anndata (transitive via scanpy in pathway_audit's loader).

NO tabulate, matplotlib, seaborn, scipy.stats.

Usage:
  python scripts/run_pipeline_a_bridge_b.py \\
    --axes_csv reports/cascade_pairs/cytokine_axes.csv \\
    --binary_ig_parquet results/gene_dynamics_phase0/binary_ig/binary_ig.parquet \\
    --manifest_path /cs/.../Oesinghaus_pseudotubes/manifest.json \\
    --hvg_path /cs/.../Oesinghaus_pseudotubes/hvg_list.json \\
    --output_dir results/gene_dynamics_phase0/pipeline_a_b \\
    --top_n 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.oesinghaus_cell_loader import (  # noqa: E402
    load_oesinghaus_cells_by_pair,
)
from cytokine_mil.analysis.pathway_audit import (  # noqa: E402
    directional_asymmetry_test,
)


# ----------------------------------------------------------------------------
# Defaults (cluster paths — overridable via CLI)
# ----------------------------------------------------------------------------

AXES_CSV_DEFAULT = (
    REPO_ROOT / "reports" / "cascade_pairs" / "cytokine_axes.csv"
)
BINARY_IG_PARQUET_DEFAULT = (
    REPO_ROOT
    / "results"
    / "gene_dynamics_phase0"
    / "binary_ig"
    / "binary_ig.parquet"
)
MANIFEST_PATH_DEFAULT = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
)
HVG_PATH_DEFAULT = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
)


# ----------------------------------------------------------------------------
# CLI + setup
# ----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--axes_csv", default=str(AXES_CSV_DEFAULT))
    p.add_argument("--binary_ig_parquet", default=str(BINARY_IG_PARQUET_DEFAULT))
    p.add_argument("--manifest_path", default=MANIFEST_PATH_DEFAULT)
    p.add_argument("--hvg_path", default=HVG_PATH_DEFAULT)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_n", type=int, default=50,
                   help="Top N genes by binary IG to use as S_X")
    p.add_argument("--pbs_label", default="PBS")
    p.add_argument("--min_cells", type=int, default=10,
                   help="Min cells per (cyt, cell_type) for §24")
    p.add_argument("--max_tubes_per_cytokine", type=int, default=None,
                   help="Cap on tubes per cytokine (smoke runs)")
    p.add_argument(
        "--restrict_axes_to", nargs="+", default=None,
        help="If given, only run the axes whose unordered pair lex-key matches "
             "any of the supplied strings of form 'A__B' (sorted lex). Smoke flag.",
    )
    p.add_argument(
        "--include_donors", nargs="+", default=None,
        help="If set, only manifest entries from these donors are loaded. "
             "Mutually exclusive with --exclude_donors.",
    )
    p.add_argument(
        "--exclude_donors", nargs="+", default=None,
        help="If set, manifest entries from these donors are dropped. "
             "Use --exclude_donors Donor2 Donor3 to restrict §24 to the "
             "10 train donors (consistent with the binary MIL training split).",
    )
    return p.parse_args()


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _check_required_imports() -> None:
    """Defensive: fail fast if biovenv is missing a hard dep."""
    for required in ("numpy", "pandas", "pyarrow", "scanpy", "anndata"):
        try:
            __import__(required)
        except ImportError as e:
            print(
                f"FATAL: missing dependency '{required}': {e}", flush=True,
            )
            sys.exit(2)


def _md_table(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    """Manual GitHub-flavoured markdown table — avoids the tabulate dep."""
    cols = columns or list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows: List[str] = []
    for _, row in df.iterrows():
        cells: List[str] = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append("NaN" if np.isnan(v) else f"{v:+.4f}")
            elif isinstance(v, bool):
                cells.append("True" if v else "False")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


# ----------------------------------------------------------------------------
# Step 1: load Path A axes and filter to evaluable rows
# ----------------------------------------------------------------------------

def _load_axes(
    axes_csv: str, covered_cytokines: set, restrict_axes_to: Optional[List[str]],
) -> pd.DataFrame:
    df = pd.read_csv(axes_csv)
    df = df.copy()

    def _evaluable(row: pd.Series) -> bool:
        return (row["axis_a"] in covered_cytokines
                and row["axis_b"] in covered_cytokines)

    df["evaluable"] = df.apply(_evaluable, axis=1)
    df["lex_key"] = df.apply(
        lambda r: "__".join(sorted([r["axis_a"], r["axis_b"]])),
        axis=1,
    )
    df_ev = df[df["evaluable"]].copy()
    if restrict_axes_to is not None:
        restrict_set = set(restrict_axes_to)
        df_ev = df_ev[df_ev["lex_key"].isin(restrict_set)].copy()
    return df_ev


# ----------------------------------------------------------------------------
# Step 2: build S_X^binary per cytokine from binary_ig.parquet
# ----------------------------------------------------------------------------

def _build_s_x_per_cytokine(
    binary_ig_parquet: str, top_n: int,
) -> Tuple[Dict[str, List[str]], set]:
    """
    Returns:
        s_x_genes: {cytokine -> [gene_name, ...]} top-N by rank_ig
        covered:   set of cytokines present in binary_ig.parquet
    """
    df = pd.read_parquet(binary_ig_parquet)
    needed_cols = {"cytokine", "gene", "rank_ig"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"binary_ig.parquet missing columns: {missing}. Found: {list(df.columns)}"
        )
    s_x_genes: Dict[str, List[str]] = {}
    for cyt, sub in df.groupby("cytokine"):
        sub_sorted = sub.sort_values("rank_ig")
        s_x_genes[cyt] = sub_sorted["gene"].head(top_n).tolist()
    return s_x_genes, set(s_x_genes)


def _build_pathway_idx_dict(
    s_x_genes: Dict[str, List[str]], gene_names: List[str],
) -> Dict[str, np.ndarray]:
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    out: Dict[str, np.ndarray] = {}
    for cyt, genes in s_x_genes.items():
        idx = np.array(
            [gene_to_idx[g] for g in genes if g in gene_to_idx],
            dtype=np.int64,
        )
        out[cyt] = idx
    return out


# ----------------------------------------------------------------------------
# Step 4+5: per-axis directional test and aggregation
# ----------------------------------------------------------------------------

def _expected_sign(lit_dir: str) -> Optional[int]:
    """Return +1 (positive directional_score expected), -1, or None."""
    if not isinstance(lit_dir, str):
        return None
    lit = lit_dir.strip().lower()
    if lit == "a_to_b":
        return +1
    if lit == "b_to_a":
        return -1
    return None


def _evaluate_one_axis(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    pathway_idx_dict: Dict[str, np.ndarray],
    axis_a: str,
    axis_b: str,
    pbs_label: str,
    min_cells: int,
) -> Dict[str, object]:
    """
    Runs §24 in both directions and aggregates a per-axis summary.

    Returns a dict with:
        per_celltype_AB    DataFrame (§24 output for A=axis_a, B=axis_b)
        per_celltype_BA    DataFrame (§24 output for A=axis_b, B=axis_a)
        overall_score_AB   mean over cell_types of directional_score (AB)
        overall_score_BA   mean over cell_types of directional_score (BA)
        antisymmetry_check |overall_score_AB + overall_score_BA|
                           — should be near 0 if §24 is well-behaved
    """
    df_AB = directional_asymmetry_test(
        cells_by_pair=cells_by_pair,
        pathway_idx_dict=pathway_idx_dict,
        A=axis_a, B=axis_b, P_A=axis_a, P_B=axis_b,
        pbs_label=pbs_label, min_cells=min_cells,
    )
    df_BA = directional_asymmetry_test(
        cells_by_pair=cells_by_pair,
        pathway_idx_dict=pathway_idx_dict,
        A=axis_b, B=axis_a, P_A=axis_b, P_B=axis_a,
        pbs_label=pbs_label, min_cells=min_cells,
    )
    if not df_AB.empty:
        overall_AB = float(df_AB["directional_score"].mean())
    else:
        overall_AB = float("nan")
    if not df_BA.empty:
        overall_BA = float(df_BA["directional_score"].mean())
    else:
        overall_BA = float("nan")
    if np.isnan(overall_AB) or np.isnan(overall_BA):
        antisymmetry = float("nan")
    else:
        antisymmetry = float(abs(overall_AB + overall_BA))
    return {
        "per_celltype_AB": df_AB,
        "per_celltype_BA": df_BA,
        "overall_score_AB": overall_AB,
        "overall_score_BA": overall_BA,
        "antisymmetry_check": antisymmetry,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _check_required_imports()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pre-flight validation ----
    for p_arg, p_val in [
        ("axes_csv", args.axes_csv),
        ("binary_ig_parquet", args.binary_ig_parquet),
        ("manifest_path", args.manifest_path),
        ("hvg_path", args.hvg_path),
    ]:
        if not Path(p_val).exists():
            _log(f"FATAL: --{p_arg} does not exist: {p_val}")
            sys.exit(2)

    _log("=" * 70)
    _log("Path A -> Bridge -> Path B pipeline")
    _log("=" * 70)
    _log(f"axes_csv:            {args.axes_csv}")
    _log(f"binary_ig_parquet:   {args.binary_ig_parquet}")
    _log(f"manifest_path:       {args.manifest_path}")
    _log(f"hvg_path:            {args.hvg_path}")
    _log(f"output_dir:          {output_dir}")
    _log(f"top_n:               {args.top_n}")
    _log(f"min_cells:           {args.min_cells}")
    _log(f"max_tubes/cytokine:  {args.max_tubes_per_cytokine}")
    _log(f"restrict_axes_to:    {args.restrict_axes_to}")
    _log(f"include_donors:      {args.include_donors}")
    _log(f"exclude_donors:      {args.exclude_donors}")
    _log("")

    # ---- Step 2: load S_X^binary per cytokine ----
    t0 = time.time()
    s_x_genes, covered_cyts = _build_s_x_per_cytokine(
        args.binary_ig_parquet, args.top_n,
    )
    if len(covered_cyts) < 2:
        _log(f"FATAL: binary_ig.parquet covers only {covered_cyts}; need >= 2.")
        sys.exit(2)
    _log(f"[step 2] S_X built for {len(covered_cyts)} cytokines: {sorted(covered_cyts)} "
         f"(top_n={args.top_n})  elapsed={time.time()-t0:.1f}s")

    # ---- Step 1: filter axes ----
    t0 = time.time()
    axes_df = _load_axes(args.axes_csv, covered_cyts, args.restrict_axes_to)
    if axes_df.empty:
        _log("FATAL: no evaluable axes after filtering.")
        sys.exit(2)
    _log(f"[step 1] evaluable axes: {len(axes_df)}  elapsed={time.time()-t0:.1f}s")
    for _, row in axes_df.iterrows():
        _log(f"           {row['axis_a']} -- {row['axis_b']}  "
             f"lit={row['literature_status']}  lit_dir={row['literature_direction']}")

    # ---- Step 3: load cells ----
    needed_cyts = set()
    for _, row in axes_df.iterrows():
        needed_cyts.add(row["axis_a"])
        needed_cyts.add(row["axis_b"])
    _log(f"\n[step 3] loading Oesinghaus cells for cytokines: "
         f"{sorted(needed_cyts)} (+ {args.pbs_label})")
    t0 = time.time()
    cells_by_pair, gene_names = load_oesinghaus_cells_by_pair(
        manifest_path=args.manifest_path,
        cytokines=sorted(needed_cyts),
        hvg_path=args.hvg_path,
        max_tubes_per_cytokine=args.max_tubes_per_cytokine,
        pbs_label=args.pbs_label,
        include_donors=args.include_donors,
        exclude_donors=args.exclude_donors,
    )
    _log(f"           loaded {len(cells_by_pair)} (cyt, cell_type) groups, "
         f"{len(gene_names)} genes  elapsed={time.time()-t0:.1f}s")

    pathway_idx_dict = _build_pathway_idx_dict(s_x_genes, gene_names)
    _log("           pathway_idx_dict sizes: "
         + ", ".join(f"{c}={len(pathway_idx_dict[c])}" for c in sorted(needed_cyts)))

    # ---- Step 4+5: per-axis test ----
    per_celltype_path = output_dir / "per_celltype.csv"
    per_axis_path = output_dir / "per_axis_summary.csv"

    per_axis_rows: List[Dict[str, object]] = []
    n_ground_truth = 0
    n_correct = 0
    n_axes = len(axes_df)

    # Stream incremental writes — write headers on first write, then append.
    first_celltype_write = True

    for i, (_, row) in enumerate(axes_df.iterrows(), start=1):
        A = row["axis_a"]
        B = row["axis_b"]
        lit_status = row["literature_status"]
        lit_dir = row.get("literature_direction", "no_lit")
        expected = _expected_sign(lit_dir)

        _log(f"\n[{i}/{n_axes}] {A} -- {B}  "
             f"lit={lit_status}  lit_dir={lit_dir}  START")
        t_axis = time.time()
        try:
            res = _evaluate_one_axis(
                cells_by_pair=cells_by_pair,
                pathway_idx_dict=pathway_idx_dict,
                axis_a=A,
                axis_b=B,
                pbs_label=args.pbs_label,
                min_cells=args.min_cells,
            )
        except Exception as e:  # defensive — one bad axis does not kill the run
            _log(f"           ERROR: {type(e).__name__}: {e}")
            _log(traceback.format_exc())
            per_axis_rows.append({
                "axis_a": A, "axis_b": B,
                "literature_status": lit_status,
                "literature_direction": lit_dir,
                "expected_sign": expected,
                "overall_score_AB": float("nan"),
                "overall_score_BA": float("nan"),
                "antisymmetry_check": float("nan"),
                "sign_correct": False,
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        # Tag per-cell-type rows with the axis identity for the long CSV
        for direction_tag, sub_df in (("AB", res["per_celltype_AB"]),
                                      ("BA", res["per_celltype_BA"])):
            if sub_df.empty:
                continue
            sub = sub_df.copy()
            sub.insert(0, "direction_tag", direction_tag)
            sub.insert(0, "axis_b", B)
            sub.insert(0, "axis_a", A)
            sub.insert(0, "literature_direction", lit_dir)
            sub.insert(0, "literature_status", lit_status)
            sub.to_csv(
                per_celltype_path,
                mode="w" if first_celltype_write else "a",
                header=first_celltype_write,
                index=False,
            )
            first_celltype_write = False

        # Per-axis summary row
        score_AB = res["overall_score_AB"]
        sign_correct = False
        if expected is not None and not np.isnan(score_AB):
            sign_correct = (np.sign(score_AB) == expected)
            n_ground_truth += 1
            if sign_correct:
                n_correct += 1
        per_axis_rows.append({
            "axis_a": A, "axis_b": B,
            "literature_status": lit_status,
            "literature_direction": lit_dir,
            "expected_sign": expected,
            "overall_score_AB": score_AB,
            "overall_score_BA": res["overall_score_BA"],
            "antisymmetry_check": res["antisymmetry_check"],
            "sign_correct": sign_correct,
            "error": "",
        })

        # Incremental write of per_axis_summary
        pd.DataFrame(per_axis_rows).to_csv(per_axis_path, index=False)

        elapsed = time.time() - t_axis
        _log(f"           overall_score_AB={score_AB:+.4f}  "
             f"overall_score_BA={res['overall_score_BA']:+.4f}  "
             f"antisymmetry={res['antisymmetry_check']:.4f}  "
             f"sign_correct={sign_correct}  elapsed={elapsed:.1f}s  DONE")

    # ---- verdict.md ----
    per_axis_df = pd.DataFrame(per_axis_rows)
    verdict_lines = [
        "# Pipeline A → Bridge → Path B — directional verdict",
        "",
        f"**axes_csv:**            `{args.axes_csv}`",
        f"**binary_ig_parquet:**   `{args.binary_ig_parquet}`",
        f"**top_n:**               {args.top_n}",
        f"**evaluable axes:**      {len(per_axis_df)}",
        f"**include_donors:**      {args.include_donors}",
        f"**exclude_donors:**      {args.exclude_donors}",
        "",
        f"## Ground-truth sign accuracy: **{n_correct} / {n_ground_truth}**",
        "",
        ("All ground-truth axes correct → method works on this dataset; queue "
         "training of 16 missing cytokines for full 19-axis validation."
         if n_ground_truth > 0 and n_correct == n_ground_truth
         else "Some ground-truth axes failed → inspect per_celltype.csv for "
              "the failing axis and decide between debug vs. methodology change."),
        "",
        "## Per-axis summary",
        "",
        _md_table(
            per_axis_df,
            columns=[
                "axis_a", "axis_b", "literature_direction",
                "expected_sign",
                "overall_score_AB", "overall_score_BA",
                "antisymmetry_check", "sign_correct",
            ],
        ),
        "",
        "## Notes",
        "",
        "- `overall_score_AB` is the mean of `directional_score` across cell types "
        "with §24 called as `(A=axis_a, B=axis_b, P_A=axis_a, P_B=axis_b)`.",
        "- `overall_score_BA` is the reverse call; |AB + BA| (antisymmetry_check) "
        "should be near 0 if §24 is well-behaved.",
        "- `expected_sign`: +1 if literature says a_to_b, -1 if b_to_a, blank for "
        "NOVEL / no_lit axes (those are reported for inspection but not scored).",
        f"- §24 min_cells = {args.min_cells}.  S_X size = top_{args.top_n} genes "
        "per cytokine from `binary_ig.parquet`.",
    ]
    (output_dir / "verdict.md").write_text("\n".join(verdict_lines) + "\n")
    _log(f"\nWrote {output_dir / 'verdict.md'}")
    _log(f"\nOVERALL: {n_correct} / {n_ground_truth} ground-truth axes "
         f"have sign matching literature.")


if __name__ == "__main__":
    main()
