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

from cytokine_mil.analysis.eda_pair_benchmark import (  # noqa: E402
    load_phase1_cells,
)
from cytokine_mil.analysis.pathway_audit import (  # noqa: E402
    directional_asymmetry_test,
)


# ----------------------------------------------------------------------------
# Defaults (cluster paths — overridable via CLI)
# ----------------------------------------------------------------------------
# SHEU variant of run_pipeline_a_bridge_b.py. The ONLY differences vs the
# Oesinghaus driver are: (1) the cell loader is eda_pair_benchmark.load_phase1_cells
# (Sheu-native, supports per-time-point filtering); (2) a required --time_filter
# selects ONE time frame's stimulated cells (single-frame cascade detection —
# no cross-time leakage). Everything else (cross_asym primary metric, null
# control, median+consensus) is identical.

AXES_CSV_DEFAULT = (
    REPO_ROOT / "reports" / "sheu_cascade" / "sheu_axes_labeled.csv"
)
BINARY_IG_PARQUET_DEFAULT = (
    REPO_ROOT / "results" / "sheu_cascade" / "binary_ig.parquet"
)
MANIFEST_PATH_DEFAULT = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json"
)
HVG_PATH_DEFAULT = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json"
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
    p.add_argument("--magnitude_threshold", type=float, default=0.01,
                   help="Min |median directional_score| to be a non-AMBIGUOUS call")
    p.add_argument("--strong_consensus", type=float, default=0.75,
                   help="Min sign_consensus for a STRONG call")
    p.add_argument("--weak_consensus", type=float, default=0.50,
                   help="Min sign_consensus for a WEAK call")
    p.add_argument("--n_null_perms", type=int, default=100,
                   help="Per-axis null permutations with random S_A/S_B "
                        "drawn from HVGs disjoint from observed S_X union. "
                        "Set 0 to disable.")
    p.add_argument("--null_seed", type=int, default=42)
    p.add_argument(
        "--time_filter", default=None,
        help="REQUIRED for single-frame detection: stimulated cells whose "
             "obs.time_point != this value are dropped (e.g. '3hr'). PBS "
             "(0h-pooled) is always kept in full. Omit only for an already "
             "time-filtered manifest.",
    )
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


def _aggregate_metric(scores: np.ndarray) -> Tuple[float, float, int, int]:
    """Median + signed-consensus for one metric's per-cell-type values.

    median    : robust to single-cell-type domination (mean aggregation broke
                IL-6/TNF-α on Oesinghaus — CD14 Mono dominated).
    consensus : fraction of cell types whose sign matches the median's sign;
                for median==0, fraction of exactly-zero rows.
    """
    if len(scores) == 0:
        return float("nan"), float("nan"), 0, 0
    med = float(np.median(scores))
    n_pos = int(np.sum(scores > 0))
    n_neg = int(np.sum(scores < 0))
    n_total = len(scores)
    if med == 0.0:
        consensus = float(np.sum(scores == 0) / n_total)
    elif med > 0:
        consensus = float(n_pos / n_total)
    else:
        consensus = float(n_neg / n_total)
    return med, consensus, n_pos, n_neg


def _cross_asym_series(df_per_celltype: pd.DataFrame) -> np.ndarray:
    """
    cross_asym = s(axis_a, S_axis_b) − s(axis_b, S_axis_a), PBS-normalised.

    In the §24 output, axis_a is A (P_A = S_axis_a) and axis_b is B
    (P_B = S_axis_b), so:
        cross_asym = sA_PB_norm − sB_PA_norm

    This is the ANTISYMMETRIC cross-engagement: positive ⇒ A engages B's
    signature more than B engages A's ⇒ A is upstream ⇒ a_to_b. It is the
    direction-bearing quantity (unlike directional_score, which is symmetric
    in (a, b) and therefore direction-blind for self-signatures). Encodes the
    "X-tube carries both Sx and Sy; Y-tube carries only Sy" cascade model.
    """
    return (df_per_celltype["sA_PB_norm"]
            - df_per_celltype["sB_PA_norm"]).to_numpy(dtype=np.float64)


def _aggregate_directional(
    df_per_celltype: pd.DataFrame,
) -> Dict[str, Tuple[float, float, int, int]]:
    """
    Aggregate §24 per-cell-type DataFrame into BOTH metrics:

      "cross"    : cross_asym (ANTISYMMETRIC, direction-bearing — PRIMARY call)
      "dirscore" : directional_score (SYMMETRIC coupling measure — secondary,
                   kept for reference; its sign does not encode direction for
                   self-signatures, validated 47% vs 88% on the Oesinghaus
                   audited benchmark).

    Each value is (median, sign_consensus, n_pos, n_neg).
    """
    if df_per_celltype.empty:
        nan4 = (float("nan"), float("nan"), 0, 0)
        return {"cross": nan4, "dirscore": nan4}
    dscore = df_per_celltype["directional_score"].to_numpy(dtype=np.float64)
    cross = _cross_asym_series(df_per_celltype)
    return {"cross": _aggregate_metric(cross),
            "dirscore": _aggregate_metric(dscore)}


def _classify_call(
    median: float, consensus: float,
    magnitude_threshold: float, strong_consensus: float, weak_consensus: float,
) -> str:
    """STRONG / WEAK / AMBIGUOUS based on (|median|, consensus)."""
    if np.isnan(median) or np.isnan(consensus):
        return "UNDEFINED"
    if abs(median) < magnitude_threshold:
        return "AMBIGUOUS"
    if consensus >= strong_consensus:
        return "STRONG"
    if consensus >= weak_consensus:
        return "WEAK"
    return "AMBIGUOUS"


def _evaluate_one_axis(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    pathway_idx_dict: Dict[str, np.ndarray],
    axis_a: str,
    axis_b: str,
    pbs_label: str,
    min_cells: int,
) -> Dict[str, object]:
    """
    Run §24 once for the axis (A=axis_a, B=axis_b, P_A=axis_a, P_B=axis_b) and
    aggregate per-cell-type rows into a single (median, consensus) call.

    Note on the dropped reverse call: directional_score is mathematically
    invariant under the simultaneous swap (A↔B, P_A↔P_B), so the previous
    AB / BA pair was redundant (we verified |AB - BA| == 0 across all 3 axes
    on jobs 30701212 and 30701234). One forward call is sufficient.
    """
    df_AB = directional_asymmetry_test(
        cells_by_pair=cells_by_pair,
        pathway_idx_dict=pathway_idx_dict,
        A=axis_a, B=axis_b, P_A=axis_a, P_B=axis_b,
        pbs_label=pbs_label, min_cells=min_cells,
    )
    agg = _aggregate_directional(df_AB)
    cm, cc, cpos, cneg = agg["cross"]       # cross_asym — primary direction call
    dm, dc, dpos, dneg = agg["dirscore"]    # directional_score — secondary
    return {
        "per_celltype_AB": df_AB,
        # primary (cross_asym)
        "cross_median": cm,
        "cross_consensus": cc,
        "cross_n_pos": cpos,
        "cross_n_neg": cneg,
        # secondary (directional_score)
        "dirscore_median": dm,
        "dirscore_consensus": dc,
        "dirscore_n_pos": dpos,
        "dirscore_n_neg": dneg,
    }


def _evaluate_null_for_axis(
    cells_by_pair: Dict[Tuple[str, str], np.ndarray],
    n_genes: int,
    excluded_indices: set,
    size_A: int,
    size_B: int,
    axis_a: str,
    axis_b: str,
    pbs_label: str,
    min_cells: int,
    n_perms: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical null distribution under random S_A / S_B of the same sizes,
    drawn from HVGs disjoint from any observed S_X (excluded_indices).

    Tests whether the DISCOVERED gene sets carry cytokine-specific
    information: under the null hypothesis "any random gene set of the
    same size would give a similar median directional_score", the
    observed |median| should be in the tail of this distribution.

    Returns:
        cross_null_medians:    (n_perms,) per-permutation median cross_asym
        dirscore_null_medians: (n_perms,) per-permutation median directional_score
    """
    pool = np.array(
        [i for i in range(n_genes) if i not in excluded_indices],
        dtype=np.int64,
    )
    if len(pool) < max(size_A, size_B):
        raise ValueError(
            f"Null pool too small: {len(pool)} non-S_X HVGs available, "
            f"need {max(size_A, size_B)} for random gene sets."
        )
    cross_medians = np.full(n_perms, np.nan, dtype=np.float64)
    dirscore_medians = np.full(n_perms, np.nan, dtype=np.float64)
    for k in range(n_perms):
        idx_A = rng.choice(pool, size=size_A, replace=False)
        idx_B = rng.choice(pool, size=size_B, replace=False)
        null_pathway_dict = {axis_a: idx_A, axis_b: idx_B}
        df = directional_asymmetry_test(
            cells_by_pair=cells_by_pair,
            pathway_idx_dict=null_pathway_dict,
            A=axis_a, B=axis_b, P_A=axis_a, P_B=axis_b,
            pbs_label=pbs_label, min_cells=min_cells,
        )
        agg = _aggregate_directional(df)
        cross_medians[k] = agg["cross"][0]
        dirscore_medians[k] = agg["dirscore"][0]
    return cross_medians, dirscore_medians


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

    # ---- Step 3: load cells (SINGLE TIME FRAME via load_phase1_cells) ----
    needed_cyts = set()
    for _, row in axes_df.iterrows():
        needed_cyts.add(row["axis_a"])
        needed_cyts.add(row["axis_b"])
    if args.exclude_donors:
        _log("FATAL: --exclude_donors is not supported by the Sheu loader; "
             "use --include_donors instead.")
        sys.exit(2)
    with open(args.hvg_path) as fh:
        hvg_list = json.load(fh)
    _log(f"\n[step 3] loading Sheu cells (time_filter={args.time_filter}) for "
         f"cytokines: {sorted(needed_cyts)} (+ {args.pbs_label}); "
         f"{len(hvg_list)} panel genes")
    if args.time_filter is None:
        _log("           WARNING: --time_filter is None; using the manifest as-is. "
             "For single-frame detection pass --time_filter <T>hr.")
    t0 = time.time()
    cells_by_pair, gene_names = load_phase1_cells(
        manifest_path=args.manifest_path,
        gene_names=hvg_list,
        time_filter=args.time_filter,
        donors=args.include_donors,
    )
    _log(f"           loaded {len(cells_by_pair)} (cyt, cell_type) groups, "
         f"{len(gene_names)} genes  elapsed={time.time()-t0:.1f}s")

    pathway_idx_dict = _build_pathway_idx_dict(s_x_genes, gene_names)
    _log("           pathway_idx_dict sizes: "
         + ", ".join(f"{c}={len(pathway_idx_dict[c])}" for c in sorted(needed_cyts)))

    # ---- Step 4+5: per-axis test (with null control) ----
    per_celltype_path = output_dir / "per_celltype.csv"
    per_axis_path = output_dir / "per_axis_summary.csv"

    # Build the union of observed S_X indices for null sampling
    excluded_indices: set = set()
    for cyt_indices in pathway_idx_dict.values():
        excluded_indices.update(cyt_indices.tolist())
    n_genes_total = len(gene_names)
    n_genes_pool = n_genes_total - len(excluded_indices)
    _log(f"\n[null setup] excluded {len(excluded_indices)} / {n_genes_total} "
         f"HVGs (union of observed S_X across all cytokines); "
         f"null pool size = {n_genes_pool}")

    null_rng = np.random.default_rng(args.null_seed)

    per_axis_rows: List[Dict[str, object]] = []
    n_ground_truth = 0
    n_correct = 0
    n_axes = len(axes_df)
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
                axis_a=A, axis_b=B,
                pbs_label=args.pbs_label,
                min_cells=args.min_cells,
            )
        except Exception as e:  # defensive — one bad axis does not kill the run
            _log(f"           ERROR (observed): {type(e).__name__}: {e}")
            _log(traceback.format_exc())
            per_axis_rows.append({
                "axis_a": A, "axis_b": B,
                "literature_status": lit_status,
                "literature_direction": lit_dir,
                "expected_sign": expected,
                "cross_median": float("nan"),
                "cross_consensus": float("nan"),
                "cross_n_pos": 0,
                "cross_n_neg": 0,
                "classification": "UNDEFINED",
                "cross_sign_correct": False,
                "null_n_perms": 0,
                "cross_null_mean": float("nan"),
                "cross_null_q025": float("nan"),
                "cross_null_q975": float("nan"),
                "cross_p_emp_two_sided": float("nan"),
                "dirscore_median": float("nan"),
                "dirscore_consensus": float("nan"),
                "dirscore_sign_correct": False,
                "dirscore_p_emp_two_sided": float("nan"),
                "median_score": float("nan"),
                "sign_consensus": float("nan"),
                "sign_correct": False,
                "error": f"observed: {type(e).__name__}: {e}",
            })
            continue

        # Stream per-cell-type rows (forward only). Add the cross_asym column
        # so downstream (retally / cross-time compare) can re-aggregate.
        sub_df = res["per_celltype_AB"]
        if not sub_df.empty:
            sub = sub_df.copy()
            sub["cross_asym"] = sub["sA_PB_norm"] - sub["sB_PA_norm"]
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

        # PRIMARY direction call = cross_asym (antisymmetric, direction-bearing)
        med = res["cross_median"]
        cons = res["cross_consensus"]
        classification = _classify_call(
            med, cons,
            magnitude_threshold=args.magnitude_threshold,
            strong_consensus=args.strong_consensus,
            weak_consensus=args.weak_consensus,
        )
        # secondary metric (directional_score — symmetric coupling, kept for ref)
        ds_med = res["dirscore_median"]
        ds_cons = res["dirscore_consensus"]

        # ---- Null control (random S_A, S_B) — cross_asym primary, dirscore ref ----
        null_mean = float("nan")
        null_q025 = float("nan")
        null_q975 = float("nan")
        p_emp = float("nan")          # cross_asym empirical two-sided p
        ds_p_emp = float("nan")       # directional_score empirical two-sided p
        null_n_actual = 0
        null_err = ""
        if args.n_null_perms > 0 and not np.isnan(med):
            try:
                size_A = len(pathway_idx_dict[A])
                size_B = len(pathway_idx_dict[B])
                cross_null, dscore_null = _evaluate_null_for_axis(
                    cells_by_pair=cells_by_pair,
                    n_genes=n_genes_total,
                    excluded_indices=excluded_indices,
                    size_A=size_A, size_B=size_B,
                    axis_a=A, axis_b=B,
                    pbs_label=args.pbs_label,
                    min_cells=args.min_cells,
                    n_perms=args.n_null_perms,
                    rng=null_rng,
                )
                valid_null = cross_null[~np.isnan(cross_null)]
                null_n_actual = int(len(valid_null))
                if null_n_actual > 0:
                    null_mean = float(np.mean(valid_null))
                    null_q025 = float(np.quantile(valid_null, 0.025))
                    null_q975 = float(np.quantile(valid_null, 0.975))
                    p_emp = float(np.mean(np.abs(valid_null) >= abs(med)))
                valid_ds = dscore_null[~np.isnan(dscore_null)]
                if len(valid_ds) > 0 and not np.isnan(ds_med):
                    ds_p_emp = float(np.mean(np.abs(valid_ds) >= abs(ds_med)))
            except Exception as e:
                null_err = f"null: {type(e).__name__}: {e}"
                _log(f"           NULL ERROR: {null_err}")

        sign_correct = False
        ds_sign_correct = False
        if expected is not None and not np.isnan(med):
            sign_correct = (np.sign(med) == expected)
            if classification != "AMBIGUOUS":
                n_ground_truth += 1
                if sign_correct:
                    n_correct += 1
        if expected is not None and not np.isnan(ds_med):
            ds_sign_correct = (np.sign(ds_med) == expected)

        per_axis_rows.append({
            "axis_a": A, "axis_b": B,
            "literature_status": lit_status,
            "literature_direction": lit_dir,
            "expected_sign": expected,
            # PRIMARY = cross_asym
            "cross_median": med,
            "cross_consensus": cons,
            "cross_n_pos": res["cross_n_pos"],
            "cross_n_neg": res["cross_n_neg"],
            "classification": classification,
            "cross_sign_correct": sign_correct,
            "null_n_perms": null_n_actual,
            "cross_null_mean": null_mean,
            "cross_null_q025": null_q025,
            "cross_null_q975": null_q975,
            "cross_p_emp_two_sided": p_emp,
            # SECONDARY = directional_score
            "dirscore_median": ds_med,
            "dirscore_consensus": ds_cons,
            "dirscore_sign_correct": ds_sign_correct,
            "dirscore_p_emp_two_sided": ds_p_emp,
            # back-compat aliases (older consumers read these)
            "median_score": med,
            "sign_consensus": cons,
            "sign_correct": sign_correct,
            "error": null_err,
        })

        pd.DataFrame(per_axis_rows).to_csv(per_axis_path, index=False)

        elapsed = time.time() - t_axis
        _log(f"           cross_asym={med:+.4f}  consensus={cons:.2f}  "
             f"({res['cross_n_pos']}+/{res['cross_n_neg']}−)  "
             f"call={classification}  null_p={p_emp:.3f}  "
             f"[dirscore={ds_med:+.4f}]  "
             f"sign_correct={sign_correct}  elapsed={elapsed:.1f}s  DONE")

    # ---- verdict.md ----
    per_axis_df = pd.DataFrame(per_axis_rows)
    n_strong = int((per_axis_df["classification"] == "STRONG").sum())
    n_weak = int((per_axis_df["classification"] == "WEAK").sum())
    n_ambig = int((per_axis_df["classification"] == "AMBIGUOUS").sum())
    n_null_pass = int(
        ((per_axis_df["cross_p_emp_two_sided"] < 0.05)
         & (per_axis_df["classification"] != "AMBIGUOUS")).sum()
    )
    # directional_score (secondary metric) sign accuracy, for the comparison
    ds_gt = per_axis_df[per_axis_df["expected_sign"].notna()]
    ds_correct = int(ds_gt["dirscore_sign_correct"].sum()) if len(ds_gt) else 0
    ds_total = int(len(ds_gt))

    verdict_lines = [
        "# Pipeline A → Bridge → Path B — directional verdict",
        "",
        "**Primary metric = cross_asym** (antisymmetric cross-engagement, "
        "`s(a,S_b) − s(b,S_a)`, PBS-normalised). Its sign encodes direction: "
        "positive ⇒ axis_a upstream (a_to_b). `directional_score` (the §24 "
        "scalar) is reported as a SECONDARY reference — it is symmetric in "
        "(a,b) and does NOT encode direction for self-signatures.",
        "",
        f"**axes_csv:**            `{args.axes_csv}`",
        f"**binary_ig_parquet:**   `{args.binary_ig_parquet}`",
        f"**top_n:**               {args.top_n}",
        f"**evaluable axes:**      {len(per_axis_df)}",
        f"**include_donors:**      {args.include_donors}",
        f"**exclude_donors:**      {args.exclude_donors}",
        f"**n_null_perms:**        {args.n_null_perms}",
        "",
        "## Headline",
        "",
        f"- Classification breakdown (cross_asym): **{n_strong} STRONG, "
        f"{n_weak} WEAK, {n_ambig} AMBIGUOUS** (out of {len(per_axis_df)})",
        f"- **cross_asym** ground-truth sign accuracy (non-AMBIGUOUS only): "
        f"**{n_correct} / {n_ground_truth}**",
        f"- directional_score (secondary) sign accuracy (all graded axes): "
        f"**{ds_correct} / {ds_total}**",
        f"- cross_asym non-AMBIGUOUS axes also passing the null control "
        f"(p_emp < 0.05): **{n_null_pass}**",
        "",
        "## Per-axis summary (cross_asym primary, directional_score secondary)",
        "",
        _md_table(
            per_axis_df,
            columns=[
                "axis_a", "axis_b", "literature_direction",
                "expected_sign",
                "cross_median", "cross_consensus",
                "cross_n_pos", "cross_n_neg",
                "classification",
                "cross_p_emp_two_sided",
                "cross_sign_correct",
                "dirscore_median", "dirscore_sign_correct",
            ],
        ),
        "",
        "## Definitions",
        "",
        "- `cross_median`: median across cell types of "
        "`cross_asym = s(axis_a,S_axis_b) − s(axis_b,S_axis_a)` (PBS-normalised). "
        "Positive ⇒ axis_a engages axis_b's discovered signature more than the "
        "reverse ⇒ axis_a upstream ⇒ a_to_b. This is the direction-bearing call.",
        "- `cross_consensus`: fraction of cell types whose `cross_asym` matches "
        "the sign of `cross_median`.",
        "- `dirscore_median`: median of the §24 `directional_score` "
        "(= asym_PA − asym_PB). Symmetric in (a,b); kept only as a coupling-"
        "strength reference. On the Oesinghaus audited benchmark it scored "
        "8/17 (chance) vs cross_asym 15/17 — it cannot infer direction from "
        "self-signatures.",
        "- `classification` (on cross_asym):",
        f"    - **STRONG**: `|cross_median| ≥ {args.magnitude_threshold}` AND "
        f"`consensus ≥ {args.strong_consensus}`",
        f"    - **WEAK**: `|cross_median| ≥ {args.magnitude_threshold}` AND "
        f"`{args.weak_consensus} ≤ consensus < {args.strong_consensus}`",
        f"    - **AMBIGUOUS**: otherwise — not scored against literature",
        "- Null control: cross_asym `p_emp` from "
        f"{args.n_null_perms} random (S_A, S_B) gene-set pairs of the same sizes, "
        "drawn from HVGs disjoint from any observed S_X. Tests whether the "
        "DISCOVERED S_X carries cytokine-specific information vs random "
        "activation-responsive HVGs (the trap §23 Audit 2 caught on Sheu).",
        "- `expected_sign`: +1 if literature says a_to_b, −1 if b_to_a, blank "
        "for NOVEL / no_lit axes.",
        f"- §24 min_cells = {args.min_cells}.  S_X size = top_{args.top_n} genes "
        "per cytokine from `binary_ig.parquet`.",
    ]
    (output_dir / "verdict.md").write_text("\n".join(verdict_lines) + "\n")
    _log(f"\nWrote {output_dir / 'verdict.md'}")
    _log(f"\nOVERALL: {n_strong} STRONG / {n_weak} WEAK / {n_ambig} AMBIGUOUS; "
         f"ground-truth sign accuracy {n_correct}/{n_ground_truth} on "
         f"non-AMBIGUOUS calls; {n_null_pass} pass null p < 0.05")


if __name__ == "__main__":
    main()
