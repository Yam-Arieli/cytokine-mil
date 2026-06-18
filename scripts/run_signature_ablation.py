"""
Signature-definition ablation (2x2): {IG, DE} x {vs-PBS, vs-panel}.

GOAL. Hold the WORKING pipeline fixed (the cross-engagement matrix M, the
cross_asym direction call, the gene-set-null coupling gate, the audited
benchmark) and swap ONLY how each cytokine's signature S_X is defined. Four
definitions, evaluated on the SAME cells, SAME eval:

  selector x contrast :
    IG  x vs-PBS    -> rank genes by binary-IG attribution (the CURRENT method).
    IG  x vs-panel  -> rank by  ig_X[g] - mean_over_cytokines ig[g]  (residualise
                       the IG against the panel = remove shared activation).
    DE  x vs-PBS    -> rank by  mean(X cells)[g] - mean(PBS cells)[g]  (no encoder
                       / no IG -- plain differential expression vs resting).
    DE  x vs-panel  -> rank by  dX[g] - mean_over_cytokines dX[g]  with
                       dX = mean(X) - mean(PBS)  (DE residualised against panel).

The vs-panel column subtracts the cross-cytokine mean shift, stripping the
shared-activation program every cytokine co-induces while KEEPING genes a
cytokine shares with only a partner (those sit above the panel average).

THREE THINGS THIS ANSWERS AT ONCE (one CPU job, no retraining):
  Q1/Q3 specificity: does vs-panel SHRINK the over-permissive coupling gate
                     (Oes: 894/1128 coupled) and reduce hub-domination, while
                     HOLDING the 88% direction accuracy?
  Q4 encoder:        does DE ~= IG?  If yes, the encoder+IG is unnecessary
                     scaffolding for the direction result.
  baseline:          IG x vs-PBS must reproduce the published ~88% (sanity).

Per variant we report:
  - direction accuracy: sign(median_over_celltypes cross_asym) vs expected_sign
    on the audited benchmark (counts_in_benchmark=True) -- EXACTLY the retally
    aggregation (median of per-cell-type cross_asym).
  - coupling gate: fraction of pairs clearing the gene-set null (p<0.05), plus
    hub-domination stats over the top coupled pairs.

Allowed imports: argparse, json, sys, time, pathlib, typing, numpy, pandas, and
the numpy-only cytokine_mil.analysis.signature_coupling. NO scipy/matplotlib.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.signature_coupling import (  # noqa: E402
    coupling_direction,
    engagement_per_celltype,
)


# ---------------------------------------------------------------------------
# Signature builders -> {cytokine: [gene_name, ...]} (top_n)
# ---------------------------------------------------------------------------

def _ig_matrix(binary_ig_parquet: str) -> Tuple[pd.DataFrame, List[str]]:
    """Pivot the long binary_ig.parquet to a (cytokine x gene) IG matrix.

    Returns (ig_df, gene_order) where ig_df.index = cytokines, columns = genes.
    Requires the parquet to store ig for every gene per cytokine (it does --
    run_binary_ig_probe writes one row per (cytokine, gene)).
    """
    df = pd.read_parquet(binary_ig_parquet)
    need = {"cytokine", "gene", "ig"}
    if need - set(df.columns):
        raise ValueError(f"binary_ig missing {need - set(df.columns)}")
    ig_df = df.pivot_table(index="cytokine", columns="gene", values="ig", aggfunc="mean")
    return ig_df, list(ig_df.columns)


def _topn_from_scores(score_df: pd.DataFrame, top_n: int) -> Dict[str, List[str]]:
    """For each row (cytokine), the top_n column (gene) names by descending score."""
    out: Dict[str, List[str]] = {}
    genes = np.asarray(score_df.columns)
    for cyt, row in score_df.iterrows():
        vals = row.to_numpy(dtype=np.float64)
        order = np.argsort(-vals)
        out[str(cyt)] = [str(genes[i]) for i in order[: top_n]]
    return out


def build_signature_variants(
    ig_df: pd.DataFrame,
    de_df: pd.DataFrame,
    top_n: int,
) -> Dict[str, Dict[str, List[str]]]:
    """Build the four S_X variants. ig_df / de_df: (cytokine x gene) score matrices
    aligned to the SAME cytokines and genes."""
    # Align both matrices to the shared cytokines & genes
    cyts = sorted(set(ig_df.index) & set(de_df.index))
    genes = [g for g in de_df.columns if g in set(ig_df.columns)]
    ig = ig_df.loc[cyts, genes]
    de = de_df.loc[cyts, genes]

    # Panel-residualised: subtract per-gene mean across cytokines
    ig_panel = ig.sub(ig.mean(axis=0), axis=1)
    de_panel = de.sub(de.mean(axis=0), axis=1)

    return {
        "IG_vsPBS": _topn_from_scores(ig, top_n),
        "IG_vsPanel": _topn_from_scores(ig_panel, top_n),
        "DE_vsPBS": _topn_from_scores(de, top_n),
        "DE_vsPanel": _topn_from_scores(de_panel, top_n),
    }


# ---------------------------------------------------------------------------
# DE matrix from loaded cells
# ---------------------------------------------------------------------------

def de_matrix_from_cells(
    cells: Dict[Tuple[str, str], np.ndarray],
    cytokines: Sequence[str],
    gene_names: Sequence[str],
    pbs_label: str = "PBS",
) -> pd.DataFrame:
    """dX[g] = mean(X cells, pooled over cell types)[g] - mean(PBS cells)[g].

    Returns a (cytokine x gene) DataFrame.
    """
    def _pooled_mean(cyt: str) -> Optional[np.ndarray]:
        arrs = [v for (c, _t), v in cells.items() if c == cyt and len(v)]
        if not arrs:
            return None
        tot = sum(a.shape[0] for a in arrs)
        acc = np.zeros(len(gene_names), dtype=np.float64)
        for a in arrs:
            acc += a.sum(axis=0)
        return acc / tot

    pbs_mean = _pooled_mean(pbs_label)
    if pbs_mean is None:
        raise ValueError(f"No {pbs_label!r} cells to form the DE baseline.")
    rows: Dict[str, np.ndarray] = {}
    for cyt in cytokines:
        if cyt == pbs_label:
            continue
        m = _pooled_mean(cyt)
        if m is not None:
            rows[cyt] = m - pbs_mean
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(gene_names))


# ---------------------------------------------------------------------------
# Direction accuracy (faithful: median of per-cell-type cross_asym, sign vs label)
# ---------------------------------------------------------------------------

def direction_accuracy(
    cells: Dict[Tuple[str, str], np.ndarray],
    sig_idx: Dict[str, np.ndarray],
    audit_df: pd.DataFrame,
    pbs_label: str,
    min_cells: int,
) -> Tuple[int, int, pd.DataFrame]:
    """Sign of median(per-cell-type cross_asym) vs audited expected_sign.

    Only rows with counts_in_benchmark == True are graded. Returns
    (n_correct, n_total, detail_df).
    """
    cyts, cell_types, E = engagement_per_celltype(cells, sig_idx, pbs_label, min_cells)
    col = {c: j for j, c in enumerate(cyts)}

    bench = audit_df[audit_df["counts_in_benchmark"].astype(str).str.lower() == "true"].copy()
    detail = []
    n_correct = 0
    n_total = 0
    for _, r in bench.iterrows():
        a, b = str(r["axis_a"]), str(r["axis_b"])
        if a not in col or b not in col:
            continue
        i, j = col[a], col[b]
        per_ct = E[:, i, j] - E[:, j, i]  # per-cell-type cross_asym (a<b canonical)
        per_ct = per_ct[np.isfinite(per_ct)]
        if per_ct.size == 0:
            continue
        med = float(np.median(per_ct))
        obs_sign = int(np.sign(med))
        try:
            exp_sign = int(float(r["expected_sign"]))
        except (ValueError, TypeError):
            continue
        ok = obs_sign == exp_sign
        n_total += 1
        n_correct += int(ok)
        detail.append({
            "axis_a": a, "axis_b": b, "pair_status": r.get("pair_status", ""),
            "expected_sign": exp_sign, "observed_sign": obs_sign,
            "cross_asym_median": med, "n_celltypes": int(per_ct.size),
            "correct": ok,
        })
    return n_correct, n_total, pd.DataFrame(detail)


# ---------------------------------------------------------------------------
# Coupling-gate stats
# ---------------------------------------------------------------------------

def coupling_stats(rows: List[dict], top_k: int = 20) -> Dict[str, object]:
    """Over-permissiveness + hub-domination summary of a coupling result."""
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_pairs": 0, "n_coupled": 0, "coupled_frac": float("nan")}
    coupled = df[df["coupling_null_p"] < 0.05]
    n_pairs, n_coupled = len(df), len(coupled)

    # hub domination over the top-K coupled pairs (by coupling strength)
    top = coupled.sort_values("coupling", ascending=False).head(top_k)
    cyts = pd.concat([top["axis_a"], top["axis_b"]])
    vc = cyts.value_counts()
    return {
        "n_pairs": n_pairs,
        "n_coupled": n_coupled,
        "coupled_frac": n_coupled / n_pairs if n_pairs else float("nan"),
        "top20_distinct_cyts": int(vc.size),
        "top20_max_cyt": (str(vc.index[0]) if vc.size else ""),
        "top20_max_cyt_count": int(vc.iloc[0]) if vc.size else 0,
    }


# ---------------------------------------------------------------------------
# Cell loading (mirrors run_signature_coupling)
# ---------------------------------------------------------------------------

def _to_idx(s_x: Dict[str, List[str]], gene_names: List[str]) -> Dict[str, np.ndarray]:
    g2i = {g: i for i, g in enumerate(gene_names)}
    return {c: np.array([g2i[g] for g in genes if g in g2i], dtype=np.int64)
            for c, genes in s_x.items()}


def _load_cells(args, sig_cyts):
    if args.dataset == "oesinghaus":
        from cytokine_mil.analysis.oesinghaus_cell_loader import (
            load_oesinghaus_cells_by_pair,
        )
        return load_oesinghaus_cells_by_pair(
            manifest_path=args.manifest_path,
            cytokines=sorted(sig_cyts),
            hvg_path=args.hvg_path,
            pbs_label=args.pbs_label,
            exclude_donors=args.exclude_donors,
        )
    elif args.dataset == "sheu":
        import json
        from cytokine_mil.analysis.eda_pair_benchmark import load_phase1_cells
        with open(args.hvg_path) as fh:
            gene_names = json.load(fh)
        cells, genes = load_phase1_cells(
            manifest_path=args.manifest_path, gene_names=gene_names,
            time_filter=args.time_filter, donors=None,
        )
        keep = set(sig_cyts) | {args.pbs_label}
        cells = {k: v for k, v in cells.items() if k[0] in keep}
        return cells, genes
    raise ValueError(f"unknown dataset {args.dataset}")


def _md(df: pd.DataFrame, cols) -> str:
    cols = list(cols)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append("NaN" if np.isnan(v) else f"{v:+.4f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["oesinghaus", "sheu"])
    p.add_argument("--binary_ig_parquet", required=True)
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--hvg_path", required=True)
    p.add_argument("--audit_csv", required=True,
                   help="cytokine_axes_audited.csv (axis_a,axis_b,expected_sign,"
                        "counts_in_benchmark,pair_status)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_n", type=int, default=50)
    p.add_argument("--pbs_label", default="PBS")
    p.add_argument("--min_cells", type=int, default=10)
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--null_seed", type=int, default=42)
    p.add_argument("--time_filter", default=None, help="Sheu single-frame, e.g. 5hr")
    p.add_argument("--exclude_donors", nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m="": print(m, flush=True)

    for pth in (args.binary_ig_parquet, args.manifest_path, args.hvg_path, args.audit_csv):
        if not Path(pth).exists():
            log(f"FATAL: missing input {pth}")
            sys.exit(2)

    log("=" * 70)
    log(f"Signature ablation (2x2) — dataset={args.dataset} top_n={args.top_n} "
        f"n_perm={args.n_perm} exclude_donors={args.exclude_donors}")
    log("=" * 70)

    # IG matrix (cytokine x gene) over ALL genes -> defines the IG cytokine set
    ig_df, _ = _ig_matrix(args.binary_ig_parquet)
    sig_cyts = sorted(str(c) for c in ig_df.index)
    log(f"IG matrix: {len(sig_cyts)} cytokines x {ig_df.shape[1]} genes")

    # Cells (also defines gene order used for DE + indexing)
    t0 = time.time()
    cells, gene_names = _load_cells(args, sig_cyts)
    log(f"loaded {len(cells)} (cyt,cell_type) groups, {len(gene_names)} genes "
        f"({time.time()-t0:.0f}s)")

    # DE matrix from cells (same cytokine set), aligned to gene_names
    de_df = de_matrix_from_cells(cells, sig_cyts, gene_names, pbs_label=args.pbs_label)
    # Align IG columns to gene_names present in cells
    ig_df = ig_df.reindex(columns=gene_names).dropna(axis=1, how="all").fillna(0.0)
    log(f"DE matrix: {de_df.shape[0]} cytokines x {de_df.shape[1]} genes")

    variants = build_signature_variants(ig_df, de_df, args.top_n)
    audit_df = pd.read_csv(args.audit_csv)

    summary_rows = []
    for name, s_x in variants.items():
        sig_idx = {c: v for c, v in _to_idx(s_x, gene_names).items() if len(v) > 0}
        log(f"\n--- variant {name}: {len(sig_idx)} signatures ---")

        # direction accuracy (faithful per-cell-type median sign)
        n_corr, n_tot, det = direction_accuracy(
            cells, sig_idx, audit_df, args.pbs_label, args.min_cells)
        det.to_csv(out / f"direction_detail_{name}.csv", index=False)

        # coupling + null gate
        t0 = time.time()
        rng = np.random.default_rng(args.null_seed)
        rows = coupling_direction(
            cells, sig_idx, pbs_label=args.pbs_label, min_cells=args.min_cells,
            n_perm=args.n_perm, rng=rng)
        pd.DataFrame(rows).to_csv(out / f"coupling_{name}.csv", index=False)
        cs = coupling_stats(rows)
        log(f"  direction {n_corr}/{n_tot}  coupled {cs['n_coupled']}/{cs['n_pairs']} "
            f"({cs['coupled_frac']:.0%})  ({time.time()-t0:.0f}s)")

        summary_rows.append({
            "variant": name,
            "n_signatures": len(sig_idx),
            "direction_correct": n_corr,
            "direction_total": n_tot,
            "direction_acc": (n_corr / n_tot) if n_tot else float("nan"),
            "coupled": cs["n_coupled"],
            "pairs": cs["n_pairs"],
            "coupled_frac": cs["coupled_frac"],
            "top20_distinct_cyts": cs.get("top20_distinct_cyts", 0),
            "top20_max_cyt": cs.get("top20_max_cyt", ""),
            "top20_max_cyt_count": cs.get("top20_max_cyt_count", 0),
        })

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out / "ablation_summary.csv", index=False)

    # ---- report ----
    L = [f"# Signature-definition ablation (2x2) — {args.dataset}", ""]
    L.append(f"- binary_ig: `{args.binary_ig_parquet}`")
    L.append(f"- top_n={args.top_n}, n_perm={args.n_perm}, "
             f"exclude_donors={args.exclude_donors}, time_filter={args.time_filter}")
    L.append("- **direction_acc** = sign(median per-cell-type cross_asym) vs audited "
             "expected_sign (counts_in_benchmark=True). Higher = better.")
    L.append("- **coupled_frac** = fraction of pairs clearing the gene-set null "
             "(p<0.05). LOWER = less over-permissive (toward real biology).")
    L.append("- **top20_max_cyt_count** = times the most-frequent cytokine appears in the "
             "top-20 coupled pairs. LOWER = less hub-dominated.")
    L.append("")
    L.append(_md(summ, ["variant", "n_signatures", "direction_correct",
                        "direction_total", "direction_acc", "coupled", "pairs",
                        "coupled_frac", "top20_distinct_cyts", "top20_max_cyt",
                        "top20_max_cyt_count"]))
    L.append("")
    L.append("## How to read this")
    L.append("- **IG_vsPBS** is the current method; it should reproduce the published "
             "~88% direction (sanity).")
    L.append("- **vs-panel shrinks coupled_frac while holding direction_acc** => "
             "specificity is the lever; adopt panel-residualised signatures (then a "
             "learned cytokine-conditioned gate to make it differentiable).")
    L.append("- **DE ~= IG on direction_acc** => the encoder+IG is unnecessary "
             "scaffolding for the direction result (major simplification).")
    L.append("- **neither helps** => specificity is not the lever; pivot to donor-level "
             "statistics as the primary coupling fix.")
    L.append("")
    L.append("Per-variant detail: `direction_detail_<variant>.csv` (which benchmark "
             "pairs flip), `coupling_<variant>.csv` (full pair table).")
    (out / "ablation_report.md").write_text("\n".join(L) + "\n")
    log(f"\nwrote {out/'ablation_report.md'}")
    log("\n" + _md(summ, ["variant", "direction_acc", "coupled_frac",
                          "top20_max_cyt_count"]))
    log("\nDONE.")


if __name__ == "__main__":
    main()
