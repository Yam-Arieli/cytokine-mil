"""
Signature-space coupling driver (CLAUDE.md §28).

Reframe of Path A: instead of measuring cytokine coupling in the encoder
embedding (latent geometry, which is dominated by the shared post-activation
program), measure it DIRECTLY in the cytokine-specific dimensions — the
discovered binary-IG signatures S_X. Builds the cross-engagement matrix
M[a,b]=s(a,S_b) (PBS-normalised), then:

    coupling(a,b)  = M[a,b] + M[b,a]   (symmetric; gated by a gene-set null)
    cross_asym(a,b)= M[a,b] - M[b,a]   (antisymmetric; direction, §26)

Two datasets:
  --dataset oesinghaus : compare the coupling axis set + literature support to
                         latent-geometry Path A (cytokine_axes.csv).
  --dataset sheu       : the decisive 'irrelevant features' test — does
                         signature-space coupling recover the textbook TLR pairs
                         (LPS-TNF, polyIC-IFNb) that the latent-geometry gate
                         FAILED (q=1 everywhere)? Uses --time_filter (single frame).

Allowed imports: argparse, json, sys, time, pathlib, typing, numpy, pandas, and
the numpy-only cytokine_mil.analysis.signature_coupling. NO tabulate/matplotlib/
seaborn/scipy (Spearman is computed by hand).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.signature_coupling import coupling_direction  # noqa: E402


# ---------------------------------------------------------------------------
# S_X from binary_ig.parquet
# ---------------------------------------------------------------------------

def _build_s_x(binary_ig_parquet: str, top_n: int) -> Dict[str, List[str]]:
    df = pd.read_parquet(binary_ig_parquet)
    need = {"cytokine", "gene", "rank_ig"}
    if need - set(df.columns):
        raise ValueError(f"binary_ig missing {need - set(df.columns)}")
    out: Dict[str, List[str]] = {}
    for cyt, sub in df.groupby("cytokine"):
        out[cyt] = sub.sort_values("rank_ig")["gene"].head(top_n).tolist()
    return out


def _to_idx(s_x: Dict[str, List[str]], gene_names: List[str]) -> Dict[str, np.ndarray]:
    g2i = {g: i for i, g in enumerate(gene_names)}
    return {c: np.array([g2i[g] for g in genes if g in g2i], dtype=np.int64)
            for c, genes in s_x.items()}


def _canon(a: str, b: str):
    return (a, b) if a <= b else (b, a)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via rank + Pearson (numpy only)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    c = np.corrcoef(rx, ry)
    return float(c[0, 1])


def _md(df: pd.DataFrame, cols) -> str:
    cols = list(cols)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append("NaN" if (isinstance(v, float) and np.isnan(v)) else f"{v:+.4f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Cell loading (dataset-specific)
# ---------------------------------------------------------------------------

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


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["oesinghaus", "sheu"])
    p.add_argument("--binary_ig_parquet", required=True)
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--hvg_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--axes_csv", default=None,
                   help="Labels for comparison (cytokine_axes.csv / sheu_axes_labeled.csv)")
    p.add_argument("--top_n", type=int, default=50)
    p.add_argument("--pbs_label", default="PBS")
    p.add_argument("--min_cells", type=int, default=10)
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--null_seed", type=int, default=42)
    p.add_argument("--time_filter", default=None, help="Sheu single-frame, e.g. 5hr")
    p.add_argument("--exclude_donors", nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m="": print(m, flush=True)

    for pth in (args.binary_ig_parquet, args.manifest_path, args.hvg_path):
        if not Path(pth).exists():
            log(f"FATAL: missing input {pth}")
            sys.exit(2)

    log("=" * 70)
    log(f"Signature-space coupling — dataset={args.dataset} "
        f"time_filter={args.time_filter} exclude_donors={args.exclude_donors}")
    log("=" * 70)

    s_x = _build_s_x(args.binary_ig_parquet, args.top_n)
    sig_cyts = sorted(s_x)
    log(f"S_X for {len(sig_cyts)} cytokines (top_n={args.top_n})")

    t0 = time.time()
    cells, gene_names = _load_cells(args, sig_cyts)
    log(f"loaded {len(cells)} (cyt,cell_type) groups, {len(gene_names)} genes "
        f"({time.time()-t0:.0f}s)")
    sig_idx = _to_idx(s_x, gene_names)
    sig_idx = {c: v for c, v in sig_idx.items() if len(v) > 0}

    t0 = time.time()
    rng = np.random.default_rng(args.null_seed)
    rows = coupling_direction(
        cells, sig_idx, pbs_label=args.pbs_label, min_cells=args.min_cells,
        n_perm=args.n_perm, rng=rng,
    )
    df = pd.DataFrame(rows)
    log(f"coupling computed for {len(df)} unordered pairs ({time.time()-t0:.0f}s)")

    # canonical key + null-pass flag
    df["coupled"] = df["coupling_null_p"] < 0.05
    df["direction_call"] = np.where(
        df["cross_asym"] > 0, df["axis_a"] + " -> " + df["axis_b"],
        np.where(df["cross_asym"] < 0, df["axis_b"] + " -> " + df["axis_a"], "ambiguous"))

    # ---- join labels (optional) ----
    status_col = None
    if args.axes_csv and Path(args.axes_csv).exists():
        lab = pd.read_csv(args.axes_csv)
        lab["_k"] = lab.apply(lambda r: "__".join(_canon(str(r["axis_a"]), str(r["axis_b"]))), axis=1)
        df["_k"] = df.apply(lambda r: "__".join(_canon(r["axis_a"], r["axis_b"])), axis=1)
        for cand in ("benchmark_class", "pair_status", "literature_status"):
            if cand in lab.columns:
                status_col = cand
                break
        keep_cols = ["_k"] + ([status_col] if status_col else [])
        if "axis_strength" in lab.columns:
            keep_cols.append("axis_strength")
        df = df.merge(lab[keep_cols].drop_duplicates("_k"), on="_k", how="left")
        df = df.drop(columns=["_k"])

    df = df.sort_values("coupling", ascending=False).reset_index(drop=True)
    df.to_csv(out / "coupling_axes.csv", index=False)
    log(f"wrote {out/'coupling_axes.csv'}")

    # ---- report ----
    L = [f"# Signature-space coupling — {args.dataset}", ""]
    L.append(f"- binary_ig: `{args.binary_ig_parquet}`")
    L.append(f"- cytokines with S_X: {len(sig_cyts)}; pairs scored: {len(df)}; "
             f"time_filter={args.time_filter}; exclude_donors={args.exclude_donors}")
    L.append(f"- coupling(a,b) = M[a,b]+M[b,a] (symmetric, specific-dimension "
             f"engagement); gate = gene-set null p<0.05 ({args.n_perm} perms).")
    n_coupled = int(df["coupled"].sum())
    L.append(f"- **coupled pairs (null p<0.05): {n_coupled} / {len(df)}**")
    L.append("")

    if args.dataset == "sheu" and status_col:
        L.append("## Decisive test: does signature-space coupling recover the TLR "
                 "pairs that latent-geometry Path A MISSED (q=1)?")
        L.append("")
        must = df[df[status_col].astype(str).str.contains("MUST", case=False, na=False)]
        L.append(_md(df.sort_values(status_col),
                     ["axis_a", "axis_b", status_col, "coupling",
                      "coupling_null_p", "cross_asym", "direction_call"]))
        L.append("")
        rec = must[must["coupled"]] if len(must) else must
        L.append(f"**MUST pairs clearing the coupling null: {len(rec)}/{len(must)}** "
                 f"(latent-geometry Path A recovered 0/2). cross_asym gives direction "
                 f"on the coupled ones.")
    else:
        if "axis_strength" in df.columns:
            rho = _spearman(df["coupling"].to_numpy(),
                            pd.to_numeric(df["axis_strength"], errors="coerce").to_numpy())
            L.append(f"## vs latent-geometry Path A")
            L.append(f"Spearman(coupling, Path A axis_strength) = **{rho:+.3f}** over "
                     f"{len(df)} pairs (low ⇒ they rank pairs differently).")
            L.append("")
        if status_col:
            known = df[df[status_col].astype(str).str.contains(
                "KNOWN|PRE_REG|DIRECTIONAL|COREG", case=False, na=False)]
            n_known_coupled = int(known["coupled"].sum())
            L.append(f"Of literature-known pairs present, {n_known_coupled}/{len(known)} "
                     f"clear the signature-coupling null.")
            L.append("")
        L.append("## Top-20 coupled pairs (signature space)")
        L.append("")
        top = df.head(20)
        cols = ["axis_a", "axis_b", "coupling", "coupling_null_p", "cross_asym",
                "direction_call"]
        if status_col:
            cols.insert(2, status_col)
        L.append(_md(top, cols))

    L.append("")
    L.append("## Caveats")
    L.append("- Coupling lives in S_X (specific dimensions); validity rides on S_X "
             "specificity (the gene-set null is the gate). cross_asym gives DIRECTION "
             "only, on coupled pairs (existence != direction; §26.4).")
    (out / "coupling_report.md").write_text("\n".join(L) + "\n")
    log(f"wrote {out/'coupling_report.md'}")
    log(f"\nDONE. coupled pairs (null p<0.05): {n_coupled}/{len(df)}")


if __name__ == "__main__":
    main()
