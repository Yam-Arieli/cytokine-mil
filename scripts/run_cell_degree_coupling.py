"""
Cell-level degree-corrected coupling — generalization test on a targeted panel.

The donor-level gate is inapplicable to Sheu (4 effective pseudo-donors -> the IFN
cascades are untestable). But the DEGREE CORRECTION (the novel piece) CAN be tested at
the cell level, where all pairs are testable. Question: does adding degree correction to
the powered cell-level signature-coupling gate PRESERVE the textbook IFN cascades
(LPS-IFNb, PIC-IFNb) while cutting Sheu's over-call (~18/21 coupled raw)?

Runs IG_vsPBS + IG_vsPanel; for each, cell_coupling_degree gives RAW and DEGREE-corrected
coupling, both gene-set-null gated. Benchmark = pre-registered MUST/SHOULD (positive) /
MUST-NOT (negative) via eda labeled_pair_status. Research code only (not cascadir).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_signature_ablation as rsa  # noqa: E402
from cytokine_mil.analysis.signature_coupling import cell_coupling_degree  # noqa: E402

VARIANTS = ["IG_vsPBS", "IG_vsPanel"]


def _build_ig_variants(binary_ig_parquet, gene_names, top_n):
    ig_df, _ = rsa._ig_matrix(binary_ig_parquet)
    ig_df = ig_df.reindex(columns=gene_names).dropna(axis=1, how="all").fillna(0.0)
    ig_panel = ig_df.sub(ig_df.mean(axis=0), axis=1)
    return ({"IG_vsPBS": rsa._topn_from_scores(ig_df, top_n),
             "IG_vsPanel": rsa._topn_from_scores(ig_panel, top_n)},
            sorted(str(c) for c in ig_df.index))


def _load_pooled(args, sig_cyts):
    if args.dataset == "sheu":
        from cytokine_mil.analysis.eda_pair_benchmark import load_phase1_cells
        with open(args.hvg_path) as fh:
            gene_names = json.load(fh)
        cells, gn = load_phase1_cells(
            manifest_path=args.manifest_path, gene_names=gene_names,
            time_filter=args.time_filter, donors=None)
        keep = set(sig_cyts) | {args.pbs_label}
        return {k: v for k, v in cells.items() if k[0] in keep}, gn
    else:  # oesinghaus or id -- both are generic (cyt, cell_type) manifests;
        # load_oesinghaus_cells_by_pair wraps load_phase1_cells, which pools any
        # manifest with path/cytokine keys and obs["cell_type"] (the ID §26 pathB
        # run already used this loader on the ID manifest -- _oesinghaus_filtered_*.json).
        from cytokine_mil.analysis.oesinghaus_cell_loader import load_oesinghaus_cells_by_pair
        return load_oesinghaus_cells_by_pair(
            manifest_path=args.manifest_path, cytokines=sorted(sig_cyts),
            hvg_path=args.hvg_path, pbs_label=args.pbs_label,
            exclude_donors=args.exclude_donors)


def _labels_from_csv(axes_csv):
    """Build (pos, neg) coupling labels from an axes-labeled CSV (e.g. ID's
    id_axes_labeled.csv). Maps the directional pair_status to a COUPLING (existence)
    label: a real cascade (DIRECTIONAL_* / BIDIRECTIONAL) is a coupled positive; an
    antagonism (NEGATIVE_NO_CASCADE) is a not-coupled negative; UNKNOWN / OVERLAP_*
    pairs are descriptive (excluded from recall/false-positive). Keys are canonical
    sorted (axis_a, axis_b) tuples."""
    lab = pd.read_csv(axes_csv)
    pos, neg = set(), set()
    for _, r in lab.iterrows():
        key = tuple(sorted((str(r["axis_a"]), str(r["axis_b"]))))
        st = str(r.get("pair_status", "")).strip()
        if st.startswith("DIRECTIONAL") or st == "BIDIRECTIONAL":
            pos.add(key)
        elif st == "NEGATIVE_NO_CASCADE":
            neg.add(key)
    return pos, neg


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sheu", choices=["sheu", "oesinghaus", "id"])
    p.add_argument("--binary_ig_parquet", required=True)
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--hvg_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_n", type=int, default=50)
    p.add_argument("--pbs_label", default="PBS")
    p.add_argument("--min_cells", type=int, default=10)
    p.add_argument("--n_perm", type=int, default=500)
    p.add_argument("--null_seed", type=int, default=42)
    p.add_argument("--time_filter", default=None)
    p.add_argument("--exclude_donors", nargs="+", default=None)
    p.add_argument("--axes_csv", default=None,
                   help="If set, coupling pos/neg labels come from this CSV's "
                        "pair_status column (e.g. ID id_axes_labeled.csv) instead of "
                        "the hardcoded Sheu/Oes labeled_pair_status.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m="": print(m, flush=True)

    ig_df, _ = rsa._ig_matrix(args.binary_ig_parquet)
    sig_cyts0 = sorted(str(c) for c in ig_df.index)
    t0 = time.time()
    cells, gene_names = _load_pooled(args, sig_cyts0)
    log(f"loaded {len(cells)} (cyt,celltype) groups, {len(gene_names)} genes "
        f"({time.time()-t0:.0f}s)")
    sigs, sig_cyts = _build_ig_variants(args.binary_ig_parquet, gene_names, args.top_n)

    # benchmark labels: CSV pair_status (ID) or hardcoded labeled_pair_status (Sheu/Oes)
    if args.axes_csv:
        pos, neg = _labels_from_csv(args.axes_csv)
        log(f"labels from {args.axes_csv}: {len(pos)} coupled-positive, "
            f"{len(neg)} not-coupled-negative pairs")
    else:
        from cytokine_mil.analysis.eda_pair_benchmark import labeled_pair_status
        pos, neg = set(), set()
        gc = sorted(sig_cyts)
        for i in range(len(gc)):
            for j in range(i + 1, len(gc)):
                lab = labeled_pair_status(gc[i], gc[j])
                key = tuple(sorted((gc[i], gc[j])))
                if lab == "positive":
                    pos.add(key)
                elif lab == "negative":
                    neg.add(key)

    summary = []
    for v in VARIANTS:
        sig_idx = {c: a for c, a in rsa._to_idx(sigs[v], gene_names).items() if len(a) > 0}
        rng = np.random.default_rng(args.null_seed)
        rows = cell_coupling_degree(cells, sig_idx, pbs_label=args.pbs_label,
                                    min_cells=args.min_cells, n_perm=args.n_perm, rng=rng)
        df = pd.DataFrame(rows)
        df["pair"] = df.apply(lambda r: tuple(sorted((r["axis_a"], r["axis_b"]))), axis=1)
        df["is_pos"] = df["pair"].isin(pos)
        df["is_neg"] = df["pair"].isin(neg)
        df["label"] = np.where(df["is_pos"], "MUST/SHOULD",
                               np.where(df["is_neg"], "MUST-NOT", ""))
        for m in ("raw", "hub"):
            df[f"coupled_{m}"] = df[f"null_p_{m}"] < 0.05
        df = df.sort_values("coupling_hub", ascending=False)
        df.to_csv(out / f"cell_degree_{v}.csv", index=False)

        n = len(df)
        for m in ("raw", "hub"):
            nc = int(df[f"coupled_{m}"].sum())
            rec = (int(df[df["is_pos"]][f"coupled_{m}"].sum()), int(df["is_pos"].sum()))
            fp = (int(df[df["is_neg"]][f"coupled_{m}"].sum()), int(df["is_neg"].sum()))
            summary.append({
                "variant": v, "mode": m, "coupled": nc, "pairs": n,
                "coupled_frac": nc / n if n else float("nan"),
                "MUST_recall": f"{rec[0]}/{rec[1]}",
                "MUSTNOT_coupled": f"{fp[0]}/{fp[1]}",
            })
            log(f"  {v} [{m}]: coupled {nc}/{n}; MUST recall {rec[0]}/{rec[1]}; "
                f"MUST-NOT coupled {fp[0]}/{fp[1]}")

    summ = pd.DataFrame(summary)
    summ.to_csv(out / "cell_degree_summary.csv", index=False)

    L = [f"# Cell-level degree-corrected coupling — {args.dataset} "
         f"(time_filter={args.time_filter})", ""]
    L.append("- RAW = signature coupling + gene-set null (the prior method). "
             "HUB = degree-centered coupling + (degree-centered) gene-set null.")
    L.append("- **Key question**: does HUB keep the MUST IFN cascades coupled while "
             "cutting the over-call (coupled_frac) vs RAW?")
    L.append("")
    L.append(rsa._md(summ, ["variant", "mode", "coupled", "pairs", "coupled_frac",
                            "MUST_recall", "MUSTNOT_coupled"]))
    L.append("")
    for v in VARIANTS:
        df = pd.read_csv(out / f"cell_degree_{v}.csv")
        L.append(f"## {v} — all pairs (sorted by coupling_hub)")
        L.append(rsa._md(df, ["axis_a", "axis_b", "label", "coupling_raw", "null_p_raw",
                              "coupling_hub", "null_p_hub", "cross_asym"]))
        L.append("")
    (out / "cell_degree_report.md").write_text("\n".join(L) + "\n")
    log(f"\nwrote {out/'cell_degree_report.md'}")
    log("DONE.")


if __name__ == "__main__":
    main()
