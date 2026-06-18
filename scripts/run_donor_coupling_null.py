"""
Donor-level coupling gate — the validation test (CLAUDE.md §16, §28).

QUESTION. The signature-coupling existence gate over-calls (~77% of Oes pairs
"coupled") because its random-gene null is computed at the CELL level (thousands
of cells -> trivially beatable). Does a DONOR-level null actually discriminate,
and does the panel-residualised signature (IG_vsPanel) beat the current vs-PBS
one once the null is no longer over-powered?

METHOD (per signature variant):
  - Per donor d: excess_d[a,b] = coupling_d(a,b) - mean(random-gene coupling_d)
    using ONLY donor d's cells + d's own PBS baseline (donor_excess_matrix).
  - Across donors: one-sided sign-flip permutation test that excess > 0
    (donor_coupling_test). Effective N = #donors (~10), so power is capped and
    the gate can discriminate. BH-FDR across pairs.
  - Compare donor-coupled fraction (q<0.05 / q<0.10) to the cell-level fraction
    (read from the ablation summary), and report recall on the audited
    directional benchmark (those pairs are known cascades -> should stay coupled).

Runs IG_vsPBS and IG_vsPanel (signatures built from the binary_ig parquet; no DE
needed). Research code only -- does NOT touch cascadir. CPU-only.
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
from cytokine_mil.analysis.signature_coupling import (  # noqa: E402
    donor_excess_matrix, donor_coupling_test,
)

VARIANTS = ["IG_vsPBS", "IG_vsPanel"]


def _build_ig_variants(binary_ig_parquet: str, gene_names: List[str], top_n: int):
    """IG_vsPBS and IG_vsPanel signatures (donor-independent), aligned to gene_names."""
    ig_df, _ = rsa._ig_matrix(binary_ig_parquet)
    ig_df = ig_df.reindex(columns=gene_names).dropna(axis=1, how="all").fillna(0.0)
    ig_panel = ig_df.sub(ig_df.mean(axis=0), axis=1)
    return {
        "IG_vsPBS": rsa._topn_from_scores(ig_df, top_n),
        "IG_vsPanel": rsa._topn_from_scores(ig_panel, top_n),
    }, sorted(str(c) for c in ig_df.index)


def _donors_from_manifest(manifest_path: str, exclude: List[str]) -> List[str]:
    with open(manifest_path) as fh:
        man = json.load(fh)
    entries = man if isinstance(man, list) else man.get("entries", man.get("tubes", []))
    donors = sorted({str(e["donor"]) for e in entries if "donor" in e})
    ex = set(exclude or [])
    return [d for d in donors if d not in ex]


def _hub_in_top20(rows: List[dict]) -> Dict:
    coupled = [r for r in rows if r.get("q_donor", 1.0) < 0.10]
    top = sorted(coupled, key=lambda r: r["excess_mean"], reverse=True)[:20]
    cyts = pd.Series([c for r in top for c in (r["axis_a"], r["axis_b"])])
    vc = cyts.value_counts()
    return {"top20_distinct_cyts": int(vc.size),
            "top20_max_cyt": (str(vc.index[0]) if vc.size else ""),
            "top20_max_cyt_count": int(vc.iloc[0]) if vc.size else 0}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="oesinghaus", choices=["oesinghaus"])
    p.add_argument("--binary_ig_parquet", required=True)
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--hvg_path", required=True)
    p.add_argument("--audit_csv", required=True)
    p.add_argument("--ablation_summary", default=None,
                   help="ablation_summary.csv for the cell-level coupled_frac reference")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_n", type=int, default=50)
    p.add_argument("--pbs_label", default="PBS")
    p.add_argument("--min_cells", type=int, default=10)
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--min_donors", type=int, default=5)
    p.add_argument("--null_seed", type=int, default=42)
    p.add_argument("--exclude_donors", nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m="": print(m, flush=True)

    from cytokine_mil.analysis.oesinghaus_cell_loader import load_oesinghaus_cells_by_pair

    donors = _donors_from_manifest(args.manifest_path, args.exclude_donors or [])
    log(f"donors (train): {donors}")

    # signatures need gene order -> load one donor first
    t0 = time.time()
    cells0, gene_names = load_oesinghaus_cells_by_pair(
        manifest_path=args.manifest_path, cytokines=["PBS"], hvg_path=args.hvg_path,
        pbs_label=args.pbs_label, include_donors=[donors[0]])
    del cells0
    sigs, sig_cyts = _build_ig_variants(args.binary_ig_parquet, gene_names, args.top_n)
    sig_idx = {v: {c: a for c, a in rsa._to_idx(s, gene_names).items() if len(a) > 0}
               for v, s in sigs.items()}
    global_cyts = sorted(sig_cyts)
    log(f"{len(global_cyts)} cytokines; gene order from {donors[0]} ({time.time()-t0:.0f}s)")

    # per-donor excess matrices (one donor in memory at a time)
    stacks: Dict[str, List[np.ndarray]] = {v: [] for v in VARIANTS}
    used_donors: List[str] = []
    for d in donors:
        t0 = time.time()
        cells_d, gn = load_oesinghaus_cells_by_pair(
            manifest_path=args.manifest_path, cytokines=global_cyts,
            hvg_path=args.hvg_path, pbs_label=args.pbs_label, include_donors=[d])
        if list(gn) != list(gene_names):
            log(f"  WARN: gene order differs for {d}; skipping"); del cells_d; continue
        rng = np.random.default_rng(args.null_seed)
        for v in VARIANTS:
            stacks[v].append(donor_excess_matrix(
                cells_d, sig_idx[v], global_cyts, pbs_label=args.pbs_label,
                min_cells=args.min_cells, n_perm=args.n_perm, rng=rng))
        used_donors.append(d)
        del cells_d
        log(f"  donor {d}: excess computed ({time.time()-t0:.0f}s)")

    # cell-level reference (from the ablation summary)
    cell_frac = {}
    if args.ablation_summary and Path(args.ablation_summary).exists():
        s = pd.read_csv(args.ablation_summary).set_index("variant")
        for v in VARIANTS:
            if v in s.index:
                cell_frac[v] = float(s.loc[v, "coupled_frac"])

    # audited directional benchmark pairs (known cascades -> recall target)
    audit = pd.read_csv(args.audit_csv)
    bench = audit[audit["counts_in_benchmark"].astype(str).str.lower() == "true"]
    bench_pairs = {tuple(sorted((str(r["axis_a"]), str(r["axis_b"]))))
                   for _, r in bench.iterrows()}
    status_map = {}
    if "pair_status" in audit.columns:
        for _, r in audit.iterrows():
            status_map[tuple(sorted((str(r["axis_a"]), str(r["axis_b"])))) ] = r["pair_status"]

    summary_rows = []
    for v in VARIANTS:
        rng = np.random.default_rng(args.null_seed + 1)
        rows = donor_coupling_test(
            np.stack(stacks[v], axis=0), global_cyts,
            min_donors=args.min_donors, n_signflip=4000, rng=rng)
        df = pd.DataFrame(rows)
        for col, thr in [("coupled_q05", 0.05), ("coupled_q10", 0.10)]:
            df[col] = df["q_donor"] < thr
        df["is_benchmark"] = df.apply(
            lambda r: tuple(sorted((r["axis_a"], r["axis_b"]))) in bench_pairs, axis=1)
        df["pair_status"] = df.apply(
            lambda r: status_map.get(tuple(sorted((r["axis_a"], r["axis_b"]))), ""), axis=1)
        df.to_csv(out / f"donor_coupling_{v}.csv", index=False)

        n_tested = len(df)
        n_q05, n_q10 = int(df["coupled_q05"].sum()), int(df["coupled_q10"].sum())
        bench_tested = df[df["is_benchmark"]]
        recall10 = (int(bench_tested["coupled_q10"].sum()), len(bench_tested))
        hub = _hub_in_top20(rows)
        summary_rows.append({
            "variant": v,
            "cell_level_coupled_frac": cell_frac.get(v, float("nan")),
            "donor_tested_pairs": n_tested,
            "donor_coupled_q05": n_q05,
            "donor_coupled_q05_frac": n_q05 / n_tested if n_tested else float("nan"),
            "donor_coupled_q10": n_q10,
            "donor_coupled_q10_frac": n_q10 / n_tested if n_tested else float("nan"),
            "benchmark_recall_q10": f"{recall10[0]}/{recall10[1]}",
            "top20_max_cyt": hub["top20_max_cyt"],
            "top20_max_cyt_count": hub["top20_max_cyt_count"],
        })
        log(f"  {v}: donor q<0.10 coupled {n_q10}/{n_tested} "
            f"(cell-level {cell_frac.get(v, float('nan')):.0%}); "
            f"benchmark recall {recall10[0]}/{recall10[1]}")

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out / "donor_coupling_summary.csv", index=False)

    # report
    L = ["# Donor-level coupling gate — validation", ""]
    L.append(f"- donors used: {used_donors} (N={len(used_donors)})")
    L.append(f"- per-donor: excess = coupling - mean(random-gene coupling); across "
             f"donors: one-sided sign-flip test (exact), BH-FDR. min_donors={args.min_donors}.")
    L.append("- **Key comparison**: donor-level coupled fraction vs the over-powered "
             "cell-level fraction. A discriminating gate drops WELL below ~77% while "
             "keeping the known-cascade benchmark pairs (recall).")
    L.append("")
    L.append(rsa._md(summ, ["variant", "cell_level_coupled_frac", "donor_tested_pairs",
                            "donor_coupled_q05", "donor_coupled_q05_frac",
                            "donor_coupled_q10_frac", "benchmark_recall_q10",
                            "top20_max_cyt", "top20_max_cyt_count"]))
    L.append("")
    L.append("## How to read")
    L.append("- donor frac ≪ cell-level frac → the over-power was the problem; donor "
             "gate discriminates.")
    L.append("- IG_vsPanel coupled-frac < IG_vsPBS at equal/better benchmark recall → "
             "**specificity helps coupling once the null is honest** (the open question).")
    L.append("- both still ~77% → specificity is NOT enough; coupling needs more than a "
             "donor null (e.g. hub/degree correction).")
    L.append("")
    for v in VARIANTS:
        df = pd.read_csv(out / f"donor_coupling_{v}.csv")
        top = df.sort_values("excess_mean", ascending=False).head(15)
        L.append(f"## {v} — top-15 donor-coupled pairs")
        L.append(rsa._md(top, ["axis_a", "axis_b", "excess_mean", "n_donors",
                               "p_donor", "q_donor", "is_benchmark", "pair_status"]))
        L.append("")
    (out / "donor_coupling_report.md").write_text("\n".join(L) + "\n")
    log(f"\nwrote {out/'donor_coupling_report.md'}")
    log("DONE.")


if __name__ == "__main__":
    main()
