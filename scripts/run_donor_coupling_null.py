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
    donor_excess_matrix, donor_residual_coupling_matrix, donor_coupling_test,
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
    p.add_argument("--dataset", default="oesinghaus", choices=["oesinghaus", "sheu"])
    p.add_argument("--binary_ig_parquet", required=True)
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--hvg_path", required=True)
    p.add_argument("--audit_csv", default=None,
                   help="oesinghaus only: cytokine_axes_audited.csv. Sheu uses "
                        "pre-registered labeled_pair_status (no CSV needed).")
    p.add_argument("--time_filter", default=None, help="Sheu single-frame, e.g. 5hr")
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

    donors = _donors_from_manifest(args.manifest_path, args.exclude_donors or [])
    log(f"donors: {donors}")

    # dataset-aware per-donor loader + gene order
    if args.dataset == "oesinghaus":
        from cytokine_mil.analysis.oesinghaus_cell_loader import load_oesinghaus_cells_by_pair
        _c0, gene_names = load_oesinghaus_cells_by_pair(
            manifest_path=args.manifest_path, cytokines=["PBS"], hvg_path=args.hvg_path,
            pbs_label=args.pbs_label, include_donors=[donors[0]])
        del _c0

        def _load(donor, cyts):
            return load_oesinghaus_cells_by_pair(
                manifest_path=args.manifest_path, cytokines=cyts, hvg_path=args.hvg_path,
                pbs_label=args.pbs_label, include_donors=[donor])
    else:  # sheu
        import json
        from cytokine_mil.analysis.eda_pair_benchmark import load_phase1_cells
        with open(args.hvg_path) as fh:
            gene_names = json.load(fh)

        def _load(donor, cyts):
            cells, gn = load_phase1_cells(
                manifest_path=args.manifest_path, gene_names=gene_names,
                time_filter=args.time_filter, donors=[donor])
            keep = set(cyts) | {args.pbs_label}
            return {k: v for k, v in cells.items() if k[0] in keep}, gn

    sigs, sig_cyts = _build_ig_variants(args.binary_ig_parquet, gene_names, args.top_n)
    sig_idx = {v: {c: a for c, a in rsa._to_idx(s, gene_names).items() if len(a) > 0}
               for v, s in sigs.items()}
    global_cyts = sorted(sig_cyts)
    log(f"{len(global_cyts)} cytokines; {len(gene_names)} genes")

    # per-donor matrices (one donor in memory at a time):
    #   raw  = excess over random-gene baseline (cell-null removed, hub NOT)
    #   hub  = degree-centered residual coupling (hub/degree removed)
    MODES = ["raw", "hub"]
    stacks: Dict[str, Dict[str, List[np.ndarray]]] = {
        m: {v: [] for v in VARIANTS} for m in MODES}
    used_donors: List[str] = []
    for d in donors:
        t0 = time.time()
        cells_d, gn = _load(d, global_cyts)
        if list(gn) != list(gene_names):
            log(f"  WARN: gene order differs for {d}; skipping"); del cells_d; continue
        if not any(k[0] == args.pbs_label for k in cells_d):
            log(f"  WARN: no PBS cells for {d}; skipping"); del cells_d; continue
        rng = np.random.default_rng(args.null_seed)
        for v in VARIANTS:
            stacks["raw"][v].append(donor_excess_matrix(
                cells_d, sig_idx[v], global_cyts, pbs_label=args.pbs_label,
                min_cells=args.min_cells, n_perm=args.n_perm, rng=rng))
            stacks["hub"][v].append(donor_residual_coupling_matrix(
                cells_d, sig_idx[v], global_cyts, pbs_label=args.pbs_label,
                min_cells=args.min_cells))
        used_donors.append(d)
        del cells_d
        log(f"  donor {d}: raw+hub computed ({time.time()-t0:.0f}s)")

    if len(used_donors) < args.min_donors:
        log(f"FATAL: only {len(used_donors)} usable donors < min_donors={args.min_donors}.")
        sys.exit(3)

    # diagnostic: per-pair donor coverage (how many donors have a finite value)
    diag = np.stack(stacks["hub"][VARIANTS[0]], axis=0)   # (D, G, G)
    G = len(global_cyts)
    cov = np.array([np.sum(np.isfinite(diag[:, i, j]))
                    for i in range(G) for j in range(i + 1, G)])
    log(f"per-pair donor coverage: max={cov.max() if cov.size else 0}, "
        f"median={int(np.median(cov)) if cov.size else 0}, "
        f">= min_donors({args.min_donors}): {int(np.sum(cov >= args.min_donors))}/{cov.size}")

    # cell-level reference (from the ablation summary)
    cell_frac = {}
    if args.ablation_summary and Path(args.ablation_summary).exists():
        s = pd.read_csv(args.ablation_summary).set_index("variant")
        for v in VARIANTS:
            if v in s.index:
                cell_frac[v] = float(s.loc[v, "coupled_frac"])

    # benchmark pairs: oesinghaus -> audited directional pairs (recall target);
    # sheu -> pre-registered MUST (positive) / MUST-NOT (negative) coupling labels.
    bench_pairs: set = set()      # should be coupled (recall)
    neg_pairs: set = set()        # should NOT be coupled (false-positive check)
    status_map: dict = {}
    if args.dataset == "oesinghaus" and args.audit_csv and Path(args.audit_csv).exists():
        audit = pd.read_csv(args.audit_csv)
        bench = audit[audit["counts_in_benchmark"].astype(str).str.lower() == "true"]
        bench_pairs = {tuple(sorted((str(r["axis_a"]), str(r["axis_b"]))))
                       for _, r in bench.iterrows()}
        if "pair_status" in audit.columns:
            for _, r in audit.iterrows():
                status_map[tuple(sorted((str(r["axis_a"]), str(r["axis_b"]))))] = r["pair_status"]
    else:  # sheu pre-registered labels
        from cytokine_mil.analysis.eda_pair_benchmark import labeled_pair_status
        for i in range(len(global_cyts)):
            for j in range(i + 1, len(global_cyts)):
                a, b = global_cyts[i], global_cyts[j]
                lab = labeled_pair_status(a, b)
                key = tuple(sorted((a, b)))
                if lab == "positive":
                    bench_pairs.add(key); status_map[key] = "MUST"
                elif lab == "negative":
                    neg_pairs.add(key); status_map[key] = "MUST-NOT"

    summary_rows = []
    for mode in MODES:
        for v in VARIANTS:
            rng = np.random.default_rng(args.null_seed + 1)
            rows = donor_coupling_test(
                np.stack(stacks[mode][v], axis=0), global_cyts,
                min_donors=args.min_donors, n_signflip=4000, rng=rng)
            if not rows:
                log(f"  [{mode}] {v}: NO testable pairs (coverage < min_donors="
                    f"{args.min_donors}); skipping.")
                summary_rows.append({"mode": mode, "variant": v,
                                     "donor_tested_pairs": 0,
                                     "benchmark_recall_q10": "0/0",
                                     "neg_coupled_q10": "0/0"})
                continue
            df = pd.DataFrame(rows)
            for col, thr in [("coupled_q05", 0.05), ("coupled_q10", 0.10)]:
                df[col] = df["q_donor"] < thr
            df["is_benchmark"] = df.apply(
                lambda r: tuple(sorted((r["axis_a"], r["axis_b"]))) in bench_pairs, axis=1)
            df["is_negative"] = df.apply(
                lambda r: tuple(sorted((r["axis_a"], r["axis_b"]))) in neg_pairs, axis=1)
            df["pair_status"] = df.apply(
                lambda r: status_map.get(tuple(sorted((r["axis_a"], r["axis_b"]))), ""), axis=1)
            df.to_csv(out / f"donor_coupling_{mode}_{v}.csv", index=False)

            n_tested = len(df)
            n_q05, n_q10 = int(df["coupled_q05"].sum()), int(df["coupled_q10"].sum())
            bench_tested = df[df["is_benchmark"]]
            recall10 = (int(bench_tested["coupled_q10"].sum()), len(bench_tested))
            neg_tested = df[df["is_negative"]]
            neg_fp10 = (int(neg_tested["coupled_q10"].sum()), len(neg_tested))
            hub = _hub_in_top20(rows)
            summary_rows.append({
                "mode": mode, "variant": v,
                "cell_level_coupled_frac": cell_frac.get(v, float("nan")),
                "donor_tested_pairs": n_tested,
                "donor_coupled_q05": n_q05,
                "donor_coupled_q05_frac": n_q05 / n_tested if n_tested else float("nan"),
                "donor_coupled_q10": n_q10,
                "donor_coupled_q10_frac": n_q10 / n_tested if n_tested else float("nan"),
                "benchmark_recall_q10": f"{recall10[0]}/{recall10[1]}",
                "neg_coupled_q10": f"{neg_fp10[0]}/{neg_fp10[1]}",
                "top20_max_cyt": hub["top20_max_cyt"],
                "top20_max_cyt_count": hub["top20_max_cyt_count"],
            })
            log(f"  [{mode}] {v}: donor q<0.10 coupled {n_q10}/{n_tested}; "
                f"benchmark(MUST) recall {recall10[0]}/{recall10[1]}; "
                f"negatives(MUST-NOT) coupled {neg_fp10[0]}/{neg_fp10[1]}")

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out / "donor_coupling_summary.csv", index=False)

    # report
    L = ["# Donor-level coupling gate — validation (raw vs hub-corrected)", ""]
    L.append(f"- donors used: {used_donors} (N={len(used_donors)})")
    L.append("- **raw**: per-donor excess over random-gene baseline (removes cell "
             "over-power, NOT hubs). **hub**: degree-centered residual coupling "
             "(each cytokine's overall strength subtracted -> pair-SPECIFIC signal).")
    L.append(f"- across donors: one-sided sign-flip test (exact), BH-FDR. "
             f"min_donors={args.min_donors}.")
    L.append("- **Key comparison**: does hub correction drop the coupled fraction "
             "toward sparsity while KEEPING benchmark recall, and does IL-15 stop "
             "dominating the top?")
    L.append("")
    L.append(rsa._md(summ, ["mode", "variant", "cell_level_coupled_frac",
                            "donor_tested_pairs", "donor_coupled_q05_frac",
                            "donor_coupled_q10_frac", "benchmark_recall_q10",
                            "neg_coupled_q10", "top20_max_cyt", "top20_max_cyt_count"]))
    L.append("")
    L.append("## How to read")
    L.append("- cell-level ~77% → raw-donor ~53% → **hub** much lower at equal recall → "
             "the hub/degree artifact was the remaining over-call; corrected gate "
             "discriminates.")
    L.append("- IL-15 disappears from `top20_max_cyt` under **hub** → degree correction "
             "worked (broad signatures no longer look coupled to everything).")
    L.append("- IG_vsPanel ≤ IG_vsPBS coupled-frac at equal recall under **hub** → "
             "panel-residualised + hub-corrected is the signature+gate to standardize.")
    L.append("")
    for mode in MODES:
        for v in VARIANTS:
            fp = out / f"donor_coupling_{mode}_{v}.csv"
            if not fp.exists():
                L.append(f"## [{mode}] {v} — no testable pairs (coverage too low)")
                L.append("")
                continue
            df = pd.read_csv(fp)
            top = df.sort_values("excess_mean", ascending=False).head(15)
            L.append(f"## [{mode}] {v} — top-15 by residual coupling")
            L.append(rsa._md(top, ["axis_a", "axis_b", "excess_mean", "n_donors",
                                   "p_donor", "q_donor", "is_benchmark", "pair_status"]))
            L.append("")
    (out / "donor_coupling_report.md").write_text("\n".join(L) + "\n")
    log(f"\nwrote {out/'donor_coupling_report.md'}")
    log("DONE.")


if __name__ == "__main__":
    main()
