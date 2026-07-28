#!/usr/bin/env python
"""
Direction (cross_asym) for every Oesinghaus pair that passes the coupling gate.

Why this script exists
----------------------
The coupling gate (donor-level + degree-corrected, the right gate for Oesinghaus's
10 donors) flags 76 of 276 pairs, but direction had only ever been computed for the
53 axes in `cytokine_axes.csv` -- and only 23 of those overlap the coupled set. The
undocumented candidates, which are the whole point of the discovery claim, had no
direction anywhere.

All method math comes from `cascadir` -- nothing here reimplements cross_asym.
This script only:
  * loads the already-computed binary-IG signatures (the SAME
    `binary_ig_all24/binary_ig.parquet` that backs the published 88%, per
    `slurm/run_pipeline_full19.slurm`) into cascadir's `Signature` dataclass,
  * loads the pseudo-tube cells into cascadir's `cells_by_pair` contract via the
    existing research loader,
  * hands both to `cascadir.cross_asym.direction_table`.

Reusing the published signature set (rather than a fresh `CascadeDirection.fit`)
keeps coupling and direction on ONE provenance -- the same single-provenance rule
adopted for Sheu (CLAUDE.md 26.3). A fresh fit would train new signatures and
create a second, non-comparable Oesinghaus result.

A regression check on the 17 audited benchmark pairs runs first: if it does not
reproduce the published 15/17, the run aborts rather than emit numbers that cannot
be compared to the thesis.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cytokine_mil.analysis.oesinghaus_cell_loader import (  # noqa: E402
    load_oesinghaus_cells_by_pair,
)

from cascadir.config import CrossAsymConfig  # noqa: E402
from cascadir.cross_asym import direction_table  # noqa: E402
from cascadir.types import Signature  # noqa: E402

PUBLISHED_CORRECT = 15
PUBLISHED_TOTAL = 17


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_signatures(parquet_path: str, top_n: int) -> dict[str, Signature]:
    """Read the binary-IG parquet into cascadir's Signature dataclass.

    Mirrors what `cascadir.signatures.derive_signature` returns: the top_n genes by
    mean IG, most-attributed first. The IG values themselves are read, never recomputed.

    Selection is by the precomputed `rank_ig` column, exactly as
    `run_pipeline_a_bridge_b.py:226` does, so the gene sets are identical to the
    published run's (a re-sort on `ig` would be equivalent but could break ties
    differently).
    """
    df = pd.read_parquet(parquet_path)
    gene_col = "gene"
    ig_col = "ig" if "ig" in df.columns else "ig_score"
    cond_col = "cytokine" if "cytokine" in df.columns else "condition"
    if "rank_ig" not in df.columns:
        raise ValueError(f"{parquet_path} has no rank_ig column; cannot match the published selection")

    out: dict[str, Signature] = {}
    for cond, sub in df.groupby(cond_col):
        sub = sub.sort_values("rank_ig").head(top_n)
        out[str(cond)] = Signature(
            condition=str(cond),
            genes=tuple(str(g) for g in sub[gene_col]),
            ig_scores=tuple(float(v) for v in sub[ig_col]),
            top_n=top_n,
        )
    return out


def read_coupled_pairs(coupling_csv: str) -> list[tuple[str, str]]:
    """The pairs that passed the donor-level, degree-corrected coupling gate at q05."""
    pairs = []
    with open(coupling_csv) as fh:
        for row in csv.DictReader(fh):
            if row["coupled_q05"] == "True":
                pairs.append(tuple(sorted((row["axis_a"], row["axis_b"]))))
    return sorted(set(pairs))


def read_benchmark(audit_csv: str) -> dict[tuple[str, str], int]:
    """The 17 audited pairs with a signed expected direction."""
    out = {}
    with open(audit_csv) as fh:
        for row in csv.DictReader(fh):
            if row.get("counts_in_benchmark", "").strip().lower() != "true":
                continue
            try:
                sign = int(float(row["expected_sign"]))
            except (ValueError, KeyError):
                continue
            if sign == 0:
                continue
            out[tuple(sorted((row["axis_a"], row["axis_b"])))] = sign
    return out


def score_against_benchmark(
    table: pd.DataFrame, benchmark: dict[tuple[str, str], int]
) -> tuple[int, int, pd.DataFrame]:
    """Sign-accuracy of cross_asym_median against the audited expected_sign."""
    rows = []
    for _, r in table.iterrows():
        key = tuple(sorted((r["condition_a"], r["condition_b"])))
        if key not in benchmark:
            continue
        # table rows are canonicalized alphabetically, same convention as expected_sign
        got = 1 if r["cross_asym_median"] >= 0 else -1
        rows.append(
            {
                "axis_a": key[0],
                "axis_b": key[1],
                "expected_sign": benchmark[key],
                "cross_asym_median": r["cross_asym_median"],
                "got_sign": got,
                "correct": got == benchmark[key],
                "classification": r["classification"],
                "null_p": r["null_p"],
            }
        )
    df = pd.DataFrame(rows)
    n_correct = int(df["correct"].sum()) if not df.empty else 0
    return n_correct, len(df), df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coupling_csv", required=True)
    ap.add_argument("--audit_csv", required=True)
    ap.add_argument("--binary_ig_parquet", required=True)
    ap.add_argument("--manifest_path", required=True)
    ap.add_argument("--hvg_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--min_cells", type=int, default=10)
    ap.add_argument("--n_null_perms", type=int, default=100)
    ap.add_argument("--pbs_label", default="PBS")
    ap.add_argument(
        "--allow_regression",
        action="store_true",
        help="Emit results even if the 17-pair benchmark check does not reproduce.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CrossAsymConfig(
        top_n=args.top_n, min_cells=args.min_cells, n_null_perms=args.n_null_perms
    )

    coupled = read_coupled_pairs(args.coupling_csv)
    benchmark = read_benchmark(args.audit_csv)
    _log(f"[1] coupled pairs (q05): {len(coupled)}")
    _log(f"    audited benchmark pairs: {len(benchmark)}")

    sigs = load_signatures(args.binary_ig_parquet, args.top_n)
    _log(f"[2] signatures loaded: {len(sigs)} conditions from {args.binary_ig_parquet}")

    # Every condition appearing in either the coupled set or the benchmark.
    needed = sorted(
        {c for p in coupled for c in p} | {c for p in benchmark for c in p}
    )
    missing = [c for c in needed if c not in sigs]
    if missing:
        _log(f"    ERROR: no signature for: {missing}")
        return 2

    t0 = time.time()
    cells_by_pair, gene_names = load_oesinghaus_cells_by_pair(
        manifest_path=args.manifest_path,
        cytokines=needed,
        hvg_path=args.hvg_path,
        pbs_label=args.pbs_label,
    )
    _log(
        f"[3] cells loaded: {len(cells_by_pair)} (cond, cell_type) groups, "
        f"{len(gene_names)} genes  [{time.time()-t0:.1f}s]"
    )

    gene_names_t = tuple(gene_names)

    # ---- regression check on the audited 17 first -------------------------
    _log("[4] regression check on the 17 audited benchmark pairs")
    bench_tbl = direction_table(
        cells_by_pair, sigs, gene_names_t, sorted(benchmark.keys()),
        control_label=args.pbs_label, config=cfg,
    )
    n_ok, n_tot, bench_scored = score_against_benchmark(bench_tbl, benchmark)
    bench_scored.to_csv(out_dir / "benchmark_regression.csv", index=False)
    _log(f"    cascadir reproduces {n_ok}/{n_tot}  (published {PUBLISHED_CORRECT}/{PUBLISHED_TOTAL})")

    reproduced = (n_ok == PUBLISHED_CORRECT and n_tot == PUBLISHED_TOTAL)
    if not reproduced:
        _log("    !! does NOT match the published figure")
        if not args.allow_regression:
            _log("    aborting (pass --allow_regression to emit anyway)")
            (out_dir / "REGRESSION_FAILED.txt").write_text(
                f"cascadir direction_table scored {n_ok}/{n_tot}; "
                f"published is {PUBLISHED_CORRECT}/{PUBLISHED_TOTAL}.\n"
            )
            return 3

    # ---- the actual run: all coupled pairs --------------------------------
    _log(f"[5] direction over all {len(coupled)} coupled pairs")
    t0 = time.time()
    tbl = direction_table(
        cells_by_pair, sigs, gene_names_t, coupled,
        control_label=args.pbs_label, config=cfg,
    )
    _log(f"    done [{time.time()-t0:.1f}s]")

    # annotate with the coupling value and literature status
    coup_meta = {}
    with open(args.coupling_csv) as fh:
        for row in csv.DictReader(fh):
            coup_meta[tuple(sorted((row["axis_a"], row["axis_b"])))] = row
    tbl["coupling"] = [
        float(coup_meta[tuple(sorted((a, b)))]["excess_mean"])
        for a, b in zip(tbl["condition_a"], tbl["condition_b"])
    ]
    tbl["pair_status"] = [
        coup_meta[tuple(sorted((a, b)))]["pair_status"]
        for a, b in zip(tbl["condition_a"], tbl["condition_b"])
    ]
    tbl["in_benchmark"] = [
        tuple(sorted((a, b))) in benchmark
        for a, b in zip(tbl["condition_a"], tbl["condition_b"])
    ]
    tbl.to_csv(out_dir / "direction_coupled76.csv", index=False)

    meta = {
        "n_coupled_pairs": len(coupled),
        "benchmark_regression": {
            "n_correct": n_ok,
            "n_total": n_tot,
            "published_correct": PUBLISHED_CORRECT,
            "published_total": PUBLISHED_TOTAL,
            "reproduced": reproduced,
        },
        "signature_source": args.binary_ig_parquet,
        "coupling_source": args.coupling_csv,
        "config": {
            "top_n": cfg.top_n,
            "min_cells": cfg.min_cells,
            "n_null_perms": cfg.n_null_perms,
            "null_seed": cfg.null_seed,
        },
        "computed_by": "cascadir.cross_asym.direction_table",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    n_unlab = int((~tbl["in_benchmark"] & (tbl["pair_status"] == "")).sum())
    _log(f"[6] wrote {out_dir/'direction_coupled76.csv'}")
    _log(f"    {n_unlab} of {len(tbl)} coupled pairs carry no literature label")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
