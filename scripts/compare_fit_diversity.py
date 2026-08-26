"""Signature diversity across fits — one descriptive table, no method math.

Every Oesinghaus-90 re-fit measured so far sits at mean between-cytokine Jaccard 0.18-0.39
while the published anchor sits at 0.065 (CLAUDE.md §38.3/§38.4). Breadth, width, k,
epochs, top_n, over-training, memorisation, D2/D3 leakage, donor structure and Stage-1
volume have all been eliminated. The variable that still separates them perfectly is the
CODE PATH: every re-fit ran through `cascadir`, the anchor through `cytokine_mil`.

This script scores any set of signature tables on the same footing so that hypothesis can
be tested against fits that already exist, instead of by training anything new. It reuses
`analyze_encsweep.diversity` verbatim — the same function that produced every number in
§38.3/§38.4 — and only reshapes inputs to the (cytokine, gene, rank_ig) contract it wants.

Accepts either schema:
  * a signature table with `rank_ig` (the sweeps, §36, §37);
  * a recurrent-IG trajectory with an `epoch` column (§31) — the LAST epoch is taken.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_encsweep import diversity  # noqa: E402


def load_signatures(path: Path, top_n: int) -> pd.DataFrame:
    """Return a (cytokine, gene, rank_ig) frame, whichever schema `path` uses."""
    df = pd.read_parquet(path)
    if "epoch" in df.columns:
        last = df.epoch.max()
        df = df[df.epoch == last].copy()
        print(f"    [{path.name}] trajectory schema — taking epoch {last}")
    if "rank_ig" not in df.columns:
        if "ig" not in df.columns:
            raise SystemExit(f"{path}: needs `rank_ig` or `ig`; has {list(df.columns)}")
        df = df.sort_values(["cytokine", "ig"], ascending=[True, False])
        df["rank_ig"] = df.groupby("cytokine").cumcount()
    return df[["cytokine", "gene", "rank_ig"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", action="append", required=True, metavar="LABEL=PATH",
                    help="repeatable, e.g. --fit published=/path/binary_ig_all24.parquet")
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="restrict every fit to these cytokines (panel matching)")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    rows = []
    for spec in args.fit:
        label, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            print(f"[skip] {label}: {p} not found")
            continue
        df = load_signatures(p, args.top_n)
        if args.conditions:
            df = df[df.cytokine.isin(args.conditions)]
            if df.empty:
                print(f"[skip] {label}: none of the requested cytokines present")
                continue
        rec = {"fit": label}
        rec.update(diversity(df, args.top_n))
        rows.append(rec)

    if not rows:
        raise SystemExit("no fits could be scored")
    out = pd.DataFrame(rows)[
        ["fit", "n_cytokines", "top5_pool", "top5_worst_gene", "top5_worst_n",
         "mean_jaccard", "jaccard_vs_chance", "distinct_genes", "slots", "collapse_x"]
    ]
    print()
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"\n[write] {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
