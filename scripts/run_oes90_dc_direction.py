"""Stage 7 of the §40 dropout+curation run — cross_asym direction, one arm, one shard.

Two outputs, both from `cascadir`:

  * `direction_table.csv` — `est.direction_table()`, the orchestrator's direction API and
    the only one. Direction is NOT read off the coupling table's `cross_asym` column: that
    column is the difference-of-medians `M[a,b] - M[b,a]`, a fast approximation that
    disagrees in *sign* on some pairs (CLAUDE.md §28).
  * `engagement_per_celltype.parquet` — the per-cell-type engagement values
    (`sA_PB_norm` = s(a, S_b) - PBS, `sB_PA_norm` = s(b, S_a) - PBS) that the summary table
    medians away. These come from `cascadir.cross_asym.directional_asymmetry_test`, a
    public cascadir export, used here purely to persist numbers cascadir computed — no
    call, gate, or aggregate is derived from them in this file.

`--arm` picks which signatures to score: `curated` (§40's result) or `raw` (the uncurated
top-200 control). The arms may cover DIFFERENT condition sets — curation drops conditions
it empties — so their pair lists can differ and must be compared on the intersection.

§40 shards the pair list by default (`--n_shards`, `--pairs_shard`): the per-pair
random-gene-set null scales with gene-set size, and top-200 roughly doubles §37's runtime
at top-100. `merge_oes90_dc_direction.py` concatenates the shards, so the DONE marker is
written there rather than here whenever a shard index is given.

Direction gives ordering, never existence: a pair with no coupling can still score a large
`cross_asym` (CLAUDE.md §26.4). Existence is the coupling stage's job.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_dc_config as C  # noqa: E402
import _oes90_dc_estimator as E  # noqa: E402


def all_pairs(conditions) -> list:
    conds = sorted(conditions)
    return [
        (conds[i], conds[j])
        for i in range(len(conds))
        for j in range(i + 1, len(conds))
    ]


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--arm", default="curated", choices=sorted(C.ARMS))
    ap.add_argument("--pairs_shard", type=int, default=None,
                    help="Score only this shard of the pair list; merged by merge_oes90_dc_direction.py.")
    ap.add_argument("--n_shards", type=int, default=C.N_DIRECTION_SHARDS)
    ap.add_argument("--skip_engagement", action="store_true",
                    help="Write only direction_table.csv.")
    args = ap.parse_args()

    import pandas as pd

    from cascadir.cross_asym import _signatures_to_idx, directional_asymmetry_test

    out = Path(args.out_dir)
    est, prov = E.build_estimator(out, arm=args.arm)

    all_pair_list = all_pairs(est.signatures.keys())
    pairs = all_pair_list
    suffix = f"_{args.arm}"
    if args.pairs_shard is not None:
        pairs = [p for i, p in enumerate(pairs) if i % args.n_shards == args.pairs_shard]
        suffix = f"_{args.arm}_shard{args.pairs_shard}"
        C.log(
            f"[shard] {args.pairs_shard}/{args.n_shards}: {len(pairs)} of "
            f"{len(all_pair_list)} pairs"
        )
    C.log(f"[direction] scoring {len(pairs)} pairs with n_null_perms={C.N_NULL_PERMS}")

    t0 = time.time()
    direction = est.direction_table(pairs=pairs)
    C.log(f"[direction] done in {(time.time()-t0)/60:.1f} min")
    dir_path = out / f"direction_table{suffix}.csv"
    direction.to_csv(dir_path, index=False)

    if "classification" in direction.columns:
        C.log(f"[calls] {direction.classification.value_counts().to_dict()}")
    if "direction" in direction.columns:
        C.log(f"[direction] {direction.direction.value_counts().to_dict()}")

    if not args.skip_engagement:
        t1 = time.time()
        # cascadir's own signature -> gene-index mapping, so the engagement rows use
        # exactly the columns direction_table used.
        sig_idx = _signatures_to_idx(est.signatures, est.tube_set.gene_names)
        rows = []
        for k, (a, b) in enumerate(pairs, 1):
            df = directional_asymmetry_test(
                est._cells_by_pair,
                sig_idx,
                a,
                b,
                control_label=C.CONTROL,
                min_cells=C.MIN_CELLS,
            )
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "condition_b", b)
            df.insert(0, "condition_a", a)
            rows.append(df)
            if k % 500 == 0:
                C.log(f"  [{k}/{len(pairs)}] pairs  ({(time.time()-t1)/60:.1f} min)")
        eng = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        eng_path = out / f"engagement_per_celltype{suffix}.parquet"
        eng.to_parquet(eng_path, index=False)
        C.log(
            f"[engagement] {len(eng)} (pair x cell type) rows over "
            f"{eng.groupby(['condition_a','condition_b']).ngroups if len(eng) else 0} pairs "
            f"in {(time.time()-t1)/60:.1f} min -> {eng_path.name}"
        )

    C.write_json(out / f"direction_meta{suffix}.json", {
        **prov,
        "n_pairs_scored": len(pairs),
        "n_null_perms": C.N_NULL_PERMS,
        "null_seed": C.NULL_SEED,
        "min_cells": C.MIN_CELLS,
        "arm": args.arm,
        "n_pairs_total": len(all_pair_list),
        "pairs_shard": args.pairs_shard,
        "n_shards": args.n_shards,
        "elapsed_s": round(time.time() - t0, 1),
        "note": (
            "Direction comes from direction_table (the cross_asym statistic), never from "
            "the coupling table's cross_asym column — see CLAUDE.md §28."
        ),
    })
    if args.pairs_shard is None:
        C.mark_done(out, f"direction_{args.arm}")
    C.log(f"\n[done] {dir_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
