"""Stage 8 of the §40 dropout+curation run — concatenate the direction shards of one arm.

`run_oes90_dc_direction.py` splits the pair list across a SLURM array (the per-pair
random-gene-set null scales with signature size, and top-200 roughly doubles §37's
runtime at top-100). Each task writes `direction_table_<arm>_shard<k>.csv` and, unless
`--skip_engagement`, `engagement_per_celltype_<arm>_shard<k>.parquet`. This stitches them
back into the single per-arm tables the run's deliverables list.

Pure concatenation. It computes no statistic: every value is exactly what
`est.direction_table()` produced in its shard. The one thing it *does* do is verify that
the shards agree on provenance and that together they cover the pair list exactly once —
a shard silently missing would otherwise look like a smaller but valid result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_dc_config as C  # noqa: E402


def _check_uniform_provenance(metas: list, arm: str) -> dict:
    """Every shard must have read the same tubes, the same signatures and the same null."""
    for key in ("tubes_shards_sha256", "signatures_file", "top_n", "n_pairs_total",
                "n_null_perms", "null_seed", "n_conditions"):
        vals = {str(m.get(key)) for m in metas}
        if len(vals) != 1:
            raise AssertionError(
                f"[{arm}] shards disagree on {key}: {sorted(vals)}. They are not one fit."
            )
    return {k: metas[0][k] for k in metas[0] if k not in ("pairs_shard", "elapsed_s")}


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--arm", default="curated", choices=sorted(C.ARMS))
    ap.add_argument("--n_shards", type=int, default=C.N_DIRECTION_SHARDS)
    args = ap.parse_args()

    import pandas as pd

    out = Path(args.out_dir)
    arm = args.arm

    metas, dir_parts, eng_parts = [], [], []
    for k in range(args.n_shards):
        meta_p = out / f"direction_meta_{arm}_shard{k}.json"
        dir_p = out / f"direction_table_{arm}_shard{k}.csv"
        if not meta_p.exists() or not dir_p.exists():
            raise FileNotFoundError(
                f"[{arm}] shard {k} incomplete ({meta_p.name} / {dir_p.name}) — "
                "refusing to merge a partial direction table."
            )
        metas.append(C.read_json(meta_p))
        dir_parts.append(pd.read_csv(dir_p))
        eng_p = out / f"engagement_per_celltype_{arm}_shard{k}.parquet"
        if eng_p.exists():
            eng_parts.append(pd.read_parquet(eng_p))

    prov = _check_uniform_provenance(metas, arm)
    n_expected = int(metas[0]["n_pairs_total"])
    n_scored = sum(int(m["n_pairs_scored"]) for m in metas)
    if n_scored != n_expected:
        raise AssertionError(
            f"[{arm}] shards scored {n_scored} pairs but the arm has {n_expected}. "
            "A shard is missing or the shard count changed between submit and merge."
        )

    direction = pd.concat(dir_parts, ignore_index=True)
    key = [c for c in ("condition_a", "condition_b") if c in direction.columns]
    if len(key) == 2:
        dup = int(direction.duplicated(key).sum())
        if dup:
            raise AssertionError(f"[{arm}] {dup} duplicate pairs across shards")
        direction = direction.sort_values(key).reset_index(drop=True)
    if len(direction) != n_expected:
        raise AssertionError(
            f"[{arm}] merged table has {len(direction)} rows, expected {n_expected}"
        )

    dir_path = out / f"direction_table_{arm}.csv"
    direction.to_csv(dir_path, index=False)
    C.log(f"[{arm}] {len(direction)} pairs merged from {args.n_shards} shards -> {dir_path.name}")
    for col in ("classification", "direction"):
        if col in direction.columns:
            C.log(f"[{arm}] {col}: {direction[col].value_counts().to_dict()}")

    if eng_parts:
        eng = pd.concat(eng_parts, ignore_index=True)
        eng_path = out / f"engagement_per_celltype_{arm}.parquet"
        eng.to_parquet(eng_path, index=False)
        C.log(f"[{arm}] {len(eng)} (pair x cell type) engagement rows -> {eng_path.name}")

    C.write_json(out / f"direction_meta_{arm}.json", {
        **prov,
        "n_pairs_scored": len(direction),
        "n_shards_merged": args.n_shards,
        "elapsed_s_total": round(sum(float(m.get("elapsed_s", 0)) for m in metas), 1),
    })
    C.mark_done(out, f"direction_{arm}")
    C.log(f"\n[done] {dir_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
