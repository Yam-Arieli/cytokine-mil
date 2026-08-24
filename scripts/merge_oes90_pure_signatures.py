"""Stage 5 of the Oesinghaus 90-cytokine PURE run — merge the per-chunk signatures.

Concatenates the nine IG array outputs into one parquet per tube set and refuses to write
unless the provenance is uniform: every chunk must report the SAME encoder digest, the
SAME tube digest, and the same signature size, and every condition must appear exactly
once. Signatures derived under different encoders are not comparable, and both coupling
and `cross_asym` compare signatures across cytokines, so a silent mismatch here would
corrupt every downstream number (CLAUDE.md §27.6).

Also emits `signature_stability.csv`: per cytokine, the Jaccard between the signature
derived from the tubes the model trained on and the one derived from the disjoint reserve.
That is a set overlap of gene identifiers — a property of the artifacts, not a method
statistic, and not a benchmark score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_pure_config as C  # noqa: E402


def _check_uniform_provenance(metas: list) -> dict:
    encoders = {m["encoder_sha256"] for m in metas}
    tubes = {m["tubes_shards_sha256"] for m in metas}
    top_ns = {int(m["top_n"]) for m in metas}
    if len(encoders) != 1:
        raise AssertionError(
            f"chunks used {len(encoders)} different encoders: "
            f"{sorted(s[:16] for s in encoders)}. Signatures are not comparable."
        )
    if len(tubes) != 1:
        raise AssertionError(
            f"chunks used {len(tubes)} different tube sets: "
            f"{sorted(s[:16] for s in tubes)}. Signatures are not comparable."
        )
    if len(top_ns) != 1:
        raise AssertionError(f"chunks used different top_n values: {sorted(top_ns)}")
    return {
        "encoder_sha256": encoders.pop(),
        "tubes_shards_sha256": tubes.pop(),
        "top_n": top_ns.pop(),
    }


def _merge_one(out: Path, which: str, n_chunks: int, top_n: int, expected: set):
    import pandas as pd

    parts = []
    for k in range(n_chunks):
        p = out / f"signatures_{which}_chunk_{k}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — chunk {k} did not finish.")
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)

    got = set(df.cytokine.unique())
    if got != expected:
        missing, extra = sorted(expected - got), sorted(got - expected)
        raise AssertionError(
            f"[{which}] condition mismatch: {len(missing)} missing {missing[:5]}, "
            f"{len(extra)} unexpected {extra[:5]}"
        )
    sizes = df.groupby("cytokine").size()
    if not (sizes == top_n).all():
        bad = sizes[sizes != top_n]
        raise AssertionError(f"[{which}] {len(bad)} signatures are not {top_n} genes: "
                             f"{bad.head().to_dict()}")
    dup = df.duplicated(["cytokine", "gene"]).sum()
    if dup:
        raise AssertionError(f"[{which}] {dup} duplicate (cytokine, gene) rows")

    path = out / f"signatures_{which}.parquet"
    df.to_parquet(path, index=False)
    C.log(
        f"[{which}] {df.cytokine.nunique()} signatures x {top_n} genes, "
        f"{df.gene.nunique()} distinct genes -> {path.name}"
    )
    return df


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--n_chunks", type=int, default=C.N_CHUNKS)
    args = ap.parse_args()

    import pandas as pd

    from cytokine_mil.analysis.full90_tube_io import read_meta

    out = Path(args.out_dir)
    metas = [C.read_json(out / f"ig_chunk_{k}_meta.json") for k in range(args.n_chunks)]
    prov = _check_uniform_provenance(metas)
    C.log(
        f"[provenance] uniform across {args.n_chunks} chunks: "
        f"encoder={prov['encoder_sha256'][:16]}... tubes={prov['tubes_shards_sha256'][:16]}..."
    )

    split = C.read_json(out / "tube_split.json")
    expected = {
        s["condition"] for s in read_meta(split["shard_dir"])["shards"]
    } - {C.CONTROL}

    main_df = _merge_one(out, "main", args.n_chunks, prov["top_n"], expected)
    reserve_df = _merge_one(out, "reserve", args.n_chunks, prov["top_n"], expected)

    # --- signature stability: same model, tubes it saw vs tubes it did not ----
    main_sets = {c: set(g.gene) for c, g in main_df.groupby("cytokine")}
    res_sets = {c: set(g.gene) for c, g in reserve_df.groupby("cytokine")}
    rows = []
    for c in sorted(main_sets):
        a, b = main_sets[c], res_sets[c]
        rows.append({
            "cytokine": c,
            "n_main": len(a),
            "n_reserve": len(b),
            "n_shared": len(a & b),
            "jaccard": len(a & b) / len(a | b),
        })
    stab = pd.DataFrame(rows).sort_values("jaccard")
    stab.to_csv(out / "signature_stability.csv", index=False)
    C.log(
        f"[stability] Jaccard(main, reserve): min={stab.jaccard.min():.3f} "
        f"median={stab.jaccard.median():.3f} max={stab.jaccard.max():.3f}"
    )
    C.log(f"[stability] least stable: {stab.head(3)[['cytokine','jaccard']].to_dict('records')}")

    C.write_json(out / "signatures_meta.json", {
        **prov,
        "n_conditions": int(main_df.cytokine.nunique()),
        "n_distinct_genes_main": int(main_df.gene.nunique()),
        "n_distinct_genes_reserve": int(reserve_df.gene.nunique()),
        "stability_jaccard_min": float(stab.jaccard.min()),
        "stability_jaccard_median": float(stab.jaccard.median()),
        "stability_jaccard_max": float(stab.jaccard.max()),
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "reserve_tube_indices": C.RESERVE_TUBE_INDICES,
    })
    C.mark_done(out, "merge")
    C.log("\n[done] signatures_main.parquet + signatures_reserve.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
