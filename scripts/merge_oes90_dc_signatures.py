"""Stage 5 of the §40 dropout+curation run — merge, then CURATE, the signatures.

Concatenates the nine IG array outputs into one parquet per tube set and refuses to write
unless the provenance is uniform: every chunk must report the SAME encoder digest, the
SAME tube digest, and the same signature size, and every condition must appear exactly
once. Signatures derived under different encoders are not comparable, and both coupling
and `cross_asym` compare signatures across cytokines, so a silent mismatch here would
corrupt every downstream number (CLAUDE.md §27.6).

Then applies §40's promiscuous-gene curation (CLAUDE.md §40.2). This is the ONLY place it
can happen: a gene's occurrence count spans all 90 conditions, so it is undefined per
chunk. The cap is DERIVED from (n_conditions, top_n, n_genes) via
`cascadir.null_calibrated_max_occurrences` and never hardcoded — a fixed cap means very
different things at different scales (at K=90, n=200, G=4000 a gene's expected count under
a uniform-random null is 4.5, so a cap of 3 would delete ~83% of every signature even if
the signatures were perfectly random).

Both the uncurated and curated parquets are written. The uncurated one is not a leftover:
it is the control arm that coupling and direction are also run on, which is the only way
to say whether the curation helped, hurt, or did nothing.

Also emits `signature_stability{,_curated}.csv`: per cytokine, the Jaccard between the
signature derived from the tubes the model trained on and the one derived from the disjoint
reserve. That is a set overlap of gene identifiers — a property of the artifacts, not a
method statistic, and not a benchmark score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_dc_config as C  # noqa: E402


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


def _signatures_from_df(df) -> dict:
    """{condition: Signature} from a merged signature parquet, ordered by rank_ig."""
    from cascadir.types import Signature

    out = {}
    for cond, g in df.groupby("cytokine"):
        g = g.sort_values("rank_ig")
        out[str(cond)] = Signature(
            condition=str(cond),
            genes=tuple(g.gene),
            ig_scores=tuple(float(v) for v in g.ig),
            top_n=len(g),
        )
    return out


def _signatures_to_df(signatures) -> "pd.DataFrame":
    """Back to the parquet schema, re-ranking 0..n-1 within each (now shorter) signature."""
    import pandas as pd

    rows = []
    for cond in sorted(signatures):
        sig = signatures[cond]
        for rank, (gene, ig) in enumerate(zip(sig.genes, sig.ig_scores)):
            rows.append({"cytokine": cond, "gene": gene, "ig": float(ig), "rank_ig": rank})
    return pd.DataFrame(rows)


def _stability(main_sets: dict, res_sets: dict) -> "pd.DataFrame":
    """Jaccard(main, reserve) per cytokine, over whichever conditions BOTH sides have."""
    import pandas as pd

    rows = []
    for c in sorted(set(main_sets) & set(res_sets)):
        a, b = main_sets[c], res_sets[c]
        rows.append({
            "cytokine": c,
            "n_main": len(a),
            "n_reserve": len(b),
            "n_shared": len(a & b),
            "jaccard": len(a & b) / len(a | b) if (a | b) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("jaccard")


def _curate_arm(out: Path, main_df, reserve_df, cap: int, n_genes: int, expected_n: int):
    """Curate main + reserve with the SAME cap, write both parquets and the report.

    Main and reserve must share a cap: they are the two halves of one comparison, and
    calibrating each to its own gene pool would make their Jaccard incomparable.
    """
    from cascadir.signatures import curate_signatures, null_expected_removal

    curated_main, report = curate_signatures(
        _signatures_from_df(main_df),
        max_occurrences=cap,
        min_genes=C.CURATION_MIN_GENES,
    )
    curated_res, res_report = curate_signatures(
        _signatures_from_df(reserve_df),
        max_occurrences=cap,
        min_genes=C.CURATION_MIN_GENES,
    )

    if not curated_main:
        raise AssertionError(
            f"curation with cap={cap} emptied every signature — refusing to write. "
            "Check n_conditions / top_n / n_genes going into the calibration."
        )

    _signatures_to_df(curated_main).to_parquet(
        out / "signatures_main_curated.parquet", index=False)
    _signatures_to_df(curated_res).to_parquet(
        out / "signatures_reserve_curated.parquet", index=False)
    report.to_csv(out / "curation_report.csv", index=False)

    observed = float(report.n_removed.sum() / max(report.n_before.sum(), 1))
    expected = null_expected_removal(expected_n, int(report.n_before.max()), n_genes, cap)
    n_dropped = int(report.dropped.sum())
    sizes = report.loc[~report.dropped, "n_after"]

    C.log(
        f"[curation] cap={cap} (genes may appear in at most {cap} of {expected_n} "
        f"signatures)\n"
        f"[curation] removed {observed*100:.1f}% of signature slots; a uniform-random "
        f"null would remove {expected*100:.1f}%  ->  EXCESS {(observed-expected)*100:+.1f} "
        f"points\n"
        f"[curation] surviving size: min={int(sizes.min())} median={int(sizes.median())} "
        f"max={int(sizes.max())};  {n_dropped} of {len(report)} conditions dropped entirely"
    )
    if n_dropped:
        C.log(f"[curation] dropped: {sorted(report.loc[report.dropped, 'condition'])}")

    meta = {
        "cap": int(cap),
        "cap_source": "cascadir.null_calibrated_max_occurrences",
        "target_null_removal": C.CURATION_TARGET_NULL_REMOVAL,
        "expected_null_removal": float(expected),
        "observed_removal_frac": observed,
        "excess_removal": float(observed - expected),
        "min_genes": C.CURATION_MIN_GENES,
        "n_conditions_in": int(len(report)),
        "n_conditions_kept": int(len(curated_main)),
        "n_conditions_dropped": n_dropped,
        "dropped_conditions": sorted(report.loc[report.dropped, "condition"]),
        "size_min": int(sizes.min()),
        "size_median": float(sizes.median()),
        "size_max": int(sizes.max()),
        "n_distinct_genes_curated": int(_signatures_to_df(curated_main).gene.nunique()),
        "n_conditions_dropped_reserve": int(res_report.dropped.sum()),
        "note": (
            "excess_removal is observed minus the uniform-random-null expectation; it "
            "measures how far the real signatures are from independent. The curated and "
            "raw arms may cover different condition sets — compare on the intersection."
        ),
    }
    return curated_main, curated_res, meta


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
    stab = _stability(main_sets, res_sets)
    stab.to_csv(out / "signature_stability.csv", index=False)
    C.log(
        f"[stability] Jaccard(main, reserve): min={stab.jaccard.min():.3f} "
        f"median={stab.jaccard.median():.3f} max={stab.jaccard.max():.3f}"
    )
    C.log(f"[stability] least stable: {stab.head(3)[['cytokine','jaccard']].to_dict('records')}")

    # --- §40 curation (CLAUDE.md §40.2) --------------------------------------
    from cascadir.signatures import null_calibrated_max_occurrences

    n_conditions = int(main_df.cytokine.nunique())
    n_genes = int(read_meta(split["shard_dir"])["n_genes"])
    cap = null_calibrated_max_occurrences(
        n_conditions,
        int(prov["top_n"]),
        n_genes,
        target_null_removal=C.CURATION_TARGET_NULL_REMOVAL,
    )
    C.log(
        f"\n[curation] calibrating: K={n_conditions} conditions, top_n={prov['top_n']}, "
        f"G={n_genes} genes, target null damage {C.CURATION_TARGET_NULL_REMOVAL:.4f}"
    )
    curated_main, curated_res, cur_meta = _curate_arm(
        out, main_df, reserve_df, cap, n_genes, n_conditions
    )

    cur_stab = _stability(
        {c: set(s.genes) for c, s in curated_main.items()},
        {c: set(s.genes) for c, s in curated_res.items()},
    )
    cur_stab.to_csv(out / "signature_stability_curated.csv", index=False)
    C.log(
        f"[stability:curated] Jaccard(main, reserve) over {len(cur_stab)} conditions: "
        f"min={cur_stab.jaccard.min():.3f} median={cur_stab.jaccard.median():.3f} "
        f"max={cur_stab.jaccard.max():.3f}"
    )

    C.write_json(out / "curation_meta.json", cur_meta)
    C.write_json(out / "signatures_meta.json", {
        **prov,
        "n_conditions": n_conditions,
        "n_genes": n_genes,
        "n_distinct_genes_main": int(main_df.gene.nunique()),
        "n_distinct_genes_reserve": int(reserve_df.gene.nunique()),
        "stability_jaccard_min": float(stab.jaccard.min()),
        "stability_jaccard_median": float(stab.jaccard.median()),
        "stability_jaccard_max": float(stab.jaccard.max()),
        "curated_stability_jaccard_median": float(cur_stab.jaccard.median()),
        "curation_cap": cur_meta["cap"],
        "curation_excess_removal": cur_meta["excess_removal"],
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "reserve_tube_indices": C.RESERVE_TUBE_INDICES,
    })
    C.mark_done(out, "merge")
    C.log(
        "\n[done] signatures_{main,reserve}.parquet (raw) + "
        "signatures_{main,reserve}_curated.parquet + curation_report.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
