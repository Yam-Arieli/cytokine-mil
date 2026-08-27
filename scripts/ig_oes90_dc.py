"""Stage 4 of the §40 dropout+curation run — Integrated Gradients signatures.

Rebuilds each saved binary model from the shared encoder plus its head, then derives its
signature TWICE with `cascadir.derive_signatures`:

  * over the MAIN tubes — the tubes the model trained on. This is the run's signature set,
    and attributing a model on the data it learned from is the normal, correct thing for a
    model-explanation method to do.
  * over the disjoint RESERVE tubes — the same model, cells it never saw. Comparing the two
    gene sets measures directly whether `S_X` is a real transcriptional signature or
    memorised tube idiosyncrasy, which is the one thing in-sample derivation leaves open
    (CLAUDE.md §37).

§40 derives **top-200** rather than top-100, and derives them UNCURATED. Curation is a
global operation — a gene's occurrence count spans all 90 conditions — so it cannot be
done per chunk; it happens once, in the merge stage.

IG cannot use the stage-2 embedding cache: it attributes back to genes, so it runs the full
model from gene-space inputs.
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


def _signature_rows(signatures) -> list:
    rows = []
    for cond, sig in signatures.items():
        for rank, (gene, ig) in enumerate(zip(sig.genes, sig.ig_scores)):
            rows.append(
                {"cytokine": cond, "gene": gene, "ig": float(ig), "rank_ig": rank}
            )
    return rows


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk_id", type=int, required=True)
    ap.add_argument("--n_chunks", type=int, default=C.N_CHUNKS)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top_n", type=int, default=C.TOP_N)
    args = ap.parse_args()

    import pandas as pd

    from cascadir.signatures import derive_signatures

    from cytokine_mil.analysis.full90_tube_io import read_meta

    out = Path(args.out_dir)
    encoder, enc_meta = E.load_encoder(out, device=args.device)
    encoder_sha = enc_meta["sha256"]

    split = C.read_json(out / "tube_split.json")
    all_conds = sorted(
        {s["condition"] for s in read_meta(split["shard_dir"])["shards"]} - {C.CONTROL}
    )
    chunk = C.chunk_conditions(all_conds, args.chunk_id, args.n_chunks)
    C.log(f"[chunk {args.chunk_id}/{args.n_chunks}] {len(chunk)} conditions: {chunk}")

    models = {
        cond: E.load_model(out, cond, encoder, encoder_sha, device=args.device)
        for cond in chunk
    }
    C.log(f"[models] {len(models)} heads reloaded and verified against the shared encoder")

    timings = {}
    for which in ("main", "reserve"):
        tube_set, _ = E.load_tubes(out, which=which, conditions=chunk)
        t0 = time.time()
        signatures = derive_signatures(
            models, tube_set, top_n=args.top_n, n_steps=C.N_IG_STEPS, device=args.device
        )
        elapsed = time.time() - t0
        timings[which] = round(elapsed, 1)
        df = pd.DataFrame(_signature_rows(signatures))
        path = out / f"signatures_{which}_chunk_{args.chunk_id}.parquet"
        df.to_parquet(path, index=False)
        C.log(
            f"[ig:{which}] {len(signatures)} signatures x {args.top_n} genes in "
            f"{elapsed/60:.1f} min -> {path.name}"
        )

    C.write_json(out / f"ig_chunk_{args.chunk_id}_meta.json", {
        "chunk_id": args.chunk_id,
        "n_chunks": args.n_chunks,
        "conditions": chunk,
        "top_n": args.top_n,
        "n_ig_steps": C.N_IG_STEPS,
        "encoder_sha256": encoder_sha,
        "tubes_shards_sha256": split["shards_sha256"],
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "reserve_tube_indices": C.RESERVE_TUBE_INDICES,
        "device": args.device,
        "ig_seconds": timings,
    })
    C.mark_done(out, f"ig_{args.chunk_id}")
    C.log("\n[done] main + reserve signatures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
