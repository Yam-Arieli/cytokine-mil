"""Stage 3 of the encoder-breadth sweep — Integrated Gradients signatures, per arm.

Rebuilds each saved binary model from its arm's encoder plus its head, then derives the
signature over the MAIN tubes with `cascadir.derive_signatures`. Unlike §37 there is no
reserve pass: this sweep's question is signature DIVERSITY across cytokines, and §37
already established that the models are not memorising cells (reserve Jaccard 0.942).

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

import _encsweep_config as C  # noqa: E402
import _oes90_pure_estimator as E  # noqa: E402


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    # No `choices=`: these same stages drive both the breadth sweep (CLAUDE.md §38.1)
    # and the Stage-1-construction sweep (§38.4), whose arm names differ. The arm is
    # validated by its directory existing, a few lines below.
    ap.add_argument("--arm", required=True)
    ap.add_argument("--chunk_id", type=int, required=True)
    ap.add_argument("--n_chunks", type=int, default=C.N_CHUNKS)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top_n", type=int, default=C.TOP_N)
    args = ap.parse_args()

    import pandas as pd

    from cascadir.signatures import derive_signatures

    out = Path(args.out_dir)
    adir = C.arm_dir(out, args.arm)
    encoder, enc_meta = E.load_encoder(adir, device=args.device)
    encoder_sha = enc_meta["sha256"]

    panel = C.read_json(out / "encsweep_meta.json")["panel"]
    chunk = C.chunk_conditions(panel, args.chunk_id, args.n_chunks)
    C.log(f"[arm:{args.arm} chunk {args.chunk_id}/{args.n_chunks}] {len(chunk)}: {chunk}")

    models = {
        cond: E.load_model(adir, cond, encoder, encoder_sha, device=args.device)
        for cond in chunk
    }
    C.log(f"[models] {len(models)} heads reloaded and verified against this arm's encoder")

    tube_set, _ = E.load_tubes(adir, which="main", conditions=chunk)
    t0 = time.time()
    signatures = derive_signatures(
        models, tube_set, top_n=args.top_n, n_steps=C.N_IG_STEPS, device=args.device
    )
    elapsed = time.time() - t0

    rows = [
        {"cytokine": cond, "gene": gene, "ig": float(ig), "rank_ig": rank}
        for cond, sig in signatures.items()
        for rank, (gene, ig) in enumerate(zip(sig.genes, sig.ig_scores))
    ]
    path = adir / f"signatures_chunk_{args.chunk_id}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    C.log(f"[ig] {len(signatures)} signatures x {args.top_n} genes in "
          f"{elapsed/60:.1f} min -> {path.name}")

    C.write_json(adir / f"ig_chunk_{args.chunk_id}_meta.json", {
        "arm": args.arm,
        "chunk_id": args.chunk_id,
        "conditions": chunk,
        "top_n": args.top_n,
        "n_ig_steps": C.N_IG_STEPS,
        "encoder_sha256": encoder_sha,
        "device": args.device,
        "ig_seconds": round(elapsed, 1),
    })
    C.mark_done(out, f"ig_{args.arm}_{args.chunk_id}")
    C.log("\n[done] signatures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
