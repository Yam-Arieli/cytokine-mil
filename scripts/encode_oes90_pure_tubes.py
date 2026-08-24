"""Stage 2 of the Oesinghaus 90-cytokine PURE run — encode and persist the pseudo-tubes.

With a frozen encoder, ``H = encoder(X)`` is constant across every Stage-2 epoch, so the
per-step encoder forward is pure recomputation. `cascadir.build_frozen_embedding_cache`
computes it once per tube; the trainers then run only the attention + classifier on the
cached embeddings, which is **bit-identical** and much cheaper (CLAUDE.md §29).

Persisting the cache here rather than letting each training task rebuild it means the
saved encoded tubes are provably the ones the models were trained on, and it is a run
artifact in its own right.

Integrated Gradients does NOT use this cache: it attributes back to genes, so it must run
the full model from gene-space inputs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_pure_config as C  # noqa: E402
import _oes90_pure_estimator as E  # noqa: E402


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from cascadir.train import build_frozen_embedding_cache

    from cytokine_mil.analysis.full90_tube_io import save_embedding_cache

    out = Path(args.out_dir)
    encoder, enc_meta = E.load_encoder(out, device=args.device)
    tube_set, _ = E.load_tubes(out, which="main")

    t0 = time.time()
    cache = build_frozen_embedding_cache(encoder, tube_set, device=args.device)
    C.log(f"[encode] {len(cache)} tubes encoded in {(time.time()-t0)/60:.1f} min")

    t1 = time.time()
    meta = save_embedding_cache(cache, out / "embeddings")
    (out / "embeddings_sha256.txt").write_text(meta["shards_sha256"] + "\n")
    C.log(
        f"[save] {meta['n_tubes']} tubes over {meta['n_shards']} shards, "
        f"embed_dim={meta['embed_dim']} ({(time.time()-t1)/60:.1f} min)"
    )

    C.write_json(out / "embeddings_meta_summary.json", {
        "n_tubes": meta["n_tubes"],
        "n_shards": meta["n_shards"],
        "embed_dim": meta["embed_dim"],
        "shards_sha256": meta["shards_sha256"],
        "encoder_sha256": enc_meta["sha256"],
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "device": args.device,
        "elapsed_s": round(time.time() - t0, 1),
    })
    C.mark_done(out, "encode")
    C.log(f"\n[done] embeddings sha256={meta['shards_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
