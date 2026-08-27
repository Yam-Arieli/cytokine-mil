"""Stage 3 of the §40 dropout+curation run — binary AB-MIL, chunked over a GPU array.

One array task trains ~10 binary models (cytokine vs PBS) on the SHARED Stage-1 encoder
and saves each model's head plus its per-epoch loss curve.

Before any training the task asserts, and refuses to run on a mismatch:
  * the encoder digest equals stage 1's (CLAUDE.md §27.6 — the array must never shard
    encoder training);
  * the tube shard digest equals stage 0's;
  * the embedding-cache digest equals stage 2's, so the models are provably trained on the
    embeddings that were persisted.

The encoder stays frozen and in `eval()` mode throughout, so its dropout is inert here —
Stage 2 trains only the attention + classifier, exactly as in §37.

All training is `cascadir.train_all_binary`; this script chooses no formula of its own.
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


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk_id", type=int, required=True)
    ap.add_argument("--n_chunks", type=int, default=C.N_CHUNKS)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=C.STAGE2_EPOCHS)
    args = ap.parse_args()

    from cascadir.train import train_all_binary

    from cytokine_mil.analysis.full90_tube_io import (
        load_embedding_cache,
        read_embedding_meta,
    )

    out = Path(args.out_dir)
    encoder, enc_meta = E.load_encoder(out, device=args.device)
    encoder_sha = enc_meta["sha256"]

    # --- the conditions for this task, from the tubes alone -------------------
    split = C.read_json(out / "tube_split.json")
    from cytokine_mil.analysis.full90_tube_io import read_meta

    all_conds = sorted(
        {s["condition"] for s in read_meta(split["shard_dir"])["shards"]}
        - {C.CONTROL}
    )
    chunk = C.chunk_conditions(all_conds, args.chunk_id, args.n_chunks)
    C.log(
        f"[chunk {args.chunk_id}/{args.n_chunks}] {len(chunk)} of {len(all_conds)} "
        f"conditions: {chunk}"
    )

    tube_set, _ = E.load_tubes(out, which="main", conditions=chunk)

    # --- the persisted embeddings, digest-checked -----------------------------
    emb_dir = out / "embeddings"
    emb_meta = read_embedding_meta(emb_dir)
    expected_emb = (out / "embeddings_sha256.txt").read_text().strip()
    if emb_meta["shards_sha256"] != expected_emb:
        raise AssertionError(
            "EMBEDDING MISMATCH: the cache hashes to "
            f"{emb_meta['shards_sha256'][:16]}... but stage 2 recorded "
            f"{expected_emb[:16]}.... Refusing to run."
        )
    if int(emb_meta["embed_dim"]) != int(encoder.embed_dim):
        raise AssertionError(
            f"embedding cache has embed_dim={emb_meta['embed_dim']} but the encoder "
            f"produces {encoder.embed_dim} — they were not built together."
        )
    cache = load_embedding_cache(emb_dir, conditions=chunk, control_label=C.CONTROL)
    C.log(f"[embeddings] verified sha256={expected_emb[:16]}...  {len(cache)} tubes loaded")

    # --- train ---------------------------------------------------------------
    t0 = time.time()
    models, histories = train_all_binary(
        tube_set,
        encoder,
        conditions=chunk,
        control_label=C.CONTROL,
        attention_hidden_dim=C.ATTENTION_HIDDEN_DIM,
        epochs=args.epochs,
        lr=C.STAGE2_LR,
        momentum=C.MOMENTUM,
        encoder_frozen=True,
        use_embedding_cache=True,
        embedding_cache=cache,
        device=args.device,
        seed=args.seed,
        return_history=True,
    )
    train_s = time.time() - t0
    C.log(f"[train] {len(models)} binary models in {train_s/60:.1f} min")

    # --- persist -------------------------------------------------------------
    hist_dir = out / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = None
    for cond, model in models.items():
        E.save_model_head(model, out, cond, encoder_sha)
        h = histories[cond]
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(cond))
        h.to_csv(hist_dir / f"{safe}_train.csv", index=False)
        if steps_per_epoch is None:
            steps_per_epoch = int(h.n_megabatches.iloc[0])
        C.log(
            f"  {cond}: loss {h.loss.iloc[0]:.5f} -> {h.loss.iloc[-1]:.5f} "
            f"({int(h.n_megabatches.iloc[0])} mega-batches/epoch)"
        )

    C.write_json(out / f"chunk_{args.chunk_id}_meta.json", {
        "chunk_id": args.chunk_id,
        "n_chunks": args.n_chunks,
        "conditions": chunk,
        "n_models": len(models),
        "encoder_sha256": encoder_sha,
        "tubes_shards_sha256": split["shards_sha256"],
        "embeddings_sha256": expected_emb,
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "stage2_epochs": args.epochs,
        "stage2_lr": C.STAGE2_LR,
        "attention_hidden_dim": C.ATTENTION_HIDDEN_DIM,
        "megabatches_per_epoch": steps_per_epoch,
        "total_gradient_steps": None if steps_per_epoch is None else steps_per_epoch * args.epochs,
        "seed": args.seed,
        "device": args.device,
        "train_seconds": round(train_s, 1),
    })
    C.mark_done(out, f"train_{args.chunk_id}")
    C.log(f"\n[done] {len(models)} heads + histories written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
