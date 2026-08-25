"""Stage 2 of the encoder-breadth sweep — binary AB-MIL for the panel, per arm.

One array task trains a slice of the readout panel against PBS on ONE arm's Stage-1
encoder, and saves each model's head plus its per-epoch loss curve.

Before training it asserts the encoder digest matches that arm's stage-1 digest and the
tube shard digest matches prepare's, and refuses to run on a mismatch.

All training is `cascadir.train_all_binary`; this script chooses no formula of its own.
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
    ap.add_argument("--arm", required=True, choices=list(C.ARMS))
    ap.add_argument("--chunk_id", type=int, required=True)
    ap.add_argument("--n_chunks", type=int, default=C.N_CHUNKS)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=C.STAGE2_EPOCHS)
    args = ap.parse_args()

    from cascadir.train import train_all_binary

    out = Path(args.out_dir)
    adir = C.arm_dir(out, args.arm)
    encoder, enc_meta = E.load_encoder(adir, device=args.device)
    encoder_sha = enc_meta["sha256"]

    panel = C.read_json(out / "encsweep_meta.json")["panel"]
    chunk = C.chunk_conditions(panel, args.chunk_id, args.n_chunks)
    C.log(f"[arm:{args.arm} chunk {args.chunk_id}/{args.n_chunks}] "
          f"{len(chunk)} of {len(panel)} panel conditions: {chunk}")

    tube_set, _ = E.load_tubes(adir, which="main", conditions=chunk)

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
        device=args.device,
        seed=args.seed,
        return_history=True,
    )
    train_s = time.time() - t0
    C.log(f"[train] {len(models)} binary models in {train_s/60:.1f} min")

    hist_dir = adir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = None
    for cond, model in models.items():
        E.save_model_head(model, adir, cond, encoder_sha)
        h = histories[cond]
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(cond))
        h.to_csv(hist_dir / f"{safe}_train.csv", index=False)
        if steps_per_epoch is None:
            steps_per_epoch = int(h.n_megabatches.iloc[0])
        C.log(f"  {cond}: loss {h.loss.iloc[0]:.5f} -> {h.loss.iloc[-1]:.5f} "
              f"({int(h.n_megabatches.iloc[0])} mega-batches/epoch)")

    C.write_json(adir / f"chunk_{args.chunk_id}_meta.json", {
        "arm": args.arm,
        "chunk_id": args.chunk_id,
        "n_chunks": args.n_chunks,
        "conditions": chunk,
        "n_models": len(models),
        "encoder_sha256": encoder_sha,
        "main_tube_indices": C.MAIN_TUBE_INDICES,
        "stage2_epochs": args.epochs,
        "stage2_lr": C.STAGE2_LR,
        "attention_hidden_dim": C.ATTENTION_HIDDEN_DIM,
        "megabatches_per_epoch": steps_per_epoch,
        "total_gradient_steps": None if steps_per_epoch is None
        else steps_per_epoch * args.epochs,
        "seed": args.seed,
        "device": args.device,
        "train_seconds": round(train_s, 1),
    })
    C.mark_done(out, f"train_{args.arm}_{args.chunk_id}")
    C.log(f"\n[done] {len(models)} heads + histories written to {adir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
