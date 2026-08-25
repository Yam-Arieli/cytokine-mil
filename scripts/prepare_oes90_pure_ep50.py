"""Stage 0 of the epoch-50 re-run — share the PURE run's encoder, tubes and embeddings.

Why an epoch-50 fit exists
--------------------------
In the §37 fit the relationship between a binary model's training loss and the validity of
the signature it yields INVERTS at epoch ~51. Before it, conditions with a real response
train faster (Spearman(self-engagement, loss) = -0.44 at epoch 10). After it, conditions
with NO real response descend past them and finish lower (+0.54 at epoch 250) -- they keep
reducing loss by fitting tube-specific noise, and IG then explains that noise. IG was run
at epoch 250, two hundred epochs after the inversion.

This re-run changes exactly ONE thing: Stage-2 stops at epoch 50 instead of 250. Because
`train_binary_mil` uses constant-LR SGD with no scheduler, training 50 epochs reproduces
the first 50 epochs of the 250-epoch run bit-for-bit -- this is that run truncated, not a
different one.

Everything upstream is SHARED, not rebuilt
------------------------------------------
The encoder, the tube split and the persisted embedding cache are symlinked from the §37
directory and then digest-verified through the normal loaders. Retraining the encoder
would introduce a second variable and void the comparison; the §27.6 guard would not catch
it, because a fresh encoder is internally consistent. Symlinking makes sharing structural.

The §37 artifacts are never written to -- all outputs land in a separate directory.

Honest note on how epoch 50 was chosen: it is the crossover of the loss/self-engagement
correlation, and self-engagement is computed from the epoch-250 signatures. So the choice
is informed by the 250-epoch fit. It is NOT informed by the audited benchmark, which is
what would bias the reported accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_pure_config as C  # noqa: E402
import _oes90_pure_estimator as E  # noqa: E402

# Artifacts the epoch-50 fit inherits verbatim. Anything NOT listed here is regenerated.
SHARED = [
    "encoder.pt",
    "encoder_last.pt",
    "encoder_sha256.txt",
    "encoder_meta.json",
    "encoder_history.csv",
    "tube_split.json",
    "embeddings",
    "embeddings_sha256.txt",
]


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src_dir", default=str(C.OUT_DIR))
    ap.add_argument("--out_dir", default=str(C.OUT_DIR) + "_ep50")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    src = Path(args.src_dir).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for marker in ("DONE_encoder", "DONE_encode"):
        if not (src / marker).exists():
            raise SystemExit(
                f"{src/marker} missing — the source run has not produced the encoder and "
                "embedding cache this re-run depends on."
            )

    for name in SHARED:
        s = src / name
        if not s.exists():
            raise SystemExit(f"{s} missing — cannot share it.")
        d = out / name
        if d.is_symlink() or d.exists():
            d.unlink()
        d.symlink_to(s)
        C.log(f"[share] {name} -> {s}")

    # Verify through the real loaders, so a broken link fails HERE and not inside a
    # GPU array task twenty minutes in.
    from cytokine_mil.analysis.full90_tube_io import read_embedding_meta, read_meta

    encoder, enc_meta = E.load_encoder(out)
    split = C.read_json(out / "tube_split.json")
    tube_meta = read_meta(split["shard_dir"])
    if tube_meta["shards_sha256"] != split["shards_sha256"]:
        raise AssertionError("tube shard digest disagrees with the shared tube_split.json")
    emb_meta = read_embedding_meta(out / "embeddings")
    expected_emb = (out / "embeddings_sha256.txt").read_text().strip()
    if emb_meta["shards_sha256"] != expected_emb:
        raise AssertionError("embedding cache digest disagrees with the shared record")
    if int(emb_meta["embed_dim"]) != int(encoder.embed_dim):
        raise AssertionError("embedding cache and encoder were not built together")

    all_conds = sorted({s["condition"] for s in tube_meta["shards"]} - {C.CONTROL})
    C.log(
        f"[verified] encoder {enc_meta['sha256'][:16]}...  tubes "
        f"{split['shards_sha256'][:16]}...  embeddings {expected_emb[:16]}...  "
        f"{len(all_conds)} conditions"
    )

    C.write_json(out / "run_meta.json", {
        "derived_from": str(src),
        "shared_artifacts": SHARED,
        "encoder_sha256": enc_meta["sha256"],
        "tubes_shards_sha256": split["shards_sha256"],
        "embeddings_sha256": expected_emb,
        "stage2_epochs": args.epochs,
        "stage2_epochs_source_run": C.STAGE2_EPOCHS,
        "top_n": C.TOP_N,
        "n_conditions": len(all_conds),
        "only_difference": "Stage-2 binary training stops at epoch "
                           f"{args.epochs} instead of {C.STAGE2_EPOCHS}",
        "rationale": "IG was previously extracted 200 epochs after the point where "
                     "memorisation overtakes signal in the binary training curves",
    })
    C.mark_done(out, "prepare")
    C.log(f"\n[done] {out} ready — train with --epochs {args.epochs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
