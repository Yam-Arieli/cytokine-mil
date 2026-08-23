#!/usr/bin/env python
"""Stage 2 of the Oesinghaus full-90 DAG — binary AB-MIL + IG for one chunk of cytokines.

One GPU array task trains ~10 binary models (cytokine vs PBS) on the SHARED Stage-1
encoder and the SHARED pseudo-tubes, then derives their Integrated-Gradients signatures.
All training and IG is `cascadir`'s (`train_all_binary`, `derive_signatures`); this script
only loads artifacts, calls the API, and writes a parquet.

Two provenance assertions run before any training, because CLAUDE.md §27.6 is the failure
this stage exists to prevent:

  1. the encoder's sha256 must equal the digest Stage 1 wrote — the array can never end up
     with a per-chunk encoder;
  2. the tube shards' `shards_sha256` must equal the digest Stage 0 wrote — every chunk
     trains on byte-identical tubes.

`cross_asym` compares signatures across cytokines, so either mismatch would make the 90
signatures non-comparable while still producing plausible-looking numbers.

Usage (cluster, GPU array):
  python scripts/train_oesinghaus_full90_chunk.py --chunk_id 0 --n_chunks 9
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _full90_config as C  # noqa: E402
from cytokine_mil.analysis.full90_tube_io import load_tube_set, read_meta  # noqa: E402


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def load_shared_encoder(out: Path, n_genes: int):
    """Load encoder.pt and refuse to continue unless its digest matches Stage 1's."""
    import torch

    from cascadir.models import InstanceEncoder

    meta = read_meta_json(out / "encoder_meta.json")
    expected = (out / "encoder_sha256.txt").read_text().strip()
    state = torch.load(out / "encoder.pt", map_location="cpu")

    got = C.state_dict_sha256(state)
    if got != expected:
        raise SystemExit(
            f"FATAL: encoder sha256 mismatch.\n  expected {expected}\n  got      {got}\n"
            "Every chunk must train on the SAME Stage-1 encoder (CLAUDE.md §27.6); "
            "signatures from different encoders are not comparable."
        )
    if int(meta["n_genes"]) != n_genes:
        raise SystemExit(
            f"FATAL: encoder was trained on {meta['n_genes']} genes but the tubes have "
            f"{n_genes}."
        )

    encoder = InstanceEncoder(
        input_dim=n_genes,
        embed_dim=int(meta["embed_dim"]),
        n_cell_types=int(meta["n_cell_types"]),
        hidden_dims=tuple(meta["hidden_dims"]),
    )
    encoder.load_state_dict(state)
    encoder.eval()
    _log(f"[encoder] loaded, sha256 verified ({expected[:16]}...)")
    return encoder, expected


def read_meta_json(path: Path) -> dict:
    import json

    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default="results/oes_full90")
    ap.add_argument("--chunk_id", type=int, required=True)
    ap.add_argument("--n_chunks", type=int, default=C.N_CHUNKS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    import pandas as pd

    from cascadir.signatures import derive_signatures
    from cascadir.train import train_all_binary

    out = Path(args.output_dir)
    shard_dir = out / "tubes"
    tube_meta = read_meta(shard_dir)

    prepare_meta = read_meta_json(out / "prepare_meta.json")
    expected_tubes_sha = prepare_meta["tubes"]["shards_sha256"]
    if tube_meta["shards_sha256"] != expected_tubes_sha:
        raise SystemExit(
            "FATAL: tube shards sha256 mismatch — the tubes on disk are not the ones "
            "Stage 0 wrote. Every chunk must train on identical tubes."
        )

    stimuli = [c for c in tube_meta["conditions"] if c != C.CONTROL]
    chunk = C.chunk_conditions(stimuli, args.chunk_id, args.n_chunks)
    if not chunk:
        _log(f"[chunk {args.chunk_id}] no conditions assigned; nothing to do")
        return 0
    _log(f"[chunk {args.chunk_id}/{args.n_chunks}] {len(chunk)} conditions: {chunk}")

    t0 = time.time()
    tube_set = load_tube_set(shard_dir, conditions=chunk, include_control=True)
    _log(f"[tubes] loaded {len(tube_set.tubes)} tubes "
         f"({sum(t.n_cells for t in tube_set.tubes)} cells) in {time.time()-t0:.0f}s")

    encoder, encoder_sha = load_shared_encoder(out, n_genes=len(tube_set.gene_names))

    t1 = time.time()
    models = train_all_binary(
        tube_set,
        encoder,
        conditions=chunk,
        control_label=C.CONTROL,
        attention_hidden_dim=C.ATTENTION_HIDDEN_DIM,
        epochs=C.STAGE2_EPOCHS,
        lr=C.STAGE2_LR,
        momentum=C.MOMENTUM,
        encoder_frozen=True,
        use_embedding_cache=True,
        device=args.device,
        seed=args.seed,
    )
    train_s = time.time() - t1
    _log(f"[train] {len(models)} binary models in {train_s/60:.1f} min")

    t2 = time.time()
    signatures = derive_signatures(
        models, tube_set, top_n=C.TOP_N, n_steps=C.N_IG_STEPS, device=args.device
    )
    ig_s = time.time() - t2
    _log(f"[ig] {len(signatures)} signatures in {ig_s/60:.1f} min")

    rows = []
    for cond, sig in signatures.items():
        for rank, (gene, ig) in enumerate(zip(sig.genes, sig.ig_scores)):
            rows.append({"cytokine": cond, "gene": gene, "ig": float(ig), "rank_ig": rank})
    df = pd.DataFrame(rows)
    sig_path = out / f"signatures_chunk_{args.chunk_id}.parquet"
    df.to_parquet(sig_path, index=False)

    C.write_json(out / f"chunk_{args.chunk_id}_meta.json", {
        "chunk_id": args.chunk_id,
        "n_chunks": args.n_chunks,
        "conditions": chunk,
        "n_signatures": len(signatures),
        "encoder_sha256": encoder_sha,
        "tubes_shards_sha256": tube_meta["shards_sha256"],
        "top_n": C.TOP_N,
        "n_ig_steps": C.N_IG_STEPS,
        "stage2_epochs": C.STAGE2_EPOCHS,
        "stage2_lr": C.STAGE2_LR,
        "attention_hidden_dim": C.ATTENTION_HIDDEN_DIM,
        "seed": args.seed,
        "device": args.device,
        "train_seconds": round(train_s, 1),
        "ig_seconds": round(ig_s, 1),
    })
    _log(f"[done] {sig_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
