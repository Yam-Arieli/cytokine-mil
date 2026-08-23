#!/usr/bin/env python
"""Stage 1 of the Oesinghaus full-90 DAG — the ONE shared Stage-1 encoder.

CLAUDE.md §27.6: the previous attempt to widen the panel (24 -> 45 cytokines) failed to
reproduce the published direction accuracy, and the suspected cause was that the training
array sharded Stage-1 encoder training, giving each chunk its own encoder. Signatures from
different encoders are not comparable, and `cross_asym` compares signatures ACROSS
cytokines, so that silently corrupts every downstream number.

This stage trains the encoder exactly once and writes its sha256 digest. Every chunk task
in Stage 2 recomputes the digest over the encoder it loaded and refuses to run on a
mismatch — the guard is an assertion, not a convention.

Hyperparameters are the published "wide" Oesinghaus config (scripts/_full90_config.py),
not cascadir's packaged TrainConfig defaults.

Usage (cluster, GPU):
  python scripts/train_oesinghaus_full90_encoder.py --output_dir results/oes_full90
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default="results/oes_full90")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    import anndata as ad
    import torch

    from cascadir.train import train_encoder

    out = Path(args.output_dir)
    stage1_path = out / "stage1_cells.h5ad"
    if not stage1_path.exists():
        raise SystemExit(f"FATAL: {stage1_path} missing — run prepare_oesinghaus_full90.py first")

    adata = ad.read_h5ad(stage1_path)
    print(f"[load] {adata.n_obs} unique cells x {adata.n_vars} genes; "
          f"{adata.obs[C.CELLTYPE_COL].nunique()} cell types", flush=True)
    if adata.obs_names.duplicated().any():
        raise SystemExit("FATAL: stage1_cells.h5ad has duplicate barcodes — encoder input "
                         "must be unique cells (cascadir train.py:96-99)")

    t0 = time.time()
    encoder = train_encoder(
        adata,
        celltype_col=C.CELLTYPE_COL,
        embed_dim=C.EMBED_DIM,
        hidden_dims=C.HIDDEN_DIMS,
        epochs=C.STAGE1_EPOCHS,
        lr=C.STAGE1_LR,
        momentum=C.MOMENTUM,
        device=args.device,
        seed=args.seed,
        verbose=True,
    )
    elapsed = time.time() - t0

    state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    torch.save(state, out / "encoder.pt")
    digest = C.state_dict_sha256(state)
    (out / "encoder_sha256.txt").write_text(digest + "\n")

    C.write_json(out / "encoder_meta.json", {
        "sha256": digest,
        "embed_dim": C.EMBED_DIM,
        "hidden_dims": list(C.HIDDEN_DIMS),
        "epochs": C.STAGE1_EPOCHS,
        "lr": C.STAGE1_LR,
        "momentum": C.MOMENTUM,
        "seed": args.seed,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_cell_types": int(adata.obs[C.CELLTYPE_COL].nunique()),
        "device": args.device,
        "elapsed_s": round(elapsed, 1),
    })
    print(f"[done] encoder.pt written in {elapsed:.0f}s  sha256={digest[:16]}...", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
