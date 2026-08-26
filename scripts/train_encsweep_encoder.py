"""Stage 1 of the encoder-breadth sweep — one Stage-1 encoder per arm.

Trains on the arm's `stage1_cells.h5ad` (already budget-matched by prepare) and persists
the encoder with a sha256 digest, so every downstream chunk can assert it trained on the
arm's own encoder and nothing else (the CLAUDE.md §27.6 guard).

Unlike §37 there is NO early stopping: the published anchor ran a fixed 20 epochs, and
§37's val-loss early stop landed at epoch 4. That is a fourth deviation, and removing it is
part of what this sweep pins back to published values.

All training is `cascadir.train_encoder`; this script chooses no formula of its own.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _encsweep_config as C  # noqa: E402


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    # No `choices=`: these same stages drive both the breadth sweep (CLAUDE.md §38.1)
    # and the Stage-1-construction sweep (§38.4), whose arm names differ. The arm is
    # validated by its directory existing, a few lines below.
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=C.STAGE1_EPOCHS)
    args = ap.parse_args()

    import scanpy as sc
    import torch

    from cascadir.train import train_encoder

    adir = C.arm_dir(args.out_dir, args.arm)
    cells_path = adir / "stage1_cells.h5ad"
    if not cells_path.exists():
        raise FileNotFoundError(f"{cells_path} missing — run prepare_encsweep.py first.")

    adata = sc.read_h5ad(cells_path)
    n_ct = adata.obs[C.CELLTYPE_COL].nunique()
    n_cond = adata.obs[C.CONDITION_COL].nunique()
    C.log(f"[arm:{args.arm}] {adata.n_obs} cells x {adata.n_vars} genes; "
          f"{n_ct} cell types; {n_cond} conditions (incl. {C.CONTROL})")
    C.log(f"[config] embed_dim={C.EMBED_DIM} hidden={C.HIDDEN_DIMS} epochs={args.epochs} "
          f"lr={C.STAGE1_LR} (fixed schedule, no early stopping)")

    t0 = time.time()
    encoder, history = train_encoder(
        adata,
        celltype_col=C.CELLTYPE_COL,
        embed_dim=C.EMBED_DIM,
        hidden_dims=C.HIDDEN_DIMS,
        epochs=args.epochs,
        lr=C.STAGE1_LR,
        momentum=C.MOMENTUM,
        device=args.device,
        seed=args.seed,
        return_history=True,
    )
    elapsed = time.time() - t0

    state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    sha = C.state_dict_sha256(state)
    torch.save(state, adir / "encoder.pt")
    (adir / "encoder_sha256.txt").write_text(sha + "\n")
    history.to_csv(adir / "encoder_history.csv", index=False)

    C.write_json(adir / "encoder_meta.json", {
        "arm": args.arm,
        "sha256": sha,
        "embed_dim": C.EMBED_DIM,
        "hidden_dims": list(C.HIDDEN_DIMS),
        "n_genes": int(adata.n_vars),
        "n_cell_types": int(n_ct),
        "n_encoder_conditions": int(n_cond),
        "n_cells": int(adata.n_obs),
        "epochs": int(args.epochs),
        "lr": C.STAGE1_LR,
        "momentum": C.MOMENTUM,
        "seed": args.seed,
        "early_stopping": False,
        "best_epoch": None,          # fixed schedule: the last epoch is the one used
        "final_train_loss": float(history.train_loss.iloc[-1]),
        "final_train_acc": float(history.train_acc.iloc[-1]),
        "elapsed_s": round(elapsed, 1),
        "device": args.device,
        "note": "encoder.pt is the final epoch; no validation split, matching the "
                "published anchor's fixed 20-epoch schedule.",
    })
    C.log(f"[done] arm={args.arm} sha256={sha[:16]}...  "
          f"train_loss={history.train_loss.iloc[-1]:.4f} "
          f"train_acc={history.train_acc.iloc[-1]:.4f}  ({elapsed:.0f}s)")
    C.mark_done(args.out_dir, f"encoder_{args.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
