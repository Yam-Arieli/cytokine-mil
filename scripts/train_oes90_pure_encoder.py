"""Stage 1 of the Oesinghaus 90-cytokine PURE run — the ONE shared Stage-1 encoder.

Trains a single cell-type encoder on the equal-weight cell set from stage 0 and persists
it with a sha256 digest. Every training chunk recomputes that digest and refuses to run on
a mismatch, so the array can never shard encoder training — the CLAUDE.md §27.6 guard made
structural rather than conventional.

Two things differ from the published wide config, both deliberate (CLAUDE.md §37):
  * the encoder is 2x wider (embed 512 -> 1024, hidden (512,512) -> (1024,1024));
  * training early-stops on a held-out, cell-type-stratified validation split, then keeps
    going past the plateau so the saved history shows the overfitting regime. The encoder
    written to `encoder.pt` is the best-validation checkpoint, not the last one.

All training is `cascadir.train_encoder`; this script chooses no formula of its own.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_pure_config as C  # noqa: E402


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=C.STAGE1_EPOCHS)
    args = ap.parse_args()

    import scanpy as sc
    import torch

    from cascadir.train import train_encoder

    out = Path(args.out_dir)
    cells_path = out / "stage1_cells.h5ad"
    if not cells_path.exists():
        raise FileNotFoundError(f"{cells_path} missing — run prepare_oes90_pure.py first.")

    adata = sc.read_h5ad(cells_path)
    n_ct = adata.obs[C.CELLTYPE_COL].nunique()
    C.log(
        f"[load] {adata.n_obs} unique cells x {adata.n_vars} genes; {n_ct} cell types; "
        f"{adata.obs[C.CONDITION_COL].nunique()} conditions"
    )
    C.log(
        f"[config] embed_dim={C.EMBED_DIM} hidden_dims={C.HIDDEN_DIMS} "
        f"epochs<={args.epochs} lr={C.STAGE1_LR} val_fraction={C.STAGE1_VAL_FRACTION} "
        f"patience={C.STAGE1_PATIENCE} extra_after_stop={C.STAGE1_EXTRA_EPOCHS}"
    )

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
        verbose=True,
        val_fraction=C.STAGE1_VAL_FRACTION,
        patience=C.STAGE1_PATIENCE,
        min_delta=C.STAGE1_MIN_DELTA,
        extra_epochs_after_stop=C.STAGE1_EXTRA_EPOCHS,
        return_history=True,
    )
    elapsed = time.time() - t0

    last_state = history.attrs.pop("last_state_dict")
    state = encoder.state_dict()
    sha = C.state_dict_sha256(state)

    torch.save(state, out / "encoder.pt")
    torch.save(last_state, out / "encoder_last.pt")
    (out / "encoder_sha256.txt").write_text(sha + "\n")
    history.to_csv(out / "encoder_history.csv", index=False)

    best_epoch = history.attrs["best_epoch"]
    stopped = history.attrs["stopped_epoch"]
    best_row = history.loc[history.epoch == best_epoch].iloc[0]
    final_row = history.iloc[-1]
    C.log(
        f"\n[train] {history.attrs['n_epochs_run']} epochs in {elapsed/60:.1f} min "
        f"({history.attrs['n_train_cells']} train / {history.attrs['n_val_cells']} val cells)"
    )
    C.log(
        f"[best]  epoch {best_epoch}: train_loss={best_row.train_loss:.4f} "
        f"train_acc={best_row.train_acc:.4f} val_loss={best_row.val_loss:.4f} "
        f"val_acc={best_row.val_acc:.4f}"
    )
    C.log(
        f"[final] epoch {int(final_row.epoch)}: train_loss={final_row.train_loss:.4f} "
        f"train_acc={final_row.train_acc:.4f} val_loss={final_row.val_loss:.4f} "
        f"val_acc={final_row.val_acc:.4f}"
    )
    if stopped is None:
        C.log(
            f"[plateau] NOT reached within {args.epochs} epochs — the encoder is still "
            "improving on validation; the epoch cap, not the plateau, ended training."
        )
    else:
        C.log(
            f"[plateau] epoch {stopped}; ran {history.attrs['n_epochs_run'] - stopped} "
            "epoch(s) past it. Gap at the end: "
            f"train_acc - val_acc = {final_row.train_acc - final_row.val_acc:+.4f} "
            f"(at best: {best_row.train_acc - best_row.val_acc:+.4f})"
        )

    C.write_json(out / "encoder_meta.json", {
        "sha256": sha,
        "embed_dim": C.EMBED_DIM,
        "hidden_dims": list(C.HIDDEN_DIMS),
        "epochs_cap": args.epochs,
        "epochs_run": int(history.attrs["n_epochs_run"]),
        "best_epoch": int(best_epoch),
        "stopped_epoch": None if stopped is None else int(stopped),
        "plateau_reached": stopped is not None,
        "lr": C.STAGE1_LR,
        "momentum": C.MOMENTUM,
        "val_fraction": C.STAGE1_VAL_FRACTION,
        "patience": C.STAGE1_PATIENCE,
        "min_delta": C.STAGE1_MIN_DELTA,
        "extra_epochs_after_stop": C.STAGE1_EXTRA_EPOCHS,
        "n_train_cells": int(history.attrs["n_train_cells"]),
        "n_val_cells": int(history.attrs["n_val_cells"]),
        "n_cell_types": int(n_ct),
        "n_genes": int(adata.n_vars),
        "best_val_loss": float(best_row.val_loss),
        "best_val_acc": float(best_row.val_acc),
        "final_val_loss": float(final_row.val_loss),
        "final_val_acc": float(final_row.val_acc),
        "seed": args.seed,
        "device": args.device,
        "elapsed_s": round(elapsed, 1),
        "note": "encoder.pt is the best-validation checkpoint; encoder_last.pt is the final epoch.",
    })
    C.mark_done(out, "encoder")
    C.log(f"\n[done] encoder.pt written  sha256={sha[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
