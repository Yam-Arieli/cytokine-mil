"""Stage 1 of the §40 dropout+curation run — the ONE shared Stage-1 encoder.

Trains a single cell-type encoder on stage 0's equal-weight cell set and persists it with
a sha256 digest. Every training chunk recomputes that digest and refuses to run on a
mismatch, so the array can never shard encoder training — the CLAUDE.md §27.6 guard made
structural rather than conventional.

Two things differ from §37's Stage 1 (CLAUDE.md §40):
  * **50% dropout on the input of the encoder's final block** — §40's intervention on the
    gene-space collapse §39.5 measured. It is a Stage-1 regulariser only: the encoder is
    saved and reloaded in `eval()` mode, so the embedding cache, Stage-2 training and IG
    are all deterministic.
  * **A fixed 20-epoch schedule with NO best-validation restore.** The validation split is
    still held out and its curve still recorded — so the overfitting §37 saw (train_acc
    1.0 against a rising val_loss) remains observable — but it does not choose the
    checkpoint. §37 early-stopped and restored the best epoch, which landed at epoch 4, so
    "how long Stage 1 ran" moved together with every other change; here it is pinned.

`encoder.pt` is therefore the **final-epoch** encoder. `encoder_last.pt` is written too and
is identical to it by construction — kept so the artifact layout matches §37's exactly.

All training is `cascadir.train_encoder`; this script chooses no formula of its own.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_dc_config as C  # noqa: E402


def main() -> int:
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=C.STAGE1_EPOCHS)
    ap.add_argument("--dropout", type=float, default=C.ENCODER_DROPOUT)
    args = ap.parse_args()

    import scanpy as sc
    import torch

    from cascadir.train import train_encoder

    out = Path(args.out_dir)
    cells_path = out / "stage1_cells.h5ad"
    if not cells_path.exists():
        raise FileNotFoundError(
            f"{cells_path} missing — run `prepare_oes90_pure.py --out_dir {out}` first "
            "(§40 reuses §37's stage 0 verbatim)."
        )

    adata = sc.read_h5ad(cells_path)
    n_ct = adata.obs[C.CELLTYPE_COL].nunique()
    C.log(
        f"[load] {adata.n_obs} unique cells x {adata.n_vars} genes; {n_ct} cell types; "
        f"{adata.obs[C.CONDITION_COL].nunique()} conditions"
    )
    C.log(
        f"[config] embed_dim={C.EMBED_DIM} hidden_dims={C.HIDDEN_DIMS} "
        f"dropout={args.dropout} epochs={args.epochs} (FIXED, no early stopping) "
        f"lr={C.STAGE1_LR} val_fraction={C.STAGE1_VAL_FRACTION} "
        f"restore_best={C.STAGE1_RESTORE_BEST}"
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
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
        verbose=True,
        val_fraction=C.STAGE1_VAL_FRACTION,
        patience=C.STAGE1_PATIENCE,
        restore_best=C.STAGE1_RESTORE_BEST,
        return_history=True,
    )
    elapsed = time.time() - t0

    last_state = history.attrs.pop("last_state_dict")
    state = encoder.state_dict()
    sha = C.state_dict_sha256(state)

    # With restore_best=False these are the same weights. Assert it rather than assume:
    # a silent divergence here would mean the saved encoder is not the one the recorded
    # history describes.
    if not C.STAGE1_RESTORE_BEST:
        if C.state_dict_sha256(last_state) != sha:
            raise AssertionError(
                "restore_best=False but encoder.pt != final-epoch weights — "
                "train_encoder restored a checkpoint it should not have."
            )

    torch.save(state, out / "encoder.pt")
    torch.save(last_state, out / "encoder_last.pt")
    (out / "encoder_sha256.txt").write_text(sha + "\n")
    history.to_csv(out / "encoder_history.csv", index=False)

    best_epoch = int(history.attrs["best_epoch"])
    best_row = history.loc[history.epoch == best_epoch].iloc[0]
    final_row = history.iloc[-1]
    C.log(
        f"\n[train] {history.attrs['n_epochs_run']} epochs in {elapsed/60:.1f} min "
        f"({history.attrs['n_train_cells']} train / {history.attrs['n_val_cells']} val cells)"
    )
    C.log(
        f"[best]  epoch {best_epoch}: train_loss={best_row.train_loss:.4f} "
        f"train_acc={best_row.train_acc:.4f} val_loss={best_row.val_loss:.4f} "
        f"val_acc={best_row.val_acc:.4f}   (recorded only — NOT the saved checkpoint)"
    )
    C.log(
        f"[saved] epoch {int(final_row.epoch)}: train_loss={final_row.train_loss:.4f} "
        f"train_acc={final_row.train_acc:.4f} val_loss={final_row.val_loss:.4f} "
        f"val_acc={final_row.val_acc:.4f}"
    )
    C.log(
        f"[overfit] final train_acc - val_acc = "
        f"{final_row.train_acc - final_row.val_acc:+.4f}; "
        f"val_loss best {best_row.val_loss:.4f} (ep {best_epoch}) -> "
        f"final {final_row.val_loss:.4f}. §37 saw 0.245 -> 0.607 with no dropout."
    )

    C.write_json(out / "encoder_meta.json", {
        "sha256": sha,
        "embed_dim": C.EMBED_DIM,
        "hidden_dims": list(C.HIDDEN_DIMS),
        "dropout": float(args.dropout),
        "epochs_cap": args.epochs,
        "epochs_run": int(history.attrs["n_epochs_run"]),
        "n_epochs_run": int(history.attrs["n_epochs_run"]),
        "best_epoch": best_epoch,
        "stopped_epoch": history.attrs["stopped_epoch"],
        "restore_best": bool(C.STAGE1_RESTORE_BEST),
        "lr": C.STAGE1_LR,
        "momentum": C.MOMENTUM,
        "val_fraction": C.STAGE1_VAL_FRACTION,
        "patience": C.STAGE1_PATIENCE,
        "n_train_cells": int(history.attrs["n_train_cells"]),
        "n_val_cells": int(history.attrs["n_val_cells"]),
        "n_cell_types": int(n_ct),
        "n_genes": int(adata.n_vars),
        "best_val_loss": float(best_row.val_loss),
        "best_val_acc": float(best_row.val_acc),
        "final_val_loss": float(final_row.val_loss),
        "final_val_acc": float(final_row.val_acc),
        "final_train_acc": float(final_row.train_acc),
        "seed": args.seed,
        "device": args.device,
        "elapsed_s": round(elapsed, 1),
        "note": (
            "encoder.pt is the FINAL-epoch checkpoint (restore_best=False); "
            "best_epoch is recorded for diagnosis only. encoder_last.pt is identical."
        ),
    })
    C.mark_done(out, "encoder")
    C.log(f"\n[done] encoder.pt written  sha256={sha[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
