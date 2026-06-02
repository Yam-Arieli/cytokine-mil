"""Step 4 — train the Stage-1 cell encoder (cell-type supervision).

The encoder maps each cell to an embedding; it is shared (frozen) by the
multiclass coupling model and the per-condition binary models downstream.

Run:  python 04_train_encoder.py
"""

from __future__ import annotations

import torch

from _common import banner, preprocessed

import cascadir as cd


def main() -> None:
    banner("Step 4 — train_encoder (Stage 1)")
    proc = preprocessed()

    encoder = cd.train_encoder(
        proc,
        celltype_col="cell_type",
        embed_dim=128,
        epochs=5,        # a real run uses ~50
        device="cpu",
        seed=42,
        verbose=False,
    )
    print(f"trained {type(encoder).__name__}: input_dim={encoder.input_dim} "
          f"embed_dim={encoder.embed_dim}")

    # quick check: embed one cell batch
    X = torch.from_numpy(proc.X[:16].astype("float32"))
    with torch.no_grad():
        H = encoder(X)
    print(f"embedded a batch: X{tuple(X.shape)} -> H{tuple(H.shape)}")


if __name__ == "__main__":
    main()
