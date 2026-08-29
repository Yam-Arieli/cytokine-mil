"""Phase 1+2 stage A — one Stage-1 encoder, trained by ONE of the two code paths.

Both paths are handed the **same cells**: `stage1_cells.h5ad`, read once, with the same
cell-type labels in the same order. `cascadir.train_encoder` takes the AnnData directly;
`cytokine_mil.train_encoder` takes a DataLoader, so the cells are wrapped in a plain
TensorDataset rather than routed through `CellDataset` — that keeps the Stage-1 INPUT
provably identical and leaves the encoder construction and training loop as the only
difference, which is the whole point of the arm.

Each path seeds itself the way it normally does. Forcing a shared initialisation would
hide a real difference if the paths' RNG conventions are part of what separates them; the
three-seed design is what separates a path effect from seed noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _phase2_config as C  # noqa: E402


def _load_stage1(path: Path):
    import anndata

    ad = anndata.read_h5ad(path)
    X = ad.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    ct = ad.obs[C.CELLTYPE_COL].astype(str).to_numpy()
    return ad, X, ct


def train_cm(X, ct, seed: int, device: str):
    """cytokine_mil Stage 1: build InstanceEncoder, then train_encoder over a DataLoader."""
    from torch.utils.data import DataLoader, TensorDataset

    from cytokine_mil.models.instance_encoder import InstanceEncoder
    from cytokine_mil.training.train_encoder import train_encoder

    classes = sorted(set(ct.tolist()))
    y = np.array([classes.index(c) for c in ct], dtype=np.int64)
    torch.manual_seed(seed)
    enc = InstanceEncoder(
        input_dim=X.shape[1], embed_dim=C.EMBED_DIM,
        n_cell_types=len(classes), hidden_dims=C.HIDDEN_DIMS,
    )
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=C.STAGE1_BATCH, shuffle=True, num_workers=0,
                        generator=g)
    enc = train_encoder(enc, loader, n_epochs=C.STAGE1_EPOCHS, lr=C.STAGE1_LR,
                        momentum=C.STAGE1_MOMENTUM, device=torch.device(device),
                        verbose=True)
    return enc, classes


def train_cd(ad, seed: int, device: str):
    """cascadir Stage 1: train_encoder builds the encoder itself from the AnnData."""
    from cascadir.train import train_encoder as cd_train_encoder

    enc = cd_train_encoder(
        ad, celltype_col=C.CELLTYPE_COL, embed_dim=C.EMBED_DIM,
        hidden_dims=C.HIDDEN_DIMS, epochs=C.STAGE1_EPOCHS, lr=C.STAGE1_LR,
        momentum=C.STAGE1_MOMENTUM, batch_size=C.STAGE1_BATCH, device=device,
        seed=seed, verbose=True,
    )
    classes = sorted(set(ad.obs[C.CELLTYPE_COL].astype(str)))
    return enc, classes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, choices=C.PATHS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage1_cells", default=str(C.STAGE1_CELLS))
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    C.OUT_DIR = Path(args.out_dir)
    d = C.encoder_path_dir(args.path, args.seed)
    d.mkdir(parents=True, exist_ok=True)

    C.log(f"[stage1] path={args.path} seed={args.seed} device={args.device}")
    ad, X, ct = _load_stage1(Path(args.stage1_cells))
    C.log(f"[stage1] cells={X.shape[0]} genes={X.shape[1]} "
          f"cell_types={len(set(ct.tolist()))}")

    if args.path == "cm":
        enc, classes = train_cm(X, ct, args.seed, args.device)
    else:
        enc, classes = train_cd(ad, args.seed, args.device)

    state = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
    torch.save(state, d / "encoder.pt")
    sha = C.state_dict_sha256(state)
    (d / "encoder_sha256.txt").write_text(sha)
    C.write_json(d / "encoder_meta.json", {
        "path": args.path, "seed": args.seed, "sha256": sha,
        "n_genes": int(X.shape[1]), "embed_dim": C.EMBED_DIM,
        "hidden_dims": list(C.HIDDEN_DIMS), "n_cell_types": len(classes),
        "cell_types": classes, "n_cells": int(X.shape[0]),
        "stage1_epochs": C.STAGE1_EPOCHS, "stage1_lr": C.STAGE1_LR,
    })
    C.mark_done(d, "encoder")
    C.log(f"[stage1] saved {d/'encoder.pt'}  sha={sha[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
