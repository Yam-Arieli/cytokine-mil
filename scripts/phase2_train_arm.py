"""Phase 1+2 stage B — one arm: (encoder path) x (Stage-2 path), all panel cytokines.

Loads the Stage-1 encoder produced by `--encoder_path` (digest-verified), trains one binary
AB-MIL per panel cytokine with `--stage2_path`'s trainer, then derives every signature with
a SINGLE attribution implementation over a SINGLE fixed tube set.

Holding attribution fixed is what Phase 0 licenses: the two IG implementations returned
identical top-50s on identical weights (Jaccard 1.000), and tube/baseline selection moved
meanJ by only +0.016. Fixing it removes the last non-weight variable, so the four arms
differ **only** in which code produced the weights.

Both Stage-2 trainers are given the same tubes. Phase 0a proved the manifest `.h5ad` tubes
and the §36 shards are bit-identical (9100/9100), so `cytokine_mil` may read the manifest
and `cascadir` the shards without that being a difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _phase2_config as C  # noqa: E402


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def load_encoder(path: str, seed: int, device: str):
    """Rebuild an arm's Stage-1 encoder and ASSERT its digest is the one stage A wrote.

    Signatures derived under different encoders are not comparable, and the whole point of
    the transplant is that a given arm uses a SPECIFIC encoder — a silent mix-up would
    invert the result without erroring.
    """
    from cascadir.models import InstanceEncoder

    d = C.encoder_path_dir(path, seed)
    meta = C.read_json(d / "encoder_meta.json")
    expected = (d / "encoder_sha256.txt").read_text().strip()
    state = torch.load(d / "encoder.pt", map_location="cpu")
    actual = C.state_dict_sha256(state)
    if actual != expected or meta["sha256"] != expected:
        raise AssertionError(
            f"ENCODER MISMATCH for path={path} seed={seed}: encoder.pt hashes to "
            f"{actual[:16]}... but stage A recorded {expected[:16]}...  Refusing to run."
        )
    enc = InstanceEncoder(
        input_dim=int(meta["n_genes"]), embed_dim=int(meta["embed_dim"]),
        n_cell_types=int(meta["n_cell_types"]), hidden_dims=tuple(meta["hidden_dims"]),
    )
    enc.load_state_dict(state)
    return enc.to(device).eval(), meta, expected


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def train_head_cd(tube_set, condition, encoder, seed, device):
    from cascadir.train import train_binary_mil

    return train_binary_mil(
        tube_set, condition, encoder, control_label=C.CONTROL,
        attention_hidden_dim=C.ATTENTION_HIDDEN_DIM, epochs=C.STAGE2_EPOCHS,
        lr=C.STAGE2_LR, momentum=C.STAGE2_MOMENTUM, encoder_frozen=True,
        device=device, seed=seed, verbose=False,
    )


def train_head_cm(manifest, condition, encoder, gene_names, seed, device, workdir):
    """cytokine_mil Stage 2 via its own manifest/dataset/trainer stack."""
    import copy

    from cytokine_mil.data.dataset import PseudoTubeDataset
    from cytokine_mil.experiment_setup import (
        build_mil_model,
        make_binary_manifest,
        split_manifest_by_donor,
    )
    from cytokine_mil.training.train_mil import train_mil

    bin_manifest, label_enc = make_binary_manifest(manifest, condition, control=C.CONTROL)
    train_m, _val_m = split_manifest_by_donor(bin_manifest, C.VAL_DONORS)
    safe = str(condition).replace("/", "_")
    mp = Path(workdir) / f"manifest_train_{safe}.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(train_m))

    ds = PseudoTubeDataset(str(mp), label_enc, gene_names=gene_names, preload=True)
    model = build_mil_model(
        copy.deepcopy(encoder), embed_dim=C.EMBED_DIM,
        attention_hidden_dim=C.ATTENTION_HIDDEN_DIM,
        n_classes=label_enc.n_classes(), encoder_frozen=True,
    )
    train_mil(
        model, ds, n_epochs=C.STAGE2_EPOCHS, lr=C.STAGE2_LR, momentum=C.STAGE2_MOMENTUM,
        # Dynamics logging is pure read-only overhead here (no grad, no RNG, no weight
        # change) and this run needs only the final model, so it is logged once.
        # `tests/test_phase2_arm.py` pins that log frequency does not move the weights.
        log_every_n_epochs=C.STAGE2_EPOCHS,
        device=torch.device(device), seed=seed, verbose=False,
    )
    mp.unlink(missing_ok=True)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder_path", required=True, choices=C.PATHS)
    ap.add_argument("--stage2_path", required=True, choices=C.PATHS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--shard_dir", default=str(C.SHARD_DIR))
    ap.add_argument("--manifest", default=C.MANIFEST_PATH)
    ap.add_argument("--hvg_path", default=C.HVG_PATH)
    ap.add_argument("--panel", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from cascadir.signatures import derive_signature
    from cytokine_mil.analysis.full90_tube_io import load_tube_set
    from cascadir.types import PseudoTubeSet

    C.OUT_DIR = Path(args.out_dir)
    panel = args.panel or C.PANEL
    d = C.arm_dir(args.encoder_path, args.stage2_path, args.seed)
    d.mkdir(parents=True, exist_ok=True)
    name = C.arm_name(args.encoder_path, args.stage2_path)
    C.log(f"[arm {name} seed={args.seed}] {len(panel)} cytokines, device={args.device}")

    enc, meta, enc_sha = load_encoder(args.encoder_path, args.seed, args.device)
    C.log(f"[arm {name}] encoder verified sha={enc_sha[:16]}...")

    with open(args.hvg_path) as fh:
        gene_names = [str(g) for g in json.load(fh)]
    with open(args.manifest) as fh:
        manifest = json.load(fh)

    # The control tubes are shared by every cytokine's tube set and by the IG baseline,
    # so they are loaded once.
    ctrl_set = load_tube_set(args.shard_dir, conditions=[C.CONTROL], include_control=True)
    ctrl_tubes = [t for t in ctrl_set.tubes if t.condition == C.CONTROL]
    C.log(f"[arm {name}] control tubes: {len(ctrl_tubes)} "
          f"({len({t.donor for t in ctrl_tubes})} donors)")

    rows: list[dict] = []
    for i, cyt in enumerate(panel, 1):
        cond_set = load_tube_set(args.shard_dir, conditions=[cyt], include_control=False)
        cond_tubes = [t for t in cond_set.tubes if t.condition == cyt]
        if not cond_tubes:
            C.log(f"  [{i}/{len(panel)}] {cyt}: SKIP (no tubes)")
            continue
        ts = PseudoTubeSet(tubes=list(cond_tubes) + list(ctrl_tubes),
                           gene_names=tuple(gene_names), control_label=C.CONTROL)

        if args.stage2_path == "cd":
            model = train_head_cd(ts, cyt, enc, args.seed, args.device)
        else:
            model = train_head_cm(manifest, cyt, enc, gene_names, args.seed,
                                  args.device, d / "_tmp")

        sig = derive_signature(model, ts, cyt, top_n=C.DEEP_N, n_steps=C.IG_STEPS,
                               device=args.device)
        rows += [{"cytokine": cyt, "gene": g, "ig": float(s), "rank_ig": r}
                 for r, (g, s) in enumerate(zip(sig.genes, sig.ig_scores))]
        C.log(f"  [{i}/{len(panel)}] {cyt}: {len(cond_tubes)} tubes -> "
              f"{len(sig.genes)} genes")
        del model, cond_set, cond_tubes, ts

    if not rows:
        raise SystemExit(f"[arm {name}] produced no signatures — refusing to write.")
    df = pd.DataFrame(rows)
    df.to_parquet(d / "signatures.parquet", index=False)
    C.write_json(d / "arm_meta.json", {
        "arm": name, "encoder_path": args.encoder_path, "stage2_path": args.stage2_path,
        "seed": args.seed, "encoder_sha256": enc_sha,
        "n_cytokines": int(df.cytokine.nunique()), "top_n_derived": C.DEEP_N,
        "n_control_tubes": len(ctrl_tubes),
        "stage2_epochs": C.STAGE2_EPOCHS, "stage2_lr": C.STAGE2_LR,
    })
    C.mark_done(d, "arm")
    C.log(f"[arm {name}] wrote {d/'signatures.parquet'} "
          f"({df.cytokine.nunique()} cytokines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
