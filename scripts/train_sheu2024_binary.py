"""
Train binary AB-MIL models (stimulus vs PBS) for the Sheu 2024 BMDM dataset —
the Bridge step that discovers each stimulus's gene signature S_X^binary for the
single-frame cascade-direction pipeline.

Clone of scripts/train_oesinghaus_binary_missing16.py with:
  * Sheu manifest/HVG paths passed via CLI (one per time point — the manifest
    IS the single-time-frame restriction: it contains time-T stimulated tubes +
    0h-pooled PBS tubes).
  * Targets auto-detected from the manifest (all non-PBS cytokines present at
    this time point), since per-time-point stimulus coverage varies.
  * Sheu val pseudo-donor M2_IL4_rep1 held out.
  * HPs tuned for the 500-gene targeted panel (smaller than the Oesinghaus
    4000-HVG "wide" variant). All binary models share one architecture so the
    IG probe loads them uniformly via state-dict shape inference.

Outputs (one run dir): encoder_shared_stage1.pt, model_<stim>.pt,
dynamics_<stim>.pkl, manifest_*_<stim>.json, label_encoder_<stim>.json.

Usage:
  python scripts/train_sheu2024_binary.py \
    --manifest_path /cs/.../Sheu2024_pseudotubes_3hr/manifest.json \
    --hvg_path      /cs/.../Sheu2024_pseudotubes_3hr/hvg_list.json \
    --output_dir    results/sheu_cascade/3hr/binary
"""

import argparse
import copy
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset  # noqa: E402
from cytokine_mil.experiment_setup import (  # noqa: E402
    build_mil_model,
    build_stage1_manifest,
    filter_manifest,
    make_binary_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.models.instance_encoder import InstanceEncoder  # noqa: E402
from cytokine_mil.training.train_encoder import train_encoder  # noqa: E402
from cytokine_mil.training.train_mil import train_mil  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MANIFEST_PATH_DEFAULT = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json"
HVG_PATH_DEFAULT = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json"
OUTPUT_BASE = REPO_ROOT / "results" / "sheu_cascade" / "binary"
VAL_DONORS = ["M2_IL4_rep1"]          # Sheu val pseudo-donor (§2.5)
CONTROL = "PBS"

# HPs for the 500-gene Sheu targeted panel (narrower than Oesinghaus wide variant).
EMBED_DIM = 128
HIDDEN_DIMS = (256, 128)
ATTENTION_HIDDEN_DIM = 64
STAGE1_EPOCHS = 20
STAGE1_LR = 0.005
STAGE1_MOMENTUM = 0.9
STAGE2_EPOCHS = 200
STAGE2_LR = 0.0001                     # 500-gene panel, higher SNR per gene than Oesinghaus
STAGE2_MOMENTUM = 0.9
LOG_EVERY = 5
SEED = 42


def _train_one_binary_model(target, control, manifest, gene_names, shared_encoder,
                            out_dir, device, seed, log,
                            embed_dim, hidden_dims, attention_hidden_dim):
    log(f"\n{'=' * 50}\nTraining binary model: {target} vs {control}\n{'=' * 50}")
    bin_manifest, label_enc = make_binary_manifest(manifest, target, control=control)
    train_m, val_m = split_manifest_by_donor(bin_manifest, VAL_DONORS)
    n_pos_train = sum(1 for e in train_m if e["cytokine"] == target)
    if n_pos_train == 0:
        log(f"  SKIP {target}: no positive tubes in train split at this time point")
        return None
    log(f"  {target}: {len(train_m)} train tubes ({n_pos_train} positive), {len(val_m)} val tubes")

    safe = target.replace("/", "_")
    train_m_path = out_dir / f"manifest_train_{safe}.json"
    val_m_path = out_dir / f"manifest_val_{safe}.json"
    with open(train_m_path, "w") as fh:
        json.dump(train_m, fh)
    with open(val_m_path, "w") as fh:
        json.dump(val_m, fh)
    with open(out_dir / f"label_encoder_{safe}.json", "w") as fh:
        json.dump({"positive": label_enc.positive, "negative": label_enc.negative}, fh)

    train_dataset = PseudoTubeDataset(str(train_m_path), label_enc, gene_names=gene_names, preload=True)
    val_dataset = (PseudoTubeDataset(str(val_m_path), label_enc, gene_names=gene_names, preload=True)
                   if len(val_m) > 0 else None)

    encoder = copy.deepcopy(shared_encoder)
    model = build_mil_model(
        encoder, embed_dim=embed_dim, attention_hidden_dim=attention_hidden_dim,
        n_classes=label_enc.n_classes(), encoder_frozen=True,
    )
    dynamics = train_mil(
        model, train_dataset, n_epochs=STAGE2_EPOCHS, lr=STAGE2_LR,
        momentum=STAGE2_MOMENTUM, log_every_n_epochs=LOG_EVERY, device=device,
        seed=seed, verbose=True, val_dataset=val_dataset,
    )
    torch.save(model.state_dict(), out_dir / f"model_{safe}.pt")
    payload = {
        "records": dynamics["records"],
        "val_records": dynamics.get("val_records", []),
        "logged_epochs": dynamics["logged_epochs"],
        "condition": target, "control": control,
    }
    with open(out_dir / f"dynamics_{safe}.pkl", "wb") as fh:
        pickle.dump(payload, fh)
    log(f"  Saved: model_{safe}.pt, dynamics_{safe}.pkl")
    return payload


def main():
    p = argparse.ArgumentParser(description="Binary AB-MIL (stimulus vs PBS) for Sheu.")
    p.add_argument("--manifest_path", default=MANIFEST_PATH_DEFAULT)
    p.add_argument("--hvg_path", default=HVG_PATH_DEFAULT)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--targets", nargs="*", default=None,
                   help="Stimuli to train (default: all non-PBS cytokines in the manifest).")
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=list(HIDDEN_DIMS))
    p.add_argument("--attention_hidden_dim", type=int, default=ATTENTION_HIDDEN_DIM)
    args = p.parse_args()

    seed = args.seed
    embed_dim = args.embed_dim
    hidden_dims = tuple(args.hidden_dims)
    attention_hidden_dim = args.attention_hidden_dim

    ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_BASE / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    for required in ("numpy", "torch", "scanpy", "anndata"):
        try:
            __import__(required)
        except ImportError as e:
            print(f"FATAL: missing dependency '{required}': {e}", flush=True)
            sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 60)
    log("Sheu binary AB-MIL training (Bridge: stimulus vs PBS)")
    log("=" * 60)
    log(f"manifest_path: {args.manifest_path}")
    log(f"hvg_path:      {args.hvg_path}")
    log(f"output_dir:    {out_dir}")
    log(f"val_donors:    {VAL_DONORS}")
    log(f"seed:          {seed}   device: {device}")
    log(f"HPs: embed_dim={embed_dim} hidden_dims={hidden_dims} att={attention_hidden_dim} "
        f"stage1={STAGE1_EPOCHS}@{STAGE1_LR} stage2={STAGE2_EPOCHS}@{STAGE2_LR}")

    with open(args.manifest_path) as fh:
        manifest = json.load(fh)
    with open(args.hvg_path) as fh:
        gene_names = json.load(fh)
    log(f"Manifest entries: {len(manifest)}   panel genes: {len(gene_names)}")

    manifest_cytokines = sorted({e["cytokine"] for e in manifest if e["cytokine"] != CONTROL})
    targets = args.targets if args.targets else manifest_cytokines
    targets = [t for t in targets if t in manifest_cytokines]
    if not targets:
        log("FATAL: no trainable targets present in manifest.")
        sys.exit(2)
    log(f"Targets ({len(targets)}): {targets}")

    filtered_manifest = filter_manifest(manifest, targets, include_pbs=True)
    log(f"Filtered manifest: {len(filtered_manifest)} entries")

    # ---- Shared Stage 1 encoder (cell-type classification) ----
    log("\n" + "=" * 60 + "\nSHARED STAGE 1 — encoder pre-training\n" + "=" * 60)
    stage1_path = out_dir / "manifest_stage1_shared.json"
    stage1_manifest = build_stage1_manifest(filtered_manifest, save_path=str(stage1_path))
    log(f"Stage 1 manifest: {len(stage1_manifest)} entries")
    cell_dataset = CellDataset(str(stage1_path), gene_names=gene_names, preload=True)
    cell_loader = DataLoader(cell_dataset, batch_size=256, shuffle=True, num_workers=0)
    log(f"Stage 1 cells: {len(cell_dataset)}  cell types: {cell_dataset.n_cell_types()}")
    encoder = InstanceEncoder(
        input_dim=len(gene_names), embed_dim=embed_dim,
        n_cell_types=cell_dataset.n_cell_types(), hidden_dims=hidden_dims,
    )
    train_encoder(encoder, cell_loader, n_epochs=STAGE1_EPOCHS, lr=STAGE1_LR,
                  momentum=STAGE1_MOMENTUM, device=device)
    torch.save(encoder.state_dict(), out_dir / "encoder_shared_stage1.pt")
    log("Shared encoder saved.")

    # ---- One binary model per stimulus ----
    n_done = 0
    for i, target in enumerate(targets, start=1):
        log(f"\n[{i}/{len(targets)}] {target} START")
        try:
            res = _train_one_binary_model(
                target=target, control=CONTROL, manifest=filtered_manifest,
                gene_names=gene_names, shared_encoder=encoder, out_dir=out_dir,
                device=device, seed=seed, log=log, embed_dim=embed_dim,
                hidden_dims=hidden_dims, attention_hidden_dim=attention_hidden_dim,
            )
            if res is not None:
                n_done += 1
            log(f"[{i}/{len(targets)}] {target} DONE")
        except Exception as e:
            import traceback
            log(f"[{i}/{len(targets)}] {target} ERROR: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            continue

    log(f"\nDone. {n_done}/{len(targets)} stimuli trained. Results in {out_dir}")
    log(f"Finished: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
