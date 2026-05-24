"""
Sheu 2024 phase 1 — Stage 1+2 training with an auxiliary signaling-adapter head.

Tests root cause #1 of the failed §21 axis-discovery gate: the default Stage 1
objective (cell-type classification) doesn't reward the encoder for preserving
receptor-architecture information, so the embedding collapses to "any-stim vs
PBS" and projections of any cytokine pair onto any direction give a positive
bias regardless of receptor biology.

Fix: during Stage 1, attach a second classification head that predicts the
**signaling adapter** of each cell's tube (MyD88_only / TRIF_only / MyD88_TRIF /
TNFR / IFNAR / unstim). Cytokines that share an adapter (e.g. P3CSK + CpG —
both MyD88-only; LPS + LPSlo — both MyD88_TRIF) get pulled together. The
auxiliary loss is added to the cell-type loss with weight `aux_weight`.

Stage 2 unchanged.

Usage:
    python scripts/train_sheu2024_stage12_aux.py --seed 42 --output_dir results/sheu2024_aux/seed_42
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import (
    build_encoder,
    build_mil_model,
    build_stage1_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.training.train_mil import train_mil


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json"
HVG_PATH      = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json"
OUTPUT_BASE   = Path(__file__).parent.parent / "results" / "sheu2024_aux"

VAL_PSEUDO_DONORS = ["M2_IL4_rep1"]

# Narrowed hyperparameters (matches train_sheu2024_stage12.py post-2026-05-24).
EMBED_DIM            = 32
ATTENTION_HIDDEN_DIM = 16
STAGE1_EPOCHS        = 30
STAGE1_LR            = 0.003
STAGE1_MOMENTUM      = 0.9
STAGE1_BATCH_SIZE    = 256
AUX_WEIGHT           = 1.0     # weight on adapter loss vs cell-type loss
STAGE2_EPOCHS        = 40
STAGE2_LR            = 0.0005
STAGE2_MOMENTUM      = 0.9
LR_WARMUP            = 5

# Adapter map for Sheu phase 1 — groups cytokines by their signaling adapter
# architecture. P3CSK + CpG share MyD88-only (no TRIF). LPS + LPSlo share
# MyD88+TRIF. polyIC is TRIF-only. TNF and IFNb are cytokine receptors.
SHEU_ADAPTER_MAP = {
    "PBS":   "unstim",
    "P3CSK": "MyD88_only",   # TLR2
    "CpG":   "MyD88_only",   # TLR9 (note: macrophages, so signal is MyD88)
    "LPS":   "MyD88_TRIF",   # TLR4
    "LPSlo": "MyD88_TRIF",   # TLR4 low-dose
    "PIC":   "TRIF_only",    # TLR3
    "TNF":   "TNFR",
    "IFNb":  "IFNAR",
}
ADAPTER_CLASSES = ["unstim", "MyD88_only", "TRIF_only", "MyD88_TRIF", "TNFR", "IFNAR"]
ADAPTER_TO_IDX = {a: i for i, a in enumerate(ADAPTER_CLASSES)}


# ---------------------------------------------------------------------------
# Slim label encoder (matches train_sheu2024_stage12.py)
# ---------------------------------------------------------------------------

class _SlimSheuLabel:
    def __init__(self, cytokines):
        non_pbs = sorted(c for c in cytokines if c != "PBS")
        names = ["PBS"] + non_pbs
        self._label_to_idx = {c: i for i, c in enumerate(names)}
        self._idx_to_label = {i: c for i, c in enumerate(names)}
    @classmethod
    def fit(cls, manifest):
        return cls({e["cytokine"] for e in manifest})
    def encode(self, c): return self._label_to_idx[c]
    def decode(self, i): return self._idx_to_label[i]
    def n_classes(self): return len(self._label_to_idx)
    @property
    def cytokines(self):
        return [self._idx_to_label[i] for i in sorted(self._idx_to_label)]


# ---------------------------------------------------------------------------
# Cell-level dataset with adapter labels
# ---------------------------------------------------------------------------

def _build_cell_arrays(stage1_manifest, gene_names, log):
    """Read each Stage 1 tube h5ad once; produce (X, ct_idx, adapter_idx) per cell.

    Returns:
        X: (N_total, G) float32
        ct_idx: (N_total,) int64
        adapter_idx: (N_total,) int64
        cell_type_to_idx: dict
    """
    # Build the cell-type map deterministically over the manifest entries.
    all_cts = set()
    for entry in stage1_manifest:
        all_cts.update(entry.get("cell_types_included", []))
    ct_to_idx = {ct: i for i, ct in enumerate(sorted(all_cts))}

    all_X = []
    all_ct = []
    all_adapter = []
    skipped = 0
    for entry in stage1_manifest:
        adata = sc.read_h5ad(entry["path"])
        adata = adata[:, gene_names]
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        cell_types = adata.obs.get("cell_type", None)
        if cell_types is None:
            skipped += 1
            continue
        cytokine = entry["cytokine"]
        adapter = SHEU_ADAPTER_MAP.get(cytokine)
        if adapter is None:
            log(f"  WARNING: no adapter mapping for cytokine={cytokine!r}, skipping")
            skipped += 1
            continue
        adapter_i = ADAPTER_TO_IDX[adapter]
        ct_i = np.array([ct_to_idx.get(str(ct), 0) for ct in cell_types.values],
                        dtype=np.int64)
        ad_i = np.full(X.shape[0], adapter_i, dtype=np.int64)
        all_X.append(X)
        all_ct.append(ct_i)
        all_adapter.append(ad_i)
    if skipped:
        log(f"  Skipped {skipped} stage-1 tubes (missing cell_type or unknown cytokine).")
    X = np.concatenate(all_X, axis=0)
    ct_idx = np.concatenate(all_ct, axis=0)
    adapter_idx = np.concatenate(all_adapter, axis=0)
    return X, ct_idx, adapter_idx, ct_to_idx


# ---------------------------------------------------------------------------
# Stage 1: train encoder with TWO heads
# ---------------------------------------------------------------------------

def train_stage1_with_aux(
    encoder: nn.Module,
    X: np.ndarray,
    ct_idx: np.ndarray,
    adapter_idx: np.ndarray,
    n_cell_types: int,
    n_adapter_classes: int,
    *,
    n_epochs: int,
    lr: float,
    momentum: float,
    batch_size: int,
    aux_weight: float,
    device: torch.device,
    log,
):
    """Train encoder with cell-type + adapter heads. Modifies encoder in-place.

    Both heads are MLPs from embed_dim → n_classes. cell_type_head is the
    encoder's existing attribute (kept; same as the default Stage 1 path).
    adapter_head is added here, returned for record-keeping.
    """
    embed_dim = encoder.embed_dim
    adapter_head = nn.Linear(embed_dim, n_adapter_classes).to(device)
    nn.init.kaiming_uniform_(adapter_head.weight, nonlinearity="relu")
    nn.init.zeros_(adapter_head.bias)

    encoder.to(device).train()
    adapter_head.train()
    params = list(encoder.parameters()) + list(adapter_head.parameters())
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    ce = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X)
    ct_t = torch.from_numpy(ct_idx)
    ad_t = torch.from_numpy(adapter_idx)
    ds = TensorDataset(X_t, ct_t, ad_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(1, n_epochs + 1):
        running_ct = 0.0
        running_ad = 0.0
        n_batches = 0
        for X_b, ct_b, ad_b in loader:
            X_b = X_b.to(device, non_blocking=True)
            ct_b = ct_b.to(device, non_blocking=True)
            ad_b = ad_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            h = encoder(X_b)
            ct_logits = encoder.cell_type_head(h)
            ad_logits = adapter_head(h)
            loss_ct = ce(ct_logits, ct_b)
            loss_ad = ce(ad_logits, ad_b)
            loss = loss_ct + aux_weight * loss_ad
            loss.backward()
            optimizer.step()
            running_ct += loss_ct.item()
            running_ad += loss_ad.item()
            n_batches += 1
        log(f"[Stage 1/{n_epochs}] epoch {epoch:>3d}  "
            f"L_ct={running_ct/n_batches:.4f}  "
            f"L_ad={running_ad/n_batches:.4f}")
    encoder.eval()
    adapter_head.eval()
    return adapter_head


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--manifest_path", type=str, default=MANIFEST_PATH)
    p.add_argument("--hvg_path", type=str, default=HVG_PATH)
    p.add_argument("--stage1_epochs", type=int, default=STAGE1_EPOCHS)
    p.add_argument("--stage2_epochs", type=int, default=STAGE2_EPOCHS)
    p.add_argument("--lr", type=float, default=STAGE2_LR)
    p.add_argument("--stage1_lr", type=float, default=STAGE1_LR)
    p.add_argument("--lr_warmup_epochs", type=int, default=LR_WARMUP)
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    p.add_argument("--attention_hidden_dim", type=int, default=ATTENTION_HIDDEN_DIM)
    p.add_argument("--aux_weight", type=float, default=AUX_WEIGHT)
    p.add_argument("--val_donors", nargs="*", default=VAL_PSEUDO_DONORS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _plot_loss_curve(loss_components, out_dir, log):
    if not loss_components.get("total"):
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(loss_components["total"]) + 1)
    ax.plot(list(epochs), loss_components["total"], label="total", color="steelblue")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Stage 2 Loss — Sheu adapter-aux variant")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    log("  Saved: loss_curve.png")


def main():
    args = _parse_args()
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_BASE / f"run_{ts}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train.log", "w")
    def log(msg=""):
        print(msg)
        print(msg, file=log_file, flush=True)

    log(f"Output : {out_dir}")
    log(f"Seed   : {args.seed}")
    log(f"Variant: ADAPTER-AUX-HEAD (Sheu phase 1)")
    log(f"Adapter classes: {ADAPTER_CLASSES}")
    log(f"aux_weight: {args.aux_weight}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    with open(args.manifest_path) as fh:
        manifest = json.load(fh)
    with open(args.hvg_path) as fh:
        gene_names = json.load(fh)
    log(f"\nManifest: {len(manifest)} tubes, {len(gene_names)} genes")

    label_enc = _SlimSheuLabel.fit(manifest)
    log(f"  Slim label encoder: {label_enc.n_classes()} classes ({label_enc.cytokines})")
    with open(out_dir / "label_encoder.json", "w") as fh:
        json.dump({"cytokines": list(label_enc.cytokines)}, fh)

    log("\nBuilding Stage 1 manifest...")
    stage1_manifest_path = out_dir / "manifest_stage1.json"
    stage1_manifest = build_stage1_manifest(
        manifest, save_path=str(stage1_manifest_path), donor_offset=0,
    )
    log(f"  {len(stage1_manifest)} stage-1 tubes")

    log("\nSplitting train/val by pseudo-donor...")
    train_manifest, val_manifest = split_manifest_by_donor(manifest, args.val_donors)
    train_m_path = out_dir / "manifest_train.json"
    val_m_path = out_dir / "manifest_val.json"
    with open(train_m_path, "w") as fh: json.dump(train_manifest, fh)
    with open(val_m_path, "w") as fh: json.dump(val_manifest, fh)
    log(f"  Train: {len(train_manifest)} | Val: {len(val_manifest)}")
    log(f"  Val pseudo-donors: {args.val_donors}")

    # ------------------------------------------------------------------
    # Stage 1: cell-type + adapter dual-head pretraining
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("STAGE 1: dual-head encoder pre-training (cell-type + adapter)")
    log("=" * 60)
    log("Building cell-level arrays with (cell_type, adapter) labels...")
    X, ct_idx, adapter_idx, ct_to_idx = _build_cell_arrays(
        stage1_manifest, gene_names, log,
    )
    n_cell_types = len(ct_to_idx)
    n_adapter_classes = len(ADAPTER_CLASSES)
    log(f"  Cells: {X.shape[0]} | cell_types: {n_cell_types} | adapter_classes: {n_adapter_classes}")
    log(f"  Adapter label distribution:")
    for i, a in enumerate(ADAPTER_CLASSES):
        c = int((adapter_idx == i).sum())
        log(f"    {a:<14}: {c}")

    encoder = build_encoder(
        n_input_genes=len(gene_names),
        n_cell_types=n_cell_types,
        embed_dim=args.embed_dim,
    )
    log(f"  Encoder: input={len(gene_names)}, embed_dim={args.embed_dim}, "
        f"n_cell_types={n_cell_types}")
    log(f"  Stage 1: {args.stage1_epochs} epochs, lr={args.stage1_lr}, "
        f"aux_weight={args.aux_weight}, batch_size={STAGE1_BATCH_SIZE}")

    adapter_head = train_stage1_with_aux(
        encoder, X, ct_idx, adapter_idx,
        n_cell_types=n_cell_types,
        n_adapter_classes=n_adapter_classes,
        n_epochs=args.stage1_epochs,
        lr=args.stage1_lr,
        momentum=STAGE1_MOMENTUM,
        batch_size=STAGE1_BATCH_SIZE,
        aux_weight=args.aux_weight,
        device=device,
        log=log,
    )
    torch.save(encoder.state_dict(), out_dir / "encoder_stage1.pt")
    torch.save(adapter_head.state_dict(), out_dir / "adapter_head.pt")
    log("  Saved: encoder_stage1.pt + adapter_head.pt")

    # ------------------------------------------------------------------
    # Stage 2: AB-MIL (unchanged from the baseline narrowed script)
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log(f"STAGE 2: AB-MIL ({label_enc.n_classes()}-class head, frozen encoder)")
    log("=" * 60)
    train_dataset = PseudoTubeDataset(
        str(train_m_path), label_enc, gene_names=gene_names, preload=True,
    )
    val_dataset = PseudoTubeDataset(
        str(val_m_path), label_enc, gene_names=gene_names, preload=True,
    )
    model = build_mil_model(
        encoder,
        embed_dim=args.embed_dim,
        attention_hidden_dim=args.attention_hidden_dim,
        n_classes=label_enc.n_classes(),
        encoder_frozen=True,
    )
    log(f"  Model: embed_dim={args.embed_dim}, attn_hidden={args.attention_hidden_dim}, "
        f"n_classes={label_enc.n_classes()}")
    log(f"  Stage 2: {args.stage2_epochs} epochs, lr={args.lr}, warmup={args.lr_warmup_epochs}")

    dynamics = train_mil(
        model, train_dataset,
        n_epochs=args.stage2_epochs,
        lr=args.lr,
        momentum=STAGE2_MOMENTUM,
        lr_warmup_epochs=args.lr_warmup_epochs,
        log_every_n_epochs=1,
        device=device,
        seed=args.seed,
        verbose=True,
        val_dataset=val_dataset,
    )
    torch.save(model.state_dict(), out_dir / "model_stage2.pt")
    log("  Saved: model_stage2.pt")

    log("\nSaving dynamics...")
    payload = {
        "records": dynamics["records"],
        "val_records": dynamics["val_records"],
        "logged_epochs": dynamics["logged_epochs"],
        "confusion_entropy_trajectory": dynamics["confusion_entropy_trajectory"],
        "val_confusion_entropy_trajectory": dynamics["val_confusion_entropy_trajectory"],
        "loss_components": dynamics["loss_components"],
        "label_encoder_cytokines": list(label_enc.cytokines),
        "seed": args.seed,
        "val_pseudo_donors": args.val_donors,
        "stage2_epochs": args.stage2_epochs,
        "stage2_lr": args.lr,
        "dataset": "Sheu2024",
        "variant": "adapter_aux",
        "adapter_classes": ADAPTER_CLASSES,
        "aux_weight": args.aux_weight,
    }
    with open(out_dir / "dynamics.pkl", "wb") as fh:
        pickle.dump(payload, fh)
    log("  Saved: dynamics.pkl")
    log(f"  Records: {len(dynamics['records'])} train, {len(dynamics['val_records'])} val")

    _plot_loss_curve(dynamics["loss_components"], out_dir, log)
    summary = {
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "variant": "adapter_aux",
        "aux_weight": args.aux_weight,
        "embed_dim": args.embed_dim,
        "attention_hidden_dim": args.attention_hidden_dim,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "stage2_lr": args.lr,
        "n_train_records": len(dynamics["records"]),
        "n_val_records": len(dynamics["val_records"]),
        "final_loss": float(dynamics["loss_components"]["total"][-1])
            if dynamics["loss_components"].get("total") else None,
        "adapter_classes": ADAPTER_CLASSES,
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log("  Saved: run_summary.json")
    log("\nDone.")
    log_file.close()


if __name__ == "__main__":
    main()
