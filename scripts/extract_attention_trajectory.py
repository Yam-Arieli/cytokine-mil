"""
Method 2: Extract per-cell-type attention trajectory from model checkpoints.

For each checkpoint epoch:
  - Load model weights
  - Forward pass on every training tube -> attention weights a_i (N-dim)
  - Load cell_type from h5ad obs["cell_type"] for the same tube
  - Group by cell type: mean attention per cell type per cytokine
  - Aggregate to donor level (median per donor, mean across donors)

Output: <run_dir>/attention_trajectory.pkl
  {
    "epochs":        [25, 50, 75, 100],
    "trajectory":    {
        cytokine_name: {
            cell_type: np.array(n_epochs)   # mean attention over time
        }
    },
    "cell_types":    sorted list of all cell types observed
    "cytokines":     sorted list of all cytokines
  }

Usage:
    python scripts/extract_attention_trajectory.py --run_dir <dir>
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import anndata
import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
import sys; sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model

HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir",           required=True)
    p.add_argument("--checkpoint_subdir", default="checkpoints",
                   help="Subdirectory under run_dir containing epoch_*.pt files.")
    p.add_argument("--device",            default="cpu")
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


def _load_label_encoder(run_dir: Path) -> CytokineLabel:
    with open(run_dir / "label_encoder.json") as f:
        data = json.load(f)
    le = CytokineLabel()
    le._label_to_idx = {c: i for i, c in enumerate(data["cytokines"])}
    le._idx_to_label = {i: c for i, c in enumerate(data["cytokines"])}
    return le


def _load_model(run_dir: Path, ckpt_path: Path, label_enc, gene_names, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_cell_types = state["encoder.cell_type_head.weight"].shape[0]
    encoder = build_encoder(len(gene_names), n_cell_types=n_cell_types, embed_dim=128)
    model = build_mil_model(encoder, embed_dim=128, attention_hidden_dim=64,
                            n_classes=label_enc.n_classes(), encoder_frozen=True)
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_tube(entry, gene_names):
    """Load X tensor and cell_type list from h5ad."""
    adata = anndata.read_h5ad(entry["path"])
    # align to gene_names
    gene_set = set(gene_names)
    avail = [g for g in gene_names if g in adata.var_names]
    X = adata[:, avail].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    cell_types = list(adata.obs["cell_type"].astype(str))
    return X_tensor, cell_types


@torch.no_grad()
def extract_epoch(model, entries, gene_names, device):
    """
    Returns: {cytokine: {donor: {cell_type: [mean_attn]}}}
    The innermost list has one element (this epoch's mean), for easy stacking later.
    """
    cyt_donor_ct = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for entry in entries:
        X, cell_types = _load_tube(entry, gene_names)
        X = X.to(device)
        cytokine = entry["cytokine"]
        donor    = entry["donor"]

        # Forward pass – attention weights
        _, a, _ = model(X)
        a_np = a.cpu().numpy()  # shape (N,)

        # Group by cell type
        ct_attn = defaultdict(list)
        for i, ct in enumerate(cell_types):
            ct_attn[ct].append(a_np[i])

        for ct, vals in ct_attn.items():
            cyt_donor_ct[cytokine][donor][ct].append(float(np.mean(vals)))

    return cyt_donor_ct


def aggregate_donor_level(cyt_donor_ct):
    """
    cyt_donor_ct: {cytokine: {donor: {cell_type: [values_this_epoch]}}}
    Returns: {cytokine: {cell_type: mean_across_donors}}
    """
    result = defaultdict(dict)
    for cyt, donors in cyt_donor_ct.items():
        # collect all cell types across donors
        all_cts = set()
        for ct_dict in donors.values():
            all_cts.update(ct_dict.keys())

        for ct in all_cts:
            donor_means = [
                np.mean(donors[d][ct]) for d in donors if ct in donors[d]
            ]
            result[cyt][ct] = float(np.mean(donor_means)) if donor_means else 0.0
    return result


def main():
    args   = _parse_args()
    run_dir = Path(args.run_dir)
    device  = torch.device(args.device)

    ckpt_dir = run_dir / args.checkpoint_subdir
    if not ckpt_dir.exists():
        _log(f"ERROR: No checkpoints directory found at {ckpt_dir}")
        sys.exit(1)

    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpt_files:
        _log(f"ERROR: No checkpoint files in {ckpt_dir}")
        sys.exit(1)

    _log(f"Found {len(ckpt_files)} checkpoints: {[f.name for f in ckpt_files]}")

    with open(HVG_PATH) as f:
        gene_names = json.load(f)

    label_enc = _load_label_encoder(run_dir)

    with open(run_dir / "manifest_train.json") as f:
        entries = json.load(f)
    _log(f"Training tubes: {len(entries)}")

    # Parse epoch numbers from filenames
    def epoch_of(p): return int(p.stem.replace("epoch_", ""))

    epochs     = [epoch_of(f) for f in ckpt_files]
    trajectory = defaultdict(lambda: defaultdict(list))
    # trajectory[cytokine][cell_type] = list of mean_attn per epoch

    for ckpt_path, epoch in zip(ckpt_files, epochs):
        _log(f"\n--- Epoch {epoch} ({ckpt_path.name}) ---")
        model = _load_model(run_dir, ckpt_path, label_enc, gene_names, device)
        cyt_donor_ct = extract_epoch(model, entries, gene_names, device)
        agg = aggregate_donor_level(cyt_donor_ct)

        for cyt, ct_dict in agg.items():
            for ct, val in ct_dict.items():
                trajectory[cyt][ct].append(val)

        _log(f"  Processed {len(agg)} cytokines.")
        del model

    # Convert to numpy arrays
    trajectory_np = {
        cyt: {ct: np.array(vals) for ct, vals in ct_dict.items()}
        for cyt, ct_dict in trajectory.items()
    }

    all_cell_types = sorted({ct for ct_dict in trajectory_np.values() for ct in ct_dict})
    all_cytokines  = sorted(trajectory_np.keys())

    out = {
        "epochs":     epochs,
        "trajectory": trajectory_np,
        "cell_types": all_cell_types,
        "cytokines":  all_cytokines,
    }

    out_path = run_dir / "attention_trajectory.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    _log(f"\nSaved: {out_path}")
    _log(f"  Cytokines: {len(all_cytokines)}, Cell types: {len(all_cell_types)}, Epochs: {epochs}")


if __name__ == "__main__":
    main()
