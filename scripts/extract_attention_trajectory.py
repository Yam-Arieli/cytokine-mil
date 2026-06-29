"""
Extract per-cell-type attention trajectory from model checkpoints (CLAUDE.md §33).

Efficient design (handles every-epoch checkpoints, 250+ per seed):
the encoder is FROZEN in Stage-2, so cell embeddings H are identical across all
epochs. We therefore compute H ONCE per tube, then for each checkpoint we load
only the attention params (attention.V/.w) and re-apply them to the cached H —
no per-checkpoint h5ad reads, no per-checkpoint encoder forwards. This turns a
~90-hour re-forward (9100 tubes x 250 checkpoints) into ~minutes.

Analysis tube subset: one tube per (cytokine, donor) by default (~910 tubes),
which is what the donor-level relay-lag / recruitment readouts need. Configurable
via --tubes_per_cyt_donor.

Output: <run_dir>/attention_trajectory.pkl
  {
    "epochs":               [1, 2, ..., 250],
    "trajectory":           {cytokine: {cell_type: np.array(n_epochs)}},          # donor-MEAN
    "trajectory_per_donor": {cytokine: {cell_type: {donor: np.array(n_epochs)}}}, # per-donor
    "concentration":        {cytokine: {cell_type: np.array(n_epochs)}},          # donor-MEAN Gini
    "cell_types":           sorted list of all cell types observed,
    "cytokines":            sorted list of all cytokines,
  }

Consumed by scripts/analyze_attention_dynamics.py /
cytokine_mil.analysis.attention_dynamics.

Usage:
    python scripts/extract_attention_trajectory.py --run_dir <dir> [--hvg_path <genes.json>]
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

from cytokine_mil.analysis.attention_dynamics import gini
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model

# Cluster default; override with --hvg_path for local/demo runs.
HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir",           required=True)
    p.add_argument("--checkpoint_subdir", default="checkpoints",
                   help="Subdirectory under run_dir containing epoch_*.pt files.")
    p.add_argument("--hvg_path",          default=HVG_PATH,
                   help="JSON list of gene names (HVGs) the model was trained on.")
    p.add_argument("--tubes_per_cyt_donor", type=int, default=1,
                   help="How many tubes to keep per (cytokine, donor) for the analysis "
                        "subset (default 1 — enough for donor-level stats).")
    p.add_argument("--embed_dim",         type=int, default=128)
    p.add_argument("--attention_hidden_dim", type=int, default=64)
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


def _build_full_model(ckpt_path: Path, label_enc, gene_names, device,
                      embed_dim: int, attention_hidden_dim: int):
    """Build a CytokineABMIL and load one checkpoint (used for the frozen encoder
    and as the container we reload attention params into per checkpoint)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_cell_types = state["encoder.cell_type_head.weight"].shape[0]
    encoder = build_encoder(len(gene_names), n_cell_types=n_cell_types, embed_dim=embed_dim)
    model = build_mil_model(encoder, embed_dim=embed_dim,
                            attention_hidden_dim=attention_hidden_dim,
                            n_classes=label_enc.n_classes(), encoder_frozen=True)
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_tube(entry, gene_names):
    """Load X tensor and cell_type list from h5ad (aligned to gene_names)."""
    adata = anndata.read_h5ad(entry["path"])
    avail = [g for g in gene_names if g in adata.var_names]
    X = adata[:, avail].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
    cell_types = list(adata.obs["cell_type"].astype(str))
    return X_tensor, cell_types


def _subset_entries(entries, tubes_per_cyt_donor: int):
    """One (or K) tube(s) per (cytokine, donor) — deterministic by tube_idx."""
    by_key = defaultdict(list)
    for e in entries:
        by_key[(e["cytokine"], e["donor"])].append(e)
    subset = []
    for key in sorted(by_key):
        chosen = sorted(by_key[key], key=lambda e: e.get("tube_idx", 0))
        subset.extend(chosen[:tubes_per_cyt_donor])
    return subset


@torch.no_grad()
def _build_embedding_cache(model, subset, gene_names, device):
    """Compute H = encoder(X) ONCE per tube (encoder is frozen across epochs)."""
    cache = []
    for entry in subset:
        X, cell_types = _load_tube(entry, gene_names)
        H = model.encoder(X.to(device))  # (N, embed_dim)
        cache.append({
            "cytokine": entry["cytokine"], "donor": entry["donor"],
            "cell_types": np.asarray(cell_types), "H": H,
        })
    return cache


@torch.no_grad()
def _attention_snapshot(model, cache):
    """Apply the model's current attention params to each cached H.

    Returns (means, ginis), each {cytokine: {donor: {cell_type: [per-tube value]}}}.
    """
    means = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ginis = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in cache:
        a = model.attention(item["H"]).cpu().numpy()  # (N,)
        cts = item["cell_types"]
        for ct in np.unique(cts):
            vals = a[cts == ct]
            means[item["cytokine"]][item["donor"]][ct].append(float(np.mean(vals)))
            ginis[item["cytokine"]][item["donor"]][ct].append(float(gini(vals)))
    return means, ginis


def _append_donor_level(epoch_nested, accum):
    """Reduce one epoch's {cyt:{donor:{ct:[per-tube]}}} to a donor-level value
    (median across tubes) and append into accum {cyt:{ct:{donor:[over epochs]}}}."""
    for cyt, donors in epoch_nested.items():
        for donor, ct_dict in donors.items():
            for ct, tube_vals in ct_dict.items():
                accum[cyt][ct][donor].append(float(np.median(tube_vals)))


def _donor_mean(per_donor):
    out = defaultdict(dict)
    for cyt, by_ct in per_donor.items():
        for ct, by_donor in by_ct.items():
            arrs = [np.asarray(v, dtype=float) for v in by_donor.values()]
            out[cyt][ct] = np.mean(np.stack(arrs), axis=0)
    return out


def _to_arrays(per_donor):
    return {
        cyt: {ct: {d: np.asarray(v, dtype=float) for d, v in by_donor.items()}
              for ct, by_donor in by_ct.items()}
        for cyt, by_ct in per_donor.items()
    }


def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)

    ckpt_dir = run_dir / args.checkpoint_subdir
    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpt_files:
        _log(f"ERROR: No checkpoint files in {ckpt_dir}")
        sys.exit(1)

    def epoch_of(p): return int(p.stem.replace("epoch_", ""))
    epochs = [epoch_of(f) for f in ckpt_files]
    _log(f"Found {len(ckpt_files)} checkpoints: epochs {epochs[0]}..{epochs[-1]}")

    with open(args.hvg_path) as f:
        gene_names = json.load(f)
    label_enc = _load_label_encoder(run_dir)
    with open(run_dir / "manifest_train.json") as f:
        entries = json.load(f)
    subset = _subset_entries(entries, args.tubes_per_cyt_donor)
    _log(f"Train tubes: {len(entries)} -> analysis subset: {len(subset)} "
         f"({args.tubes_per_cyt_donor}/cytokine-donor)")

    # Frozen encoder: build once from the first checkpoint, cache H per tube once.
    model = _build_full_model(ckpt_files[0], label_enc, gene_names, device,
                              args.embed_dim, args.attention_hidden_dim)
    _log("Computing embedding cache H (once; encoder is frozen)...")
    cache = _build_embedding_cache(model, subset, gene_names, device)
    _log(f"  cached H for {len(cache)} tubes")

    # Sanity: encoder weights are identical across checkpoints (frozen).
    if len(ckpt_files) > 1:
        s_last = torch.load(ckpt_files[-1], map_location="cpu", weights_only=False)
        w0 = model.encoder.cell_type_head.weight.detach().cpu()
        wL = s_last["encoder.cell_type_head.weight"].cpu()
        if not torch.allclose(w0, wL):
            _log("  WARNING: encoder differs across checkpoints — H cache assumption "
                 "violated (was Stage-3 / unfrozen run?). Results may be wrong.")

    attn_per_donor = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gini_per_donor = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for i, (ckpt_path, epoch) in enumerate(zip(ckpt_files, epochs)):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.to(device).eval()
        means, ginis = _attention_snapshot(model, cache)
        _append_donor_level(means, attn_per_donor)
        _append_donor_level(ginis, gini_per_donor)
        if (i + 1) % 25 == 0 or i == len(ckpt_files) - 1:
            _log(f"  processed checkpoint {i + 1}/{len(ckpt_files)} (epoch {epoch})")

    trajectory_per_donor = _to_arrays(attn_per_donor)
    trajectory = _donor_mean(attn_per_donor)
    concentration = _donor_mean(gini_per_donor)

    all_cell_types = sorted({ct for by_ct in trajectory.values() for ct in by_ct})
    all_cytokines = sorted(trajectory.keys())

    out = {
        "epochs": epochs,
        "trajectory": {c: dict(d) for c, d in trajectory.items()},
        "trajectory_per_donor": trajectory_per_donor,
        "concentration": {c: dict(d) for c, d in concentration.items()},
        "cell_types": all_cell_types,
        "cytokines": all_cytokines,
    }

    out_path = run_dir / "attention_trajectory.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    _log(f"\nSaved: {out_path}")
    _log(f"  Cytokines: {len(all_cytokines)}, Cell types: {len(all_cell_types)}, "
         f"Epochs: {epochs[0]}..{epochs[-1]} ({len(epochs)})")


if __name__ == "__main__":
    main()
