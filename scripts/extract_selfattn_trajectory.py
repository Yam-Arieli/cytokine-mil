"""
Extract self-attention trajectories from CytokineSelfAttnMIL checkpoints (§34).

Same frozen-encoder trick as scripts/extract_attention_trajectory.py: the encoder
is frozen in Stage-2, so cell embeddings H are identical across all epochs. We
compute H ONCE per tube, then for each checkpoint reload the SAB + pooling +
classifier params and re-apply them to the cached H — no per-checkpoint h5ad
reads, no per-checkpoint encoder forwards.

Emits TWO pickles:

1. <run_dir>/attention_trajectory.pkl  — IDENTICAL structure to §33
   (pooling weights a_i as the per-cell-type attention), so
   scripts/analyze_attention_dynamics.py runs UNCHANGED (P1-P4 comparable to §33):
     {epochs, trajectory, trajectory_per_donor, concentration, cell_types, cytokines}

2. <run_dir>/interaction_trajectory.pkl  — the NEW cell x cell readout:
     {
       "epochs":             [...],
       "interaction":        {cytokine: {"τ||σ": np.array(n_epochs)}},          # donor-MEAN M[τ,σ]
       "interaction_per_donor": {cytokine: {"τ||σ": {donor: np.array(n_epochs)}}},
       "offdiag":            {cytokine: np.array(n_epochs)},                     # donor-MEAN cross-type frac
       "offdiag_per_donor":  {cytokine: {donor: np.array(n_epochs)}},
       "cell_types":         [...],
       "cytokines":          [...],
       "pair_sep":           "||",
     }
   M[τ,σ] = mean over τ-cells of the total attention they place on σ-cells
   (rows sum to 1 over σ). offdiag = mean over cells of cross-cell-type attention
   (1 = all attention on other types; ~0 = self/diagonal => cells don't interact).

Usage:
    python scripts/extract_selfattn_trajectory.py --run_dir <dir> [--hvg_path <genes.json>]
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
from cytokine_mil.experiment_setup import build_encoder, build_selfattn_model

HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
PAIR_SEP = "||"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir",           required=True)
    p.add_argument("--checkpoint_subdir", default="checkpoints")
    p.add_argument("--hvg_path",          default=HVG_PATH)
    p.add_argument("--tubes_per_cyt_donor", type=int, default=1)
    p.add_argument("--exclude_cell_types", default=None)
    p.add_argument("--checkpoint_stride", type=int, default=1)
    p.add_argument("--embed_dim",         type=int, default=128)
    p.add_argument("--attention_hidden_dim", type=int, default=64)
    p.add_argument("--sab_heads",         type=int, default=4)
    p.add_argument("--sab_layers",        type=int, default=1)
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


def _build_full_model(ckpt_path, label_enc, gene_names, device, args):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_cell_types = state["encoder.cell_type_head.weight"].shape[0]
    encoder = build_encoder(len(gene_names), n_cell_types=n_cell_types, embed_dim=args.embed_dim)
    model = build_selfattn_model(
        encoder, embed_dim=args.embed_dim, attention_hidden_dim=args.attention_hidden_dim,
        n_classes=label_enc.n_classes(), encoder_frozen=True,
        sab_heads=args.sab_heads, sab_layers=args.sab_layers,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_tube(entry, gene_names):
    adata = anndata.read_h5ad(entry["path"])
    avail = [g for g in gene_names if g in adata.var_names]
    X = adata[:, avail].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
    cell_types = list(adata.obs["cell_type"].astype(str))
    return X_tensor, cell_types


def _subset_entries(entries, tubes_per_cyt_donor: int):
    by_key = defaultdict(list)
    for e in entries:
        by_key[(e["cytokine"], e["donor"])].append(e)
    subset = []
    for key in sorted(by_key):
        chosen = sorted(by_key[key], key=lambda e: e.get("tube_idx", 0))
        subset.extend(chosen[:tubes_per_cyt_donor])
    return subset


@torch.no_grad()
def _build_cache(model, subset, gene_names, device, exclude=None, store_H=True):
    cache = []
    for entry in subset:
        X, cell_types = _load_tube(entry, gene_names)
        cts = np.asarray(cell_types)
        if exclude:
            keep = np.array([c not in exclude for c in cts])
            if keep.any() and not keep.all():
                X, cts = X[keep], cts[keep]
        item = {"cytokine": entry["cytokine"], "donor": entry["donor"], "cell_types": cts}
        if store_H:
            item["H"] = model.encoder(X.to(device))
        else:
            item["X"] = X
        cache.append(item)
    return cache


def _celltype_interaction(A: np.ndarray, cts: np.ndarray):
    """From row-normalised A (N,N) and per-cell types, compute:
       M[τ,σ] = mean over τ-cells of total attention placed on σ-cells (Σ_σ M=1),
       offdiag = mean over cells of cross-cell-type attention (1 - within-type).
    Returns (dict {"τ||σ": value}, offdiag_scalar)."""
    types = np.unique(cts)
    masks = {t: (cts == t) for t in types}
    M = {}
    within_per_cell = np.zeros(A.shape[0], dtype=np.float64)
    for t, mt in masks.items():
        within_per_cell[mt] = A[mt][:, mt].sum(axis=1)  # attention each t-cell keeps within t
        row_means = A[mt]  # (|t|, N)
        for s, ms in masks.items():
            M[f"{t}{PAIR_SEP}{s}"] = float(row_means[:, ms].sum(axis=1).mean())
    offdiag = float(np.mean(1.0 - within_per_cell))
    return M, offdiag


@torch.no_grad()
def _snapshot(model, cache, device, recompute_H=False):
    """Returns pooling means/ginis (per cell type) + interaction M (per type pair)
    + offdiag (scalar), each nested {cytokine: {donor: {key: [per-tube]}}}."""
    means = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # pooling per cell type
    ginis = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    inter = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # M[τ,σ]
    offd = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))    # offdiag scalar
    for item in cache:
        H = model.encoder(item["X"].to(device)) if recompute_H else item["H"]
        a, A = model.readout_from_H(H)
        a = a.cpu().numpy()
        A = A.cpu().numpy()
        cts = item["cell_types"]
        cyt, donor = item["cytokine"], item["donor"]
        for ct in np.unique(cts):
            vals = a[cts == ct]
            means[cyt][donor][ct].append(float(np.mean(vals)))
            ginis[cyt][donor][ct].append(float(gini(vals)))
        M, offdiag = _celltype_interaction(A, cts)
        for key, v in M.items():
            inter[cyt][donor][key].append(v)
        offd[cyt][donor]["__offdiag__"].append(offdiag)
    return means, ginis, inter, offd


def _append_donor_level(epoch_nested, accum):
    for cyt, donors in epoch_nested.items():
        for donor, key_dict in donors.items():
            for key, tube_vals in key_dict.items():
                accum[cyt][key][donor].append(float(np.median(tube_vals)))


def _donor_mean(per_donor):
    out = defaultdict(dict)
    for cyt, by_key in per_donor.items():
        for key, by_donor in by_key.items():
            arrs = [np.asarray(v, dtype=float) for v in by_donor.values()]
            out[cyt][key] = np.mean(np.stack(arrs), axis=0)
    return out


def _to_arrays(per_donor):
    return {
        cyt: {key: {d: np.asarray(v, dtype=float) for d, v in by_donor.items()}
              for key, by_donor in by_key.items()}
        for cyt, by_key in per_donor.items()
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
    if args.checkpoint_stride > 1:
        ckpt_files = sorted(set(ckpt_files[::args.checkpoint_stride]) | {ckpt_files[-1]})

    def epoch_of(p): return int(p.stem.replace("epoch_", ""))
    epochs = [epoch_of(f) for f in ckpt_files]
    _log(f"Using {len(ckpt_files)} checkpoints (stride {args.checkpoint_stride}): "
         f"epochs {epochs[0]}..{epochs[-1]}")

    exclude = (set(s.strip() for s in args.exclude_cell_types.split(",") if s.strip())
               if args.exclude_cell_types else None)
    if exclude:
        _log(f"Excluding cell types: {sorted(exclude)}")

    with open(args.hvg_path) as f:
        gene_names = json.load(f)
    label_enc = _load_label_encoder(run_dir)
    with open(run_dir / "manifest_train.json") as f:
        entries = json.load(f)
    subset = _subset_entries(entries, args.tubes_per_cyt_donor)
    _log(f"Train tubes: {len(entries)} -> analysis subset: {len(subset)}")

    model = _build_full_model(ckpt_files[0], label_enc, gene_names, device, args)

    unfrozen = False
    if len(ckpt_files) > 1:
        s_last = torch.load(ckpt_files[-1], map_location="cpu", weights_only=False)
        w0 = model.encoder.cell_type_head.weight.detach().cpu()
        wL = s_last["encoder.cell_type_head.weight"].cpu()
        unfrozen = not torch.allclose(w0, wL)
    _log(f"Encoder mode: {'UNFROZEN (recompute H)' if unfrozen else 'FROZEN (cache H once)'}")

    cache = _build_cache(model, subset, gene_names, device, exclude=exclude,
                         store_H=not unfrozen)
    _log(f"  cached {len(cache)} tubes")

    pool_pd = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gini_pd = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    inter_pd = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    offd_pd = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for i, (ckpt_path, epoch) in enumerate(zip(ckpt_files, epochs)):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.to(device).eval()
        means, ginis, inter, offd = _snapshot(model, cache, device, recompute_H=unfrozen)
        _append_donor_level(means, pool_pd)
        _append_donor_level(ginis, gini_pd)
        _append_donor_level(inter, inter_pd)
        _append_donor_level(offd, offd_pd)
        if (i + 1) % 25 == 0 or i == len(ckpt_files) - 1:
            _log(f"  processed checkpoint {i + 1}/{len(ckpt_files)} (epoch {epoch})")

    # ---- pooling (§33-compatible) ----
    trajectory = _donor_mean(pool_pd)
    concentration = _donor_mean(gini_pd)
    all_cell_types = sorted({ct for by_ct in trajectory.values() for ct in by_ct})
    all_cytokines = sorted(trajectory.keys())
    pool_out = {
        "epochs": epochs,
        "trajectory": {c: dict(d) for c, d in trajectory.items()},
        "trajectory_per_donor": _to_arrays(pool_pd),
        "concentration": {c: dict(d) for c, d in concentration.items()},
        "cell_types": all_cell_types,
        "cytokines": all_cytokines,
    }
    with open(run_dir / "attention_trajectory.pkl", "wb") as f:
        pickle.dump(pool_out, f)
    _log(f"Saved: {run_dir / 'attention_trajectory.pkl'}")

    # ---- interaction (NEW) ----
    inter_mean = _donor_mean(inter_pd)
    offd_mean = _donor_mean(offd_pd)   # {cyt: {"__offdiag__": arr}}
    offdiag = {c: d["__offdiag__"] for c, d in offd_mean.items() if "__offdiag__" in d}
    offdiag_per_donor = {
        c: {donor: np.asarray(v, dtype=float) for donor, v in by_donor.items()}
        for c, by_key in _to_arrays(offd_pd).items()
        for key, by_donor in by_key.items() if key == "__offdiag__"
    }
    inter_out = {
        "epochs": epochs,
        "interaction": {c: dict(d) for c, d in inter_mean.items()},
        "interaction_per_donor": _to_arrays(inter_pd),
        "offdiag": offdiag,
        "offdiag_per_donor": offdiag_per_donor,
        "cell_types": all_cell_types,
        "cytokines": all_cytokines,
        "pair_sep": PAIR_SEP,
    }
    with open(run_dir / "interaction_trajectory.pkl", "wb") as f:
        pickle.dump(inter_out, f)
    _log(f"Saved: {run_dir / 'interaction_trajectory.pkl'}")
    _log(f"  Cytokines: {len(all_cytokines)}, Cell types: {len(all_cell_types)}, "
         f"Epochs: {epochs[0]}..{epochs[-1]} ({len(epochs)})")


if __name__ == "__main__":
    main()
