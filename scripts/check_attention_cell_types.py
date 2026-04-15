"""
Q1 attention proxy check: do high-attention cells match known primary responders?

Extracts mean attention weight per cell type per cytokine from the trained model.
Aggregates to donor level (median per donor, mean across donors).
Reports whether NK cells dominate IL-12 tube attention — if yes, the attention-as-
primary-responder proxy for Exp 3 is empirically grounded.

Key cytokines checked: IL-12, IFN-gamma, IL-4, IL-2, TNF-alpha, PBS.
Known ground-truth:
  IL-12  → NK cells should dominate (primary IFN-γ inducers)
  IFN-γ  → NK + Monocytes (primary STAT1 responders)
  IL-4   → B cells + T cells (primary STAT6 responders)
  IL-2   → CD4_T + CD8_T (primary proliferators)
  TNF-α  → Monocytes (primary NF-κB responders)

Usage:
    python scripts/check_attention_cell_types.py \
        --run_dir results/oesinghaus_full/run_20260412_161758_seed42
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder, build_mil_model

HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"

KEY_CYTOKINES = ["IL-12", "IFN-gamma", "IL-4", "IL-2", "TNF-alpha", "PBS"]

# Known expected dominant cell types per cytokine (for pass/fail labeling)
EXPECTED_DOMINANT = {
    "IL-12":     ["NK", "NK CD56bright"],
    "IFN-gamma": ["NK", "CD14 Mono", "CD14_Mono"],
    "IL-4":      ["B Naive", "B Intermediate/Memory", "CD4_T", "CD4 Naive"],
    "IL-2":      ["CD4_T", "CD8_T", "CD4 Naive", "CD8 Naive"],
    "TNF-alpha": ["CD14 Mono", "CD14_Mono"],
}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


def _load_label_encoder(run_dir: Path) -> CytokineLabel:
    with open(run_dir / "label_encoder.json") as f:
        data = json.load(f)
    cytokines_list = data["cytokines"]
    le = CytokineLabel()
    le._label_to_idx = {cyt: i for i, cyt in enumerate(cytokines_list)}
    le._idx_to_label = {i: cyt for i, cyt in enumerate(cytokines_list)}
    return le


def _load_model(run_dir: Path, label_enc, gene_names, device):
    state_dict = torch.load(
        run_dir / "model_stage2.pt", map_location="cpu", weights_only=False
    )
    # Infer n_cell_types
    n_cell_types = state_dict["encoder.cell_type_head.weight"].shape[0]
    encoder = build_encoder(
        n_input_genes=len(gene_names), n_cell_types=n_cell_types, embed_dim=128
    )
    model = build_mil_model(
        encoder, embed_dim=128, attention_hidden_dim=64,
        n_classes=label_enc.n_classes(), encoder_frozen=True,
    )
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    model.eval()
    return model


def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)

    with open(HVG_PATH) as f:
        gene_names = json.load(f)
    label_enc = _load_label_encoder(run_dir)
    model = _load_model(run_dir, label_enc, gene_names, args.device)

    train_dataset = PseudoTubeDataset(
        str(run_dir / "manifest_train.json"),
        label_enc,
        gene_names=gene_names,
        preload=False,
    )
    _log(f"Dataset: {len(train_dataset)} tubes")

    # ----------------------------------------------------------------
    # Collect per-tube attention profiles
    # ----------------------------------------------------------------
    # Structure: {cytokine -> {donor -> list of {cell_type -> mean_attention}}}
    attn_by_cyt_donor: dict = defaultdict(lambda: defaultdict(list))

    _log("Extracting attention profiles...")
    with torch.no_grad():
        for idx in range(len(train_dataset)):
            X, _label, donor, cytokine = train_dataset[idx]
            if cytokine not in KEY_CYTOKINES:
                continue
            entry = train_dataset.entries[idx]
            adata = sc.read_h5ad(entry["path"])
            if "cell_type" not in adata.obs.columns:
                continue
            cell_types = adata.obs["cell_type"].values

            X_dev = X.to(torch.device(args.device))
            outputs = model(X_dev)
            a = outputs[1].cpu().numpy()  # (N,) attention weights

            # Mean attention per cell type within this tube
            ct_mean = {}
            for ct in np.unique(cell_types):
                mask = cell_types == ct
                ct_mean[ct] = float(a[mask].mean())

            attn_by_cyt_donor[cytokine][donor].append(ct_mean)

    _log("Done extracting.")

    # ----------------------------------------------------------------
    # Aggregate: median per donor, mean across donors
    # ----------------------------------------------------------------
    # Final structure: {cytokine -> {cell_type -> donor_aggregated_mean_attention}}
    results = {}
    for cytokine, donor_dict in attn_by_cyt_donor.items():
        all_cell_types = set()
        for tubes in donor_dict.values():
            for ct_map in tubes:
                all_cell_types.update(ct_map.keys())

        donor_means: dict = defaultdict(list)
        for donor, tubes in donor_dict.items():
            for ct in all_cell_types:
                vals = [t[ct] for t in tubes if ct in t]
                if vals:
                    donor_means[ct].append(float(np.median(vals)))

        results[cytokine] = {
            ct: float(np.mean(vals))
            for ct, vals in donor_means.items()
            if vals
        }

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    _log()
    _log("=" * 65)
    _log("ATTENTION PROXY CHECK — Mean attention per cell type, donor-aggregated")
    _log("Metric: median(mean_attention_per_celltype) across tubes per donor,")
    _log("        then mean across donors")
    _log("=" * 65)

    all_pass = []
    for cytokine in KEY_CYTOKINES:
        if cytokine not in results:
            _log(f"\n{cytokine}: no tubes found in training set.")
            continue

        ct_attn = results[cytokine]
        sorted_cts = sorted(ct_attn.items(), key=lambda x: x[1], reverse=True)

        _log(f"\n{'─'*55}")
        _log(f"{cytokine} (top cell types by mean attention):")
        _log(f"  {'Cell type':<30} {'Mean attention':>14}")
        for ct, val in sorted_cts[:10]:
            _log(f"  {ct:<30} {val:>14.5f}")

        expected = EXPECTED_DOMINANT.get(cytokine, [])
        if expected:
            top3 = [ct for ct, _ in sorted_cts[:3]]
            match = any(ct in top3 for ct in expected)
            _log(f"\n  Expected dominant: {expected}")
            _log(f"  Top-3 observed:    {top3}")
            _log(f"  Match: {'YES ✓' if match else 'NO ✗'}")
            all_pass.append(match)

    _log()
    _log("=" * 65)
    n_pass = sum(all_pass)
    n_total = len(all_pass)
    _log(f"SUMMARY: {n_pass}/{n_total} cytokines match expected dominant cell type in top-3 attention.")
    if n_pass >= n_total * 0.6:
        _log("VERDICT: Attention proxy is empirically grounded.")
        _log("  → Attention-weighted KL loss for Exp 3 is justified.")
    else:
        _log("VERDICT: Attention proxy is NOT reliably grounded.")
        _log("  → Use uniform KL loss; treat attention weighting as caveat only.")
    _log("=" * 65)


if __name__ == "__main__":
    main()
