"""
Extract per-gene attribution trajectory from binary AB-MIL checkpoints.

Mirrors extract_attention_trajectory.py but extracts per-gene attribution
(not per-cell-type attention) for the gene learning-order experiment.

For each stimulus in the run dir:
  1. Load checkpoints_<stimulus>/epoch_*.pt in epoch order.
  2. For each checkpoint, build the binary model by inferring HPs from
     the state dict (same approach as run_binary_ig_probe.py).
  3. For each of the stimulus's training tubes: compute per-gene attribution
     via raw_gradient (primary: one backward / tube, ~20x cheaper than IG).
  4. Average |attribution| over all stimulus tubes -> per-gene attr at this epoch.
  5. Accumulate across epochs into a long-format table.

Also computes Integrated Gradients at the FINAL epoch only (optional sanity).

Output: <run_dir>/gene_attribution_trajectory.parquet
  columns: gene (str), epoch (int), stimulus (str), seed (int), attr (float)

  attr = mean_over_tubes(mean_over_cells(|raw_gradient|)) per gene per epoch.

Usage:
  python scripts/extract_gene_attribution_trajectory.py --run_dir results/gene_learning_order/seed_42
  python scripts/extract_gene_attribution_trajectory.py --run_dir ... --device cuda --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis.attribution import (  # noqa: E402
    integrated_gradients,
    mean_over_cells,
    raw_gradient,
)
from cytokine_mil.models.attention import AttentionModule  # noqa: E402
from cytokine_mil.models.bag_classifier import BagClassifier  # noqa: E402
from cytokine_mil.models.cytokine_abmil import CytokineABMIL  # noqa: E402
from cytokine_mil.models.instance_encoder import InstanceEncoder  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Extract per-gene attribution trajectory from binary checkpoints.",
    )
    p.add_argument("--run_dir", required=True,
                   help="Run dir produced by train_sheu_binary_learning_order.py")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed to embed in the output parquet (inferred from run_dir "
                        "if name matches seed_<N>; override with this flag)")
    p.add_argument("--device", default="cpu",
                   help="Torch device (default: cpu)")
    p.add_argument("--ig_final", action="store_true", default=False,
                   help="Also compute IG at the final epoch (saved as "
                        "gene_ig_final_<stimulus>.json per stimulus)")
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# HP inference (mirrors run_binary_ig_probe.py:_infer_hps_from_state_dict)
# ---------------------------------------------------------------------------

def _infer_hps_from_state(state: dict, n_input_genes: int) -> dict:
    """Infer binary model HPs from state-dict weight shapes.

    Keys inspected:
      encoder.input_proj.0.weight  shape (h0, input_dim)
      encoder.down1.fc1.weight     shape (h1, h0)
      encoder.down2.fc1.weight     shape (embed_dim, h1)
      attention.V.weight           shape (att_hidden, embed_dim)
      encoder.cell_type_head.weight shape (n_cell_types, embed_dim)
      classifier.classifier.weight  shape (n_classes, embed_dim)
    """
    h0        = state["encoder.input_proj.0.weight"].shape[0]
    h1        = state["encoder.down1.fc1.weight"].shape[0]
    embed_dim = state["encoder.down2.fc1.weight"].shape[0]
    att_hidden   = state["attention.V.weight"].shape[0]
    n_cell_types = state["encoder.cell_type_head.weight"].shape[0]
    n_classes    = state["classifier.classifier.weight"].shape[0]

    saved_input = state["encoder.input_proj.0.weight"].shape[1]
    if saved_input != n_input_genes:
        raise ValueError(
            f"Checkpoint input_dim={saved_input} != gene_names len={n_input_genes}. "
            "Wrong --run_dir or wrong gene_names.json?"
        )
    return {
        "h0": h0, "h1": h1, "embed_dim": embed_dim,
        "attention_hidden_dim": att_hidden,
        "n_cell_types": n_cell_types, "n_classes": n_classes,
    }


def _build_model(hps: dict, n_input_genes: int, device) -> CytokineABMIL:
    """Construct an untrained binary CytokineABMIL from inferred HPs."""
    encoder = InstanceEncoder(
        input_dim=n_input_genes,
        embed_dim=hps["embed_dim"],
        n_cell_types=hps["n_cell_types"],
        hidden_dims=(hps["h0"], hps["h1"]),
    )
    attention = AttentionModule(
        embed_dim=hps["embed_dim"],
        attention_hidden_dim=hps["attention_hidden_dim"],
    )
    classifier = BagClassifier(embed_dim=hps["embed_dim"], n_classes=hps["n_classes"])
    model = CytokineABMIL(encoder, attention, classifier, encoder_frozen=True)
    model.to(device)
    return model


def _load_model_from_ckpt(ckpt_path: Path, n_input_genes: int, device) -> CytokineABMIL:
    """Load a binary AB-MIL from a checkpoint, inferring HPs from saved shapes."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hps = _infer_hps_from_state(state, n_input_genes)
    model = _build_model(hps, n_input_genes, device)
    model.load_state_dict(state)
    model.eval()
    # Disable requires_grad for inference; raw_gradient re-enables internally.
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Tube loading
# ---------------------------------------------------------------------------

def _load_tube_X(entry: dict, gene_names: list, device) -> torch.Tensor:
    """Load one pseudo-tube's expression matrix aligned to gene_names.

    Missing genes get zero-padded (same as run_binary_ig_probe.py).
    Returns (N_cells, G) float32 tensor on device.
    """
    adata = anndata.read_h5ad(entry["path"])
    avail = [g for g in gene_names if g in adata.var_names]
    if len(avail) == len(gene_names):
        X = adata[:, gene_names].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)

    X = adata[:, avail].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    idx_map = {g: i for i, g in enumerate(gene_names)}
    full = np.zeros((X.shape[0], len(gene_names)), dtype=np.float32)
    for j, g in enumerate(avail):
        full[:, idx_map[g]] = X[:, j]
    return torch.from_numpy(full).to(device)


# ---------------------------------------------------------------------------
# PBS baseline builder (same pattern as run_binary_ig_probe.py)
# ---------------------------------------------------------------------------

def _build_pbs_baseline(pbs_entries: list, gene_names: list, device) -> torch.Tensor:
    """Per-gene mean expression over PBS tubes. Returns (G,) float32 tensor."""
    means = []
    for entry in pbs_entries:
        X = _load_tube_X(entry, gene_names, device)
        means.append(X.mean(dim=0).cpu().numpy())
    pbs_mean = np.mean(np.stack(means, axis=0), axis=0).astype(np.float32)
    return torch.from_numpy(pbs_mean).to(device)   # (G,)


# ---------------------------------------------------------------------------
# Per-epoch attribution for one stimulus
# ---------------------------------------------------------------------------

def _attr_one_epoch(
    model: CytokineABMIL,
    stim_entries: list,
    gene_names: list,
    pbs_baseline: torch.Tensor,
    device,
    target_class: int = 0,   # BinaryLabel: positive=0
) -> np.ndarray:
    """Compute mean |raw_gradient| over all stimulus training tubes.

    Returns (G,) mean |attribution| averaged over tubes.
    raw_gradient returns (N, G); mean_over_cells -> (G,); abs here.
    """
    G = len(gene_names)
    accum = np.zeros(G, dtype=np.float64)
    n_used = 0
    for entry in stim_entries:
        X = _load_tube_X(entry, gene_names, device)
        baseline = pbs_baseline.unsqueeze(0).expand_as(X).contiguous()
        attr = raw_gradient(model, X, target_class, baseline=baseline)  # (N, G)
        attr_per_gene = mean_over_cells(attr)                           # (G,)
        accum += np.abs(attr_per_gene)
        n_used += 1
    return accum / max(n_used, 1)


# ---------------------------------------------------------------------------
# Main loop for one stimulus
# ---------------------------------------------------------------------------

def _process_stimulus(
    stimulus: str,
    run_dir: Path,
    gene_names: list,
    pbs_baseline: torch.Tensor,
    device,
    seed: int,
    do_ig_final: bool,
) -> list:
    """Return list of row-dicts for gene_attribution_trajectory.parquet."""
    safe = stimulus.replace("/", "_").replace(" ", "_")
    ckpt_dir = run_dir / f"checkpoints_{safe}"

    if not ckpt_dir.exists():
        _log(f"  SKIP {stimulus}: no checkpoint dir at {ckpt_dir}")
        return []

    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpt_files:
        _log(f"  SKIP {stimulus}: no epoch_*.pt files in {ckpt_dir}")
        return []

    def _epoch_of(p: Path) -> int:
        return int(p.stem.replace("epoch_", ""))

    ckpt_files = sorted(ckpt_files, key=_epoch_of)
    epochs = [_epoch_of(f) for f in ckpt_files]
    _log(f"  {stimulus}: {len(ckpt_files)} checkpoints  "
         f"(epochs {epochs[0]}..{epochs[-1]})")

    # Load train manifest for this stimulus
    train_m_path = run_dir / f"manifest_train_{safe}.json"
    if not train_m_path.exists():
        _log(f"  SKIP {stimulus}: manifest_train_{safe}.json not found")
        return []
    with open(train_m_path) as fh:
        all_entries = json.load(fh)

    # Stimulus-only entries (exclude PBS from the attribution average)
    stim_entries = [e for e in all_entries if e["cytokine"] == stimulus]
    if not stim_entries:
        _log(f"  SKIP {stimulus}: no non-PBS entries in manifest_train_{safe}.json")
        return []
    _log(f"  {stimulus}: {len(stim_entries)} training tubes")

    n_genes = len(gene_names)
    rows = []

    for ckpt_path, epoch in zip(ckpt_files, epochs):
        model = _load_model_from_ckpt(ckpt_path, n_genes, device)
        attr_vec = _attr_one_epoch(
            model, stim_entries, gene_names, pbs_baseline, device,
        )
        for g_idx, g in enumerate(gene_names):
            rows.append({
                "gene":     g,
                "epoch":    epoch,
                "stimulus": stimulus,
                "seed":     seed,
                "attr":     float(attr_vec[g_idx]),
            })
        del model

    _log(f"  {stimulus}: {len(rows)} rows across {len(epochs)} epochs")

    # Optional: IG at final epoch (sanity / robustness column)
    if do_ig_final and ckpt_files:
        _log(f"  {stimulus}: computing IG at final epoch {epochs[-1]}...")
        final_model = _load_model_from_ckpt(ckpt_files[-1], n_genes, device)
        ig_accum = np.zeros(n_genes, dtype=np.float64)
        n_used = 0
        for entry in stim_entries:
            X = _load_tube_X(entry, gene_names, device)
            baseline = pbs_baseline.unsqueeze(0).expand_as(X).contiguous()
            ig = integrated_gradients(final_model, X, 0, baseline, n_steps=20)  # (N,G)
            ig_accum += np.abs(mean_over_cells(ig))
            n_used += 1
        ig_mean = ig_accum / max(n_used, 1)
        ig_dict = {g: float(ig_mean[i]) for i, g in enumerate(gene_names)}
        ig_out_path = run_dir / f"gene_ig_final_{safe}.json"
        with open(ig_out_path, "w") as fh:
            json.dump(ig_dict, fh)
        _log(f"  Saved IG final: {ig_out_path.name}")
        del final_model

    return rows


# ---------------------------------------------------------------------------
# Seed inference from run dir name
# ---------------------------------------------------------------------------

def _infer_seed(run_dir: Path, cli_seed) -> int:
    if cli_seed is not None:
        return cli_seed
    # Try to parse seed_<N> from run_dir name
    name = run_dir.name
    if name.startswith("seed_"):
        try:
            return int(name[5:])
        except ValueError:
            pass
    _log(f"  WARNING: could not infer seed from '{name}'; defaulting to 0. "
         "Pass --seed explicitly.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    seed = _infer_seed(run_dir, args.seed)

    _log("=" * 60)
    _log("Extract gene attribution trajectory")
    _log("=" * 60)
    _log(f"  run_dir: {run_dir}")
    _log(f"  seed:    {seed}")
    _log(f"  device:  {device}")
    _log(f"  ig_final: {args.ig_final}")

    # Gene list written by the trainer
    gene_names_path = run_dir / "gene_names.json"
    if not gene_names_path.exists():
        _log(f"FATAL: gene_names.json not found in {run_dir}")
        sys.exit(1)
    with open(gene_names_path) as fh:
        gene_names = json.load(fh)
    n_genes = len(gene_names)
    _log(f"  Genes: {n_genes}")

    # Full manifest for PBS baseline (load from any train manifest available)
    # We need PBS entries; scan all train manifests to collect them.
    pbs_entries = []
    for m_path in run_dir.glob("manifest_train_*.json"):
        with open(m_path) as fh:
            entries = json.load(fh)
        pbs_entries.extend([e for e in entries if e["cytokine"] == "PBS"])
    # Deduplicate by path
    seen = set()
    unique_pbs = []
    for e in pbs_entries:
        if e["path"] not in seen:
            seen.add(e["path"])
            unique_pbs.append(e)
    pbs_entries = unique_pbs

    if not pbs_entries:
        _log("FATAL: no PBS entries found in any manifest_train_*.json")
        sys.exit(1)
    _log(f"  PBS tubes for baseline: {len(pbs_entries)}")

    _log("\nBuilding PBS per-gene baseline...")
    pbs_baseline = _build_pbs_baseline(pbs_entries, gene_names, device)  # (G,)
    _log(f"  PBS baseline shape: {tuple(pbs_baseline.shape)}")

    # Discover which stimuli have checkpoints
    ckpt_dirs = sorted(run_dir.glob("checkpoints_*"))
    stimuli_found = [d.name[len("checkpoints_"):] for d in ckpt_dirs if d.is_dir()]
    if not stimuli_found:
        _log(f"FATAL: no checkpoints_* directories in {run_dir}")
        sys.exit(1)
    _log(f"\nStimuli with checkpoints: {stimuli_found}")

    all_rows = []
    for safe_stimulus in stimuli_found:
        # Recover the original stimulus name from the label_encoder file if present
        le_path = run_dir / f"label_encoder_{safe_stimulus}.json"
        if le_path.exists():
            with open(le_path) as fh:
                le_data = json.load(fh)
            stimulus = le_data.get("positive", safe_stimulus)
        else:
            stimulus = safe_stimulus

        _log(f"\n--- Stimulus: {stimulus} (safe: {safe_stimulus}) ---")
        rows = _process_stimulus(
            stimulus=stimulus,
            run_dir=run_dir,
            gene_names=gene_names,
            pbs_baseline=pbs_baseline,
            device=device,
            seed=seed,
            do_ig_final=args.ig_final,
        )
        all_rows.extend(rows)

    if not all_rows:
        _log("WARNING: no rows produced. Check that checkpoints exist and contain "
             "epoch_*.pt files.")
        sys.exit(0)

    df = pd.DataFrame(all_rows)
    # Ensure column order matches the spec consumed by analyze_gene_learning_order.py
    df = df[["gene", "epoch", "stimulus", "seed", "attr"]]

    out_path = run_dir / "gene_attribution_trajectory.parquet"
    df.to_parquet(out_path, index=False)
    _log(f"\nWrote {out_path}")
    _log(f"  Shape: {df.shape}  "
         f"(genes: {df['gene'].nunique()}, "
         f"epochs: {df['epoch'].nunique()}, "
         f"stimuli: {df['stimulus'].nunique()})")


if __name__ == "__main__":
    main()
