"""
Probe B (hybrid step 1) — Binary-MIL IG attribution for per-cytokine gene-set
discovery.

Loads pre-trained binary AB-MIL models from a previous oesinghaus_binary run
(produced by scripts/train_oesinghaus_binary.py) and computes integrated
gradients of logit[positive_class=0] w.r.t. input genes for each (target
cytokine) vs PBS.

Unlike Probe A (which only attributed for marker genes and used a multi-class
model where shared pathways get suppressed), this probe:
  * uses BINARY models (target vs PBS — no class competition)
  * attributes to ALL HVGs (not just marker subset)
  * outputs per-cytokine top-N genes by IG → candidate S_X^binary

Outputs:
  <out_dir>/binary_ig.parquet            — long-format (cytokine, gene, ig, rank_ig)
  <out_dir>/binary_top_genes_summary.md  — per-cytokine top-30 genes + key markers
  <out_dir>/binary_marker_hits.csv       — per-marker, where it ranks per cytokine

Usage:
  python scripts/run_binary_ig_probe.py \\
      --binary_run_dir results/oesinghaus_binary/run_20260412_114413_pid3728447 \\
      --output_dir results/gene_dynamics_phase0/binary_ig

CPU-only. ~5-10 min per cytokine = ~40-80 min total for 8 cytokines.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import anndata
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.label_encoder import BinaryLabel  # noqa: E402
from cytokine_mil.models.attention import AttentionModule  # noqa: E402
from cytokine_mil.models.bag_classifier import BagClassifier  # noqa: E402
from cytokine_mil.models.cytokine_abmil import CytokineABMIL  # noqa: E402
from cytokine_mil.models.instance_encoder import InstanceEncoder  # noqa: E402


# ----------------------------------------------------------------------------
# HPs of the binary models (must match train_oesinghaus_binary.py)
# ----------------------------------------------------------------------------

BINARY_EMBED_DIM = 32
BINARY_HIDDEN_DIMS = (128, 64)
BINARY_ATTENTION_HIDDEN_DIM = 16

# The cytokines we want to probe — must overlap with trained binary models in
# the run dir. Will skip any that don't have a model_<safe_target>.pt file.
TARGET_CYTOKINES_DEFAULT = [
    "IFN-beta",
    "IL-1-beta",
    "TNF-alpha",
    "IL-6",
    "IL-2",
    "IL-10",
    "IL-12",
    "TGF-beta1",  # non-canonical SMAD-driven control
]

# Marker panel re-used from Probe A so we can directly compare hits/misses.
MARKER_PANEL: Dict[str, Dict[str, List[str]]] = {
    "type_I_IFN_ISGs": {
        "genes": ["ISG15", "IFIT2", "IFIT3", "RSAD2"],  # MX1/MX2 not in HVG list
        "expected_winners": ["IFN-beta"],
    },
    "IFN_gamma_STAT1": {
        "genes": ["CXCL9", "CXCL10", "CXCL11", "GBP1", "GBP5", "STAT1"],
        "expected_winners": ["IFN-gamma"],  # not in default pool
    },
    "NFkB_direct": {
        "genes": ["TNF", "IL1B", "CXCL8", "BIRC3", "CCL3", "CCL4"],
        "expected_winners": ["IL-1-beta", "TNF-alpha"],
    },
    "IL2_STAT5": {
        "genes": ["IL2RA"],
        "expected_winners": ["IL-2"],
    },
    "STAT3_direct": {
        "genes": ["SOCS2"],
        "expected_winners": ["IL-6", "IL-10"],
    },
}

HVG_PATH_DEFAULT = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
)

MANIFEST_PATH_DEFAULT = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--binary_run_dir", required=True,
        help="Path to a previous oesinghaus_binary run dir containing "
             "model_<cytokine>.pt files and label_encoder_<cytokine>.json",
    )
    p.add_argument("--manifest_path", default=MANIFEST_PATH_DEFAULT)
    p.add_argument("--hvg_path", default=HVG_PATH_DEFAULT)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--targets", nargs="+", default=TARGET_CYTOKINES_DEFAULT,
        help="Cytokine targets to probe (must have trained models in run_dir)",
    )
    p.add_argument("--top_n", type=int, default=50,
                   help="Top-N genes per cytokine to consider as S_X^binary")
    p.add_argument("--n_ig_steps", type=int, default=20)
    p.add_argument("--max_tubes_per_cytokine", type=int, default=10)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

def _safe_filename(cytokine: str) -> str:
    return cytokine.replace("/", "_")


def _build_binary_mil(n_input_genes: int, n_cell_types: int, device) -> CytokineABMIL:
    """Construct an untrained binary MIL with the HPs of train_oesinghaus_binary.py."""
    encoder = InstanceEncoder(
        input_dim=n_input_genes,
        embed_dim=BINARY_EMBED_DIM,
        n_cell_types=n_cell_types,
        hidden_dims=BINARY_HIDDEN_DIMS,
    )
    attention = AttentionModule(
        embed_dim=BINARY_EMBED_DIM,
        attention_hidden_dim=BINARY_ATTENTION_HIDDEN_DIM,
    )
    classifier = BagClassifier(embed_dim=BINARY_EMBED_DIM, n_classes=2)
    model = CytokineABMIL(encoder, attention, classifier, encoder_frozen=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_binary_model(model_pt_path: Path, n_input_genes: int, device) -> CytokineABMIL:
    """Load a binary AB-MIL state dict into a freshly constructed model."""
    state = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    n_cell_types = state["encoder.cell_type_head.weight"].shape[0]
    model = _build_binary_mil(n_input_genes, n_cell_types, device)
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ----------------------------------------------------------------------------
# Tube loading helpers (same as Probe A)
# ----------------------------------------------------------------------------

def _load_tube_X(entry: dict, gene_names: List[str]) -> torch.Tensor:
    """Load X aligned to gene_names. Missing genes get zero-padded."""
    adata = anndata.read_h5ad(entry["path"])
    avail = [g for g in gene_names if g in adata.var_names]
    if len(avail) == len(gene_names):
        adata_sub = adata[:, gene_names]
        X = adata_sub.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return torch.from_numpy(np.asarray(X, dtype=np.float32))
    adata_sub = adata[:, avail]
    X = adata_sub.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    idx_map = {g: i for i, g in enumerate(gene_names)}
    full = np.zeros((X.shape[0], len(gene_names)), dtype=np.float32)
    for j, g in enumerate(avail):
        full[:, idx_map[g]] = X[:, j]
    return torch.from_numpy(full)


# ----------------------------------------------------------------------------
# Integrated Gradients (same shape as Probe A)
# ----------------------------------------------------------------------------

def integrated_gradients(
    model: torch.nn.Module,
    X: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor,
    n_steps: int = 20,
) -> torch.Tensor:
    """IG of logit[target_class] w.r.t. X. Returns (N, G) per-cell, per-gene."""
    delta = X - baseline
    alphas = torch.linspace(
        0.5 / n_steps, 1.0 - 0.5 / n_steps, n_steps, device=X.device,
    )
    grads_accum = torch.zeros_like(X)
    for alpha in alphas:
        x_interp = (baseline + alpha * delta).detach().clone().requires_grad_(True)
        logits, _, _ = model(x_interp)
        loss = logits[target_class]
        grad = torch.autograd.grad(loss, x_interp, create_graph=False)[0]
        grads_accum = grads_accum + grad
    return delta * (grads_accum / n_steps)


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def main():
    args = _parse_args()
    binary_run = Path(args.binary_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- HVG list ----
    with open(args.hvg_path) as f:
        gene_names: List[str] = json.load(f)
    n_genes = len(gene_names)
    _log(f"HVG count: {n_genes}")

    # ---- Full manifest (we need real tube paths for each cytokine) ----
    with open(args.manifest_path) as f:
        manifest = json.load(f)
    by_cyt: Dict[str, List[dict]] = defaultdict(list)
    for e in manifest:
        by_cyt[e["cytokine"]].append(e)
    _log(f"Cytokines in manifest: {len(by_cyt)}")

    pbs_entries = by_cyt.get("PBS", [])
    if not pbs_entries:
        _log("ABORT: no PBS tubes in manifest.")
        sys.exit(1)
    pbs_cap = min(len(pbs_entries), args.max_tubes_per_cytokine)
    _log(f"PBS tubes used: {pbs_cap}/{len(pbs_entries)}")

    # ---- PBS baseline (per-gene mean across PBS tubes) ----
    pbs_means = []
    for e in pbs_entries[:pbs_cap]:
        X = _load_tube_X(e, gene_names)
        pbs_means.append(X.mean(dim=0).numpy())
    pbs_per_gene = torch.from_numpy(
        np.mean(np.stack(pbs_means, axis=0), axis=0).astype(np.float32),
    )  # (G,)
    _log(f"PBS baseline shape: {tuple(pbs_per_gene.shape)}")

    # ---- For each target cytokine: load model and compute IG ----
    long_records: List[dict] = []  # (cytokine, gene, ig, rank_ig)
    per_cyt_top: Dict[str, pd.DataFrame] = {}

    for cyt in args.targets:
        safe = _safe_filename(cyt)
        model_pt = binary_run / f"model_{safe}.pt"
        if not model_pt.exists():
            _log(f"  SKIP {cyt}: no model file at {model_pt}")
            continue
        if cyt not in by_cyt:
            _log(f"  SKIP {cyt}: not in manifest")
            continue

        _log(f"\n>>> {cyt} <<<")
        _log(f"  Loading {model_pt.name}")
        model = _load_binary_model(model_pt, n_genes, device)
        target_class = 0  # BinaryLabel: positive=0, negative=1

        cyt_entries = by_cyt[cyt][: args.max_tubes_per_cytokine]
        _log(f"  Tubes: {len(cyt_entries)}")

        ig_accum = np.zeros(n_genes, dtype=np.float64)
        expr_accum = np.zeros(n_genes, dtype=np.float64)
        n_used = 0

        for entry in cyt_entries:
            X = _load_tube_X(entry, gene_names).to(device)
            baseline = pbs_per_gene.to(device).unsqueeze(0).expand_as(X).contiguous()
            ig = integrated_gradients(
                model, X, target_class, baseline, n_steps=args.n_ig_steps,
            )  # (N, G)
            ig_per_gene = ig.mean(dim=0).detach().cpu().numpy()
            expr_per_gene = X.mean(dim=0).detach().cpu().numpy()
            ig_accum += ig_per_gene
            expr_accum += expr_per_gene
            n_used += 1

        ig_mean = ig_accum / max(n_used, 1)
        expr_mean = expr_accum / max(n_used, 1)
        _log(f"  IG range across all HVGs: [{ig_mean.min():+.4f}, {ig_mean.max():+.4f}]")

        # Rank by IG descending
        order = np.argsort(-ig_mean)
        rank_ig = np.empty_like(order)
        rank_ig[order] = np.arange(len(order))

        for g_idx, g in enumerate(gene_names):
            long_records.append({
                "cytokine": cyt,
                "gene": g,
                "ig": float(ig_mean[g_idx]),
                "mean_expression": float(expr_mean[g_idx]),
                "rank_ig": int(rank_ig[g_idx]),  # 0 = highest IG
            })

        # Stash top-N for this cytokine
        top_df = pd.DataFrame({
            "rank": np.arange(min(args.top_n, len(order))),
            "gene": [gene_names[i] for i in order[: args.top_n]],
            "ig": [float(ig_mean[i]) for i in order[: args.top_n]],
            "mean_expression": [float(expr_mean[i]) for i in order[: args.top_n]],
        })
        per_cyt_top[cyt] = top_df

        del model

    # ---- Write parquet ----
    long_df = pd.DataFrame(long_records)
    long_df.to_parquet(output_dir / "binary_ig.parquet")
    _log(f"\nWrote {output_dir / 'binary_ig.parquet'}")

    # ---- Marker hits ----
    # For each marker gene, where does it rank in each cytokine's IG?
    marker_rows = []
    for pathway, info in MARKER_PANEL.items():
        for g in info["genes"]:
            row = {"marker_gene": g, "pathway": pathway,
                   "expected_winners": ", ".join(info["expected_winners"])}
            for cyt in per_cyt_top:
                # rank within all HVGs (not just top-N)
                g_rank = long_df.query("cytokine == @cyt and gene == @g")["rank_ig"]
                row[cyt] = int(g_rank.iloc[0]) if len(g_rank) else None
            marker_rows.append(row)
    marker_df = pd.DataFrame(marker_rows)
    marker_df.to_csv(output_dir / "binary_marker_hits.csv", index=False)
    _log(f"Wrote {output_dir / 'binary_marker_hits.csv'}")

    # ---- Human-readable markdown summary ----
    lines = ["# Binary-MIL IG probe — per-cytokine top genes", "",
             f"**Binary run dir:** `{binary_run}`",
             f"**HVGs:** {n_genes}  |  **Top N reported:** {args.top_n}",
             f"**IG steps:** {args.n_ig_steps}  |  **Tubes per cytokine cap:** "
             f"{args.max_tubes_per_cytokine}",
             "",
             "## Per-cytokine top-30 genes",
             ""]
    for cyt, top_df in per_cyt_top.items():
        lines.append(f"### {cyt}")
        lines.append("")
        head = top_df.head(30)
        lines.append("| rank | gene | IG | mean_expression |")
        lines.append("|---:|---|---:|---:|")
        for _, row in head.iterrows():
            lines.append(
                f"| {int(row['rank'])} | {row['gene']} | "
                f"{row['ig']:+.4f} | {row['mean_expression']:.3f} |"
            )
        lines.append("")

    lines.append("## Marker-gene IG ranks per cytokine")
    lines.append("")
    lines.append("Rank 0 = highest IG. A direct-inducer marker should rank "
                 "low (< 50) under its biological inducer cytokine.")
    lines.append("")
    cyts_present = [c for c in per_cyt_top.keys()]
    header = "| marker | pathway | expected | " + " | ".join(cyts_present) + " |"
    sep = "|---|---|---|" + "|".join(["---:"] * len(cyts_present)) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in marker_df.iterrows():
        cells = [
            row["marker_gene"], row["pathway"], row["expected_winners"]
        ] + [str(row.get(c, "NA")) for c in cyts_present]
        lines.append("| " + " | ".join(cells) + " |")

    (output_dir / "binary_top_genes_summary.md").write_text("\n".join(lines) + "\n")
    _log(f"Wrote {output_dir / 'binary_top_genes_summary.md'}")

    _log("\nDONE.")


if __name__ == "__main__":
    main()
