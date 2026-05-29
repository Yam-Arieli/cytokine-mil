"""
Probe A — Static-IG routing sanity check on a trained Oesinghaus Stage 2 model.

Tests whether multi-class integrated-gradient attribution routes 20 well-known
marker genes to their biological direct-inducer cytokines.

For each marker gene g and each cytokine class X:
  IG_g(X) = avg over X-tubes of [ avg over cells of IG_per_cell(x_g, target=X) ]
with the per-gene baseline x_PBS,g = mean expression of g across PBS tubes.

A marker "passes" if its biological direct inducer cytokine ranks in the top-3
of the IG_g(X) distribution across all cytokines.

Pre-registered pass criteria (Phase 0 falsifier — kills the per-epoch dynamics
pipeline if static IG already fails):

  (1) >= 14/20 markers: biological direct inducer in top-3 cytokine ranking
  (2) median over markers of (winner_IG - runner_up_IG)/|winner_IG| >= 0.20
  (3) median over markers of Pearson rho(IG_g(X), mean_expression_g(X)) < 0.70
      (the "highly expressed genes win" confound)

Outputs (under --output_dir):
  static_ig.parquet       — long-format (cytokine, marker_gene) -> IG + mean_expression
  ranking_summary.csv     — per-marker top-3, magnitude gap, expr correlation, pass flag
  verdict.md              — human-readable pass/fail per criterion

Usage:
  python scripts/run_static_ig_probe.py \\
      --run_dir cytokine-mil/results/oesinghaus_full_v2/seed_42 \\
      --output_dir cytokine-mil/results/gene_dynamics_phase0/static_ig_seed42

This script is CPU-only by default. ~10-30 min wall-clock on cluster CPU node
with --max_tubes_per_cytokine=10 (the default).
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

from cytokine_mil.data.label_encoder import CytokineLabel  # noqa: E402
from cytokine_mil.experiment_setup import build_encoder, build_mil_model  # noqa: E402


# ----------------------------------------------------------------------------
# Marker panel — 20 genes, 5 known signaling pathways.
# "expected_winners" lists the cytokines whose direct/canonical signaling
# induces each pathway. A marker passes if ANY listed cytokine appears in the
# top-3 of its IG ranking across all cytokines.
# ----------------------------------------------------------------------------

MARKER_PANEL: Dict[str, Dict[str, List[str]]] = {
    "type_I_IFN_ISGs": {
        "genes": ["ISG15", "MX1", "MX2", "IFIT2", "IFIT3", "RSAD2"],
        "expected_winners": [
            "IFN-alpha1", "IFN-beta", "IFN-omega", "IFN-epsilon",
            "IFN-lambda1", "IFN-lambda2", "IFN-lambda3",
        ],
    },
    "IFN_gamma_STAT1": {
        "genes": ["CXCL9", "CXCL10", "CXCL11", "GBP1", "GBP5", "STAT1"],
        "expected_winners": ["IFN-gamma"],
    },
    "NFkB_direct": {
        "genes": ["TNF", "IL1B", "CXCL8", "BIRC3", "CCL3", "CCL4"],
        "expected_winners": ["IL-1-alpha", "IL-1-beta", "TNF-alpha"],
    },
    "IL2_STAT5": {
        "genes": ["IL2RA"],
        "expected_winners": ["IL-2"],
    },
    "STAT3_direct": {
        "genes": ["SOCS2"],
        "expected_winners": ["IL-6", "IL-11", "OSM", "LIF", "IL-10"],
    },
}

HVG_PATH_DEFAULT = (
    "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir", required=True,
        help="Run dir containing model_stage2.pt, encoder_stage1.pt, "
             "label_encoder.json, manifest_train.json",
    )
    p.add_argument("--hvg_path", default=HVG_PATH_DEFAULT)
    p.add_argument(
        "--manifest_name", default="manifest_train.json",
        help="Manifest under --run_dir to source cytokine tubes from",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--n_ig_steps", type=int, default=20,
        help="Riemann-sum steps for IG approximation",
    )
    p.add_argument(
        "--max_tubes_per_cytokine", type=int, default=10,
        help="Cap on tubes per cytokine for compute (still covers all train donors)",
    )
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------

def _load_label_encoder(run_dir: Path) -> CytokineLabel:
    return CytokineLabel.load(str(run_dir / "label_encoder.json"))


def _load_model(
    run_dir: Path, n_input_genes: int, n_classes: int, device: torch.device,
):
    state = torch.load(
        run_dir / "model_stage2.pt", map_location="cpu", weights_only=False,
    )
    n_cell_types = state["encoder.cell_type_head.weight"].shape[0]
    encoder = build_encoder(n_input_genes, n_cell_types=n_cell_types, embed_dim=128)
    model = build_mil_model(
        encoder,
        embed_dim=128,
        attention_hidden_dim=64,
        n_classes=n_classes,
        encoder_frozen=True,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _load_tube_X(entry: dict, gene_names: List[str]) -> torch.Tensor:
    """Load X aligned to gene_names. Missing genes get zero-padded columns."""
    adata = anndata.read_h5ad(entry["path"])
    avail = [g for g in gene_names if g in adata.var_names]

    if len(avail) == len(gene_names):
        adata_sub = adata[:, gene_names]
        X = adata_sub.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        return torch.from_numpy(X)

    # Pad missing columns with zeros
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
# Integrated Gradients
# ----------------------------------------------------------------------------

def integrated_gradients(
    model: torch.nn.Module,
    X: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor,
    n_steps: int = 20,
) -> torch.Tensor:
    """
    IG of logit[target_class] w.r.t. X, with given baseline.

    X, baseline: (N, G) tensors on the same device.
    Returns: (N, G) per-cell, per-gene IG.

    Implementation: Riemann midpoint rule. For each alpha in (0, 1):
        x_alpha = baseline + alpha * (X - baseline)
        grad_alpha = d logit_target / d x_alpha
    Then IG = (X - baseline) * mean_alpha[grad_alpha]
    """
    delta = X - baseline  # (N, G)
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
    grads_avg = grads_accum / n_steps
    return delta * grads_avg


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- Gene names ----
    with open(args.hvg_path) as f:
        gene_names: List[str] = json.load(f)
    _log(f"HVG count: {len(gene_names)}")

    # ---- Marker gene index map ----
    marker_to_idx: Dict[str, int] = {}
    missing_markers: List[Tuple[str, str]] = []
    for pathway, info in MARKER_PANEL.items():
        for g in info["genes"]:
            if g in gene_names:
                marker_to_idx[g] = gene_names.index(g)
            else:
                missing_markers.append((pathway, g))
    _log(f"Markers found in HVG list: {len(marker_to_idx)}")
    if missing_markers:
        _log(f"Missing markers (NOT in HVG list): {missing_markers}")
    if not marker_to_idx:
        _log("ABORT: no markers found in HVG list.")
        sys.exit(1)

    # ---- Label encoder + model ----
    label_enc = _load_label_encoder(run_dir)
    n_classes = label_enc.n_classes()
    _log(f"n_classes: {n_classes}")
    cytokine_names = label_enc.cytokines  # sorted by index
    _log(f"Cytokines in label encoder: {len(cytokine_names)}")

    model = _load_model(
        run_dir, n_input_genes=len(gene_names),
        n_classes=n_classes, device=device,
    )
    # We never need gradients w.r.t. model parameters.
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- Manifest ----
    with open(run_dir / args.manifest_name) as f:
        entries = json.load(f)
    by_cyt: Dict[str, List[dict]] = defaultdict(list)
    for e in entries:
        by_cyt[e["cytokine"]].append(e)
    _log(f"Cytokines in manifest: {len(by_cyt)}")

    # ---- PBS baseline (per-gene mean across PBS tubes) ----
    pbs_entries = by_cyt.get("PBS", [])
    if not pbs_entries:
        _log("ABORT: no PBS tubes in manifest.")
        sys.exit(1)
    pbs_cap = min(len(pbs_entries), args.max_tubes_per_cytokine)
    _log(f"PBS tubes (used): {pbs_cap} / {len(pbs_entries)}")
    pbs_gene_means: List[np.ndarray] = []
    for e in pbs_entries[:pbs_cap]:
        X = _load_tube_X(e, gene_names)
        pbs_gene_means.append(X.mean(dim=0).numpy())
    pbs_per_gene = torch.from_numpy(
        np.mean(np.stack(pbs_gene_means, axis=0), axis=0).astype(np.float32),
    )  # (G,)

    # ---- Per-cytokine IG and mean-expression accumulation ----
    marker_genes: List[str] = list(marker_to_idx.keys())
    marker_idxs = np.array([marker_to_idx[g] for g in marker_genes], dtype=np.int64)

    ig_matrix = np.full(
        (len(cytokine_names), len(marker_genes)), np.nan, dtype=np.float64,
    )
    mean_expr_matrix = np.full(
        (len(cytokine_names), len(marker_genes)), np.nan, dtype=np.float64,
    )

    for cyt_idx, cyt_name in enumerate(cytokine_names):
        if cyt_name == "PBS":
            continue
        cyt_entries = by_cyt.get(cyt_name, [])
        if not cyt_entries:
            _log(f"  {cyt_name:<22s} NO TUBES")
            continue
        cyt_entries = cyt_entries[: args.max_tubes_per_cytokine]
        target_class = label_enc.encode(cyt_name)

        ig_accum = np.zeros(len(marker_genes), dtype=np.float64)
        expr_accum = np.zeros(len(marker_genes), dtype=np.float64)
        n_used = 0

        for entry in cyt_entries:
            X = _load_tube_X(entry, gene_names).to(device)
            baseline = pbs_per_gene.to(device).unsqueeze(0).expand_as(X).contiguous()
            ig = integrated_gradients(
                model, X, target_class, baseline, n_steps=args.n_ig_steps,
            )  # (N, G)
            ig_per_gene = ig.mean(dim=0).detach().cpu().numpy()
            expr_per_gene = X.mean(dim=0).detach().cpu().numpy()
            ig_accum += ig_per_gene[marker_idxs]
            expr_accum += expr_per_gene[marker_idxs]
            n_used += 1

        if n_used > 0:
            ig_matrix[cyt_idx, :] = ig_accum / n_used
            mean_expr_matrix[cyt_idx, :] = expr_accum / n_used
        _log(
            f"  {cyt_name:<22s} tubes={n_used:>2d}  "
            f"IG[min,max]=[{ig_matrix[cyt_idx, :].min():+.4f}, "
            f"{ig_matrix[cyt_idx, :].max():+.4f}]"
        )

    # ---- Long-format dump ----
    long_records = []
    for cyt_idx, cyt_name in enumerate(cytokine_names):
        for m_idx, g in enumerate(marker_genes):
            long_records.append({
                "cytokine": cyt_name,
                "marker_gene": g,
                "ig": ig_matrix[cyt_idx, m_idx],
                "mean_expression": mean_expr_matrix[cyt_idx, m_idx],
            })
    long_df = pd.DataFrame(long_records)
    long_df.to_parquet(output_dir / "static_ig.parquet")
    _log(f"\nWrote {output_dir / 'static_ig.parquet'}")

    # ---- Per-marker top-3 + gap + IG-vs-expression correlation ----
    summary_rows = []
    n_correct_top3 = 0
    n_evaluable = 0
    gaps: List[float] = []
    rhos: List[float] = []

    for m_idx, g in enumerate(marker_genes):
        pathway = next(
            (p for p, info in MARKER_PANEL.items() if g in info["genes"]), None,
        )
        expected = (
            MARKER_PANEL[pathway]["expected_winners"] if pathway is not None else []
        )

        ig_col = ig_matrix[:, m_idx]
        valid = ~np.isnan(ig_col)
        ig_v = ig_col[valid]
        names_v = [cytokine_names[i] for i in range(len(cytokine_names)) if valid[i]]
        order = np.argsort(-ig_v)
        ranked_names = [names_v[i] for i in order]
        ranked_ig = ig_v[order]

        top3 = ranked_names[:3]
        winner = ranked_names[0] if ranked_names else None
        winner_ig = float(ranked_ig[0]) if len(ranked_ig) > 0 else float("nan")
        runner_up_ig = float(ranked_ig[1]) if len(ranked_ig) > 1 else float("nan")
        if abs(winner_ig) > 0:
            gap = (winner_ig - runner_up_ig) / abs(winner_ig)
        else:
            gap = float("nan")
        gaps.append(gap)

        in_top3 = any(w in top3 for w in expected)
        n_evaluable += 1
        if in_top3:
            n_correct_top3 += 1

        expr_col = mean_expr_matrix[:, m_idx]
        both_valid = (~np.isnan(ig_col)) & (~np.isnan(expr_col))
        if both_valid.sum() >= 3:
            r = float(np.corrcoef(ig_col[both_valid], expr_col[both_valid])[0, 1])
        else:
            r = float("nan")
        rhos.append(r)

        summary_rows.append({
            "marker_gene": g,
            "pathway": pathway,
            "expected_winners": ", ".join(expected),
            "winner": winner,
            "top3": ", ".join(top3),
            "winner_ig": winner_ig,
            "runner_up_ig": runner_up_ig,
            "magnitude_gap": gap,
            "correct_in_top3": in_top3,
            "rho_ig_vs_expression": r,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "ranking_summary.csv", index=False)
    _log(f"Wrote {output_dir / 'ranking_summary.csv'}")

    # ---- Verdict ----
    n_pass_target = max(1, int(round(0.70 * len(marker_genes))))
    median_gap = float(np.nanmedian(gaps))
    median_rho = float(np.nanmedian(rhos))

    crit_top3 = n_correct_top3 >= n_pass_target
    crit_gap = (not np.isnan(median_gap)) and (median_gap >= 0.20)
    crit_rho = (not np.isnan(median_rho)) and (median_rho < 0.70)
    overall = "PASS" if (crit_top3 and crit_gap and crit_rho) else "FAIL"

    lines = [
        "# Probe A — Static-IG routing sanity check",
        "",
        f"**Run dir:** `{run_dir}`",
        f"**Manifest:** `{args.manifest_name}`",
        f"**IG steps:** {args.n_ig_steps}",
        f"**Tubes per cytokine cap:** {args.max_tubes_per_cytokine}",
        "",
        f"## Overall verdict: **{overall}**",
        "",
        "## Pre-registered pass criteria",
        "",
        "| Criterion | Threshold | Observed | Pass? |",
        "|---|---|---|---|",
        f"| Markers with biological inducer in top-3 | ≥ {n_pass_target} / "
        f"{len(marker_genes)} | {n_correct_top3} / {n_evaluable} | "
        f"{'PASS' if crit_top3 else 'FAIL'} |",
        f"| Median magnitude gap (winner − runner-up)/|winner| | ≥ 0.20 | "
        f"{median_gap:.3f} | {'PASS' if crit_gap else 'FAIL'} |",
        f"| Median ρ(IG, mean expression) across markers | < 0.70 | "
        f"{median_rho:.3f} | {'PASS' if crit_rho else 'FAIL'} |",
        "",
        "## Per-marker results",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Notes",
        "",
    ]
    if missing_markers:
        lines.append(
            f"- {len(missing_markers)} markers were NOT in the HVG list and "
            f"were dropped: {missing_markers}"
        )
    else:
        lines.append("- All marker genes were present in the HVG list.")
    lines.extend([
        "- IG baseline = per-gene mean expression across PBS tubes (pooled "
        "across cells and tubes).",
        "- Pooling across cell types (Option B). Per-cell IG is averaged "
        "within a tube; per-tube IG is averaged across X-tubes.",
    ])

    (output_dir / "verdict.md").write_text("\n".join(lines) + "\n")
    _log(f"Wrote {output_dir / 'verdict.md'}")
    _log("")
    _log(f"OVERALL: {overall}")
    _log(f"  top-3 hit rate:    {n_correct_top3} / {n_evaluable} "
         f"(need ≥ {n_pass_target})")
    _log(f"  median gap:        {median_gap:.3f}    (need ≥ 0.20)")
    _log(f"  median rho(IG,X):  {median_rho:.3f}    (need < 0.70)")


if __name__ == "__main__":
    main()
