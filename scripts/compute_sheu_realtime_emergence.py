"""
Compute per-gene real-time emergence from the raw Sheu 2024 time-course data.

For each gene in the HVG list (500-gene Sheu targeted panel), compute:
  - realtime_emergence : first time point (hours) at which the gene's mean
    expression (normalize_total + log1p) in stimulus-stimulated cells exceeds
    50% of its maximum above its 0hr/PBS baseline. This is the first-crossing
    time in real biological hours, measuring which genes respond early (cascade
    SOURCE) vs late (cascade DOWNSTREAM / autocrine products).
  - log2fc : log2((mean_stim_at_3hr + eps) / (mean_PBS + eps)) at the 3hr
    time point (effect size control for H1).

Dataset structure (CLAUDE.md §2.5):
  - Raw BD Rhapsody count files in --raw_dir (GSE224518_*.gz) + samptag metadata.
  - The `load_sheu_anndata` function in build_pseudotubes_sheu2024.py handles
    the demultiplexing; we reuse it here.
  - Time points available at 3hr train pseudo-donors: 0.25hr, 0.5hr, 1hr, 3hr,
    5hr, 8hr (0hr/Unstim = PBS baseline). Whatever is present is used.

If the raw time-course loading is unavailable (e.g. partial data deposit),
a fallback message is printed and the script exits cleanly with a note about
which time points were found.

Output: <output_dir>/realtime_emergence.csv
  columns: gene, realtime_emergence, log2fc

  realtime_emergence = NaN if the gene never crosses the 50% threshold (stays
  flat across the time course). NaN genes are kept in the CSV so that the
  analyzer can decide how to handle them.

Usage:
  python scripts/compute_sheu_realtime_emergence.py \\
      --stimulus polyIC \\
      --output_dir results/gene_learning_order
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Cluster defaults
# ---------------------------------------------------------------------------

RAW_DIR = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw"
HVG_PATH_DEFAULT = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json"
OUTPUT_BASE = REPO_ROOT / "results" / "gene_learning_order"

# All time points in the Sheu experiment (hours).
# The script uses whatever subset is actually present in the loaded data.
ALL_TIME_POINTS_HR = [0.0, 0.25, 0.5, 1.0, 3.0, 5.0, 8.0, 24.0]

# Column name for time point in the samptag metadata and the AnnData obs.
# build_pseudotubes_sheu2024.py maps "timept" -> "time_point".
TIMEPT_COL = "time_point"
CYTOKINE_COL = "cytokine"

# Fraction of max-above-baseline at which a gene is considered "emerged".
FRAC_THRESHOLD = 0.50

EPS = 1e-6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-gene real-time emergence from raw Sheu time-course.",
    )
    p.add_argument("--raw_dir", default=RAW_DIR,
                   help="Path to raw Sheu 2024 data dir (GSE224518 files).")
    p.add_argument("--stimulus", default="polyIC",
                   help="Stimulus name in the Sheu metadata for which to compute "
                        "emergence. Note: GEO uses 'PIC' internally; the script "
                        "tries both 'polyIC'/'PIC' and 'LPS'/'LPSlo' aliases. "
                        "Default: polyIC")
    p.add_argument("--hvg", default=HVG_PATH_DEFAULT,
                   help="Path to hvg_list.json (500 Sheu panel genes).")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: results/gene_learning_order).")
    p.add_argument("--frac", type=float, default=FRAC_THRESHOLD,
                   help=f"Fraction of max for first-crossing detection "
                        f"(default: {FRAC_THRESHOLD})")
    p.add_argument("--eps", type=float, default=EPS,
                   help="Epsilon for log2fc (default: 1e-6)")
    return p.parse_args()


def _log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Stimulus alias handling
# ---------------------------------------------------------------------------

# The Sheu metadata uses internal GEO stimulus codes that differ from the
# labels used in the pseudo-tube pipeline. Map pipeline names -> GEO names.
_STIMULUS_ALIASES = {
    "polyIC": ["PIC", "polyIC", "poly(I:C)"],
    "LPS":    ["LPS"],
    "LPSlo":  ["LPSlo", "LPS-lo"],
    "TNF":    ["TNF"],
    "CpG":    ["CpG"],
    "IFNb":   ["IFNb", "IFN-b", "IFNbeta"],
    "Pam3CSK4": ["P3CSK", "Pam3CSK4", "P3C"],
}


def _resolve_stimulus_label(stimulus: str, available_labels: set) -> str | None:
    """Return the actual label in available_labels that matches stimulus (any alias)."""
    candidates = _STIMULUS_ALIASES.get(stimulus, [stimulus])
    for alias in candidates:
        if alias in available_labels:
            return alias
    return None


# ---------------------------------------------------------------------------
# Time-point string -> float hours
# ---------------------------------------------------------------------------

def _parse_timept_to_hr(tp_str: str) -> float | None:
    """Convert '3hr', '0.5hr', '0hr', '24hr', 'Unstim' -> float hours or None."""
    tp_str = str(tp_str).strip()
    if tp_str.lower() in ("unstim", "0", "na", "nan"):
        return 0.0
    tp_str = tp_str.lower().replace("hr", "").replace("h", "").strip()
    try:
        return float(tp_str)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Mean expression per gene at each time point
# ---------------------------------------------------------------------------

def _compute_time_series(
    adata,
    stimulus_label: str,
    gene_names: list,
) -> tuple[np.ndarray, list[float]]:
    """Compute mean expression per gene per time point.

    Returns:
        matrix : (n_genes, n_timepoints) float64, PBS-corrected
                 (mean_stim_at_tp - mean_pbs_at_tp; pbs = 0hr mean).
        time_hrs : list of float, same length as axis 1.
    """
    import numpy as np

    obs = adata.obs.copy()
    obs["_time_hr"] = obs[TIMEPT_COL].apply(_parse_timept_to_hr)

    # Identify which stimulus label to use
    stim_mask = obs[CYTOKINE_COL] == stimulus_label
    pbs_mask  = obs[CYTOKINE_COL].isin(["PBS", "Unstim"]) | (obs["_time_hr"] == 0.0)

    # Unique non-NaN time points
    all_tps = sorted(set(obs.loc[stim_mask, "_time_hr"].dropna().tolist()))
    if 0.0 not in all_tps:
        all_tps = [0.0] + all_tps   # include PBS reference point

    _log(f"  Time points found for {stimulus_label}: {all_tps} hr")

    X_full = adata.X
    if hasattr(X_full, "toarray"):
        X_full = X_full.toarray()
    X_full = np.asarray(X_full, dtype=np.float64)

    # Gene index alignment
    adata_genes = list(adata.var_names)
    gene_idx_in_adata = [adata_genes.index(g) if g in adata_genes else None
                         for g in gene_names]

    n_genes = len(gene_names)

    # Per-time-point means for the stimulus
    stim_means = np.zeros((n_genes, len(all_tps)), dtype=np.float64)
    for t_i, tp in enumerate(all_tps):
        if tp == 0.0:
            # 0hr cells = PBS baseline; labelled PBS after relabeling
            tp_mask = pbs_mask.values
        else:
            tp_mask = (obs["_time_hr"] == tp).values & stim_mask.values
        if tp_mask.sum() == 0:
            _log(f"  WARNING: no cells at time point {tp}hr for {stimulus_label}")
            stim_means[:, t_i] = np.nan
            continue
        X_tp = X_full[tp_mask]
        for g_i, adata_i in enumerate(gene_idx_in_adata):
            if adata_i is None:
                stim_means[g_i, t_i] = np.nan
            else:
                stim_means[g_i, t_i] = float(X_tp[:, adata_i].mean())

    # PBS baseline = 0hr values
    pbs_mean_vec = stim_means[:, all_tps.index(0.0)] if 0.0 in all_tps else np.zeros(n_genes)

    # Correct each time point: stim_mean - pbs_mean (above-baseline)
    above_baseline = stim_means - pbs_mean_vec[:, None]

    return above_baseline, all_tps, pbs_mean_vec


# ---------------------------------------------------------------------------
# Emergence epoch (first-crossing in real hours)
# ---------------------------------------------------------------------------

def _realtime_emergence(above_baseline: np.ndarray, time_hrs: list, frac: float) -> np.ndarray:
    """First time point at which above_baseline >= frac * max_above_baseline.

    above_baseline: (n_genes, n_timepoints). Time 0 (pbs) should be ~0.
    Returns (n_genes,) in hours; NaN if gene never crosses threshold.
    """
    n_genes, n_tp = above_baseline.shape
    emergence = np.full(n_genes, np.nan, dtype=np.float64)
    times = np.asarray(time_hrs, dtype=np.float64)
    for g in range(n_genes):
        traj = above_baseline[g]
        max_val = np.nanmax(traj)
        if max_val <= 0 or np.isnan(max_val):
            continue
        threshold = frac * max_val
        for t_i, t in enumerate(times):
            if not np.isnan(traj[t_i]) and traj[t_i] >= threshold:
                emergence[g] = t
                break
    return emergence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_BASE
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("=" * 60)
    _log("Compute real-time gene emergence from Sheu time-course")
    _log("=" * 60)
    _log(f"  Stimulus: {args.stimulus}")
    _log(f"  Raw dir:  {args.raw_dir}")
    _log(f"  HVG:      {args.hvg}")
    _log(f"  Out dir:  {out_dir}")
    _log(f"  frac:     {args.frac}")

    # Load gene names
    import json
    with open(args.hvg) as fh:
        gene_names = json.load(fh)
    _log(f"  Genes:    {len(gene_names)}")

    # Load raw Sheu data via the existing adapter function
    # (builds a full AnnData with obs[time_point, cytokine, pseudo_donor, ...])
    _log("\nLoading raw Sheu AnnData (all time points, BD Rhapsody files)...")
    _log("  This may take a few minutes — loads all GSM count files.")

    # Import the Sheu adapter functions via sys.path so this script can be run
    # from the repo root (python scripts/compute_sheu_realtime_emergence.py).
    # build_pseudotubes_sheu2024.py lives in scripts/ which is already in the
    # Python path when the script is invoked from the repo root; add it explicitly
    # to be safe.
    import importlib.util as _ilu
    _sheu_spec = _ilu.spec_from_file_location(
        "build_pseudotubes_sheu2024",
        str(REPO_ROOT / "scripts" / "build_pseudotubes_sheu2024.py"),
    )
    _sheu_mod = _ilu.module_from_spec(_sheu_spec)
    _sheu_spec.loader.exec_module(_sheu_mod)
    load_sheu_anndata = _sheu_mod.load_sheu_anndata
    relabel_to_pbs    = _sheu_mod.relabel_to_pbs
    import scanpy as sc

    raw_dir = args.raw_dir
    adata = load_sheu_anndata(raw_dir)
    _log(f"  Loaded AnnData: {adata.n_obs} cells x {adata.n_vars} genes")

    # Normalize + log1p (same as the pseudotube builder)
    _log("\nNormalizing (normalize_total + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Relabel Unstim/0hr -> PBS so cytokine column is consistent
    adata = relabel_to_pbs(adata)

    available_labels = set(adata.obs[CYTOKINE_COL].unique())
    _log(f"  Available cytokine labels: {sorted(available_labels)}")
    available_tps = sorted(set(adata.obs[TIMEPT_COL].unique()))
    _log(f"  Available time points:     {available_tps}")

    # Resolve stimulus alias (GEO may use 'PIC' for 'polyIC', etc.)
    resolved_label = _resolve_stimulus_label(args.stimulus, available_labels)
    if resolved_label is None:
        _log(f"FATAL: could not find stimulus '{args.stimulus}' (or any alias) "
             f"in the data. Available: {sorted(available_labels)}")
        sys.exit(1)
    _log(f"\n  Resolved stimulus label: '{resolved_label}'")

    # Subset to the stimulus + PBS (0hr) cells across ALL time points
    # (we need the full time course, not just 3hr)
    relevant_mask = (
        (adata.obs[CYTOKINE_COL] == resolved_label) |
        (adata.obs[CYTOKINE_COL] == "PBS")
    )
    adata_sub = adata[relevant_mask.values].copy()
    _log(f"  Cells for time-series analysis: {adata_sub.n_obs}")

    # Compute per-gene, per-time-point mean expression (PBS-corrected)
    above_baseline, time_hrs, pbs_mean_vec = _compute_time_series(
        adata_sub, resolved_label, gene_names,
    )

    non_zero_tps = [t for t in time_hrs if t > 0.0]
    if not non_zero_tps:
        _log("FATAL: no post-stimulation time points found. "
             "Check that the raw data contains time points > 0hr.")
        sys.exit(1)

    _log(f"\n  Post-stimulation time points used: {non_zero_tps} hr")

    # Real-time emergence
    _log(f"\nComputing per-gene real-time emergence (frac={args.frac})...")
    emergence = _realtime_emergence(above_baseline, time_hrs, frac=args.frac)
    n_emerged = np.sum(~np.isnan(emergence))
    _log(f"  Genes with emergence time: {n_emerged}/{len(gene_names)}")

    # log2FC at 3hr vs PBS
    _log("\nComputing log2FC at 3hr (vs PBS)...")
    if 3.0 in time_hrs:
        t3_idx = time_hrs.index(3.0)
        stim_at_3hr = above_baseline[:, t3_idx] + pbs_mean_vec  # back to absolute
        log2fc = np.log2((stim_at_3hr + args.eps) / (pbs_mean_vec + args.eps))
    else:
        _log("  WARNING: 3hr time point not found; using final time point for log2FC.")
        final_idx = -1
        stim_at_final = above_baseline[:, final_idx] + pbs_mean_vec
        log2fc = np.log2((stim_at_final + args.eps) / (pbs_mean_vec + args.eps))

    # Build output dataframe
    df = pd.DataFrame({
        "gene":               gene_names,
        "realtime_emergence": emergence.tolist(),
        "log2fc":             log2fc.tolist(),
    })

    out_path = out_dir / "realtime_emergence.csv"
    df.to_csv(out_path, index=False)
    _log(f"\nWrote {out_path}")
    _log(f"  Rows: {len(df)}")
    _log(f"  Emerged genes:    {(~df['realtime_emergence'].isna()).sum()}")
    _log(f"  log2FC mean:      {df['log2fc'].mean():.3f}")
    _log(f"  log2FC max:       {df['log2fc'].max():.3f}")

    # Quick sanity: top-5 genes by emergence
    emerged = df.dropna(subset=["realtime_emergence"]).sort_values("realtime_emergence")
    _log(f"\n  Top-5 earliest-emerging genes:")
    for _, row in emerged.head(5).iterrows():
        _log(f"    {row['gene']:<20s}  emerge={row['realtime_emergence']:.2f}hr  "
             f"log2fc={row['log2fc']:.3f}")


if __name__ == "__main__":
    main()
