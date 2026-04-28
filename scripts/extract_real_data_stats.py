"""
Extract statistics from real Oesinghaus data (+ trained encoder) to calibrate
synthetic simulator hyperparameters.

Run on cluster — uses existing data + a previously-trained encoder.
No new training required.

Outputs saved to:
    results/real_data_stats/
        summary_report.txt       — human-readable parameter guide
        gene_stats.npz           — per-HVG variance decomposition
        cytokine_stats.npz       — per-cytokine delta vectors + effective rank
        embedding_stats.npz      — latent-space scatter/separation
        figures/                 — PNG plots

Usage:
    python scripts/extract_real_data_stats.py [--encoder_run <run_dir>]

Default encoder run: results/oesinghaus_full/new_seeds_seed1
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT     = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset
from cytokine_mil.data.label_encoder import CytokineLabel
from cytokine_mil.experiment_setup import build_encoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PSEUDO_TUBE_DIR = Path("/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes")
MANIFEST_PATH   = PSEUDO_TUBE_DIR / "manifest.json"
HVG_PATH        = PSEUDO_TUBE_DIR / "hvg_list.json"
STAGE1_MANIFEST = PSEUDO_TUBE_DIR / "manifest_stage1.json"
RESULTS_DIR     = REPO_ROOT / "results" / "real_data_stats"

DEFAULT_ENCODER_RUN = (
    REPO_ROOT / "results" / "oesinghaus_full" / "new_seeds_seed1"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        print(msg, file=fh, flush=True)


def _load_cells_from_manifest(manifest_path: Path, hvg_genes: list, n_max_tubes=None):
    """
    Load cells from all tubes in a manifest.

    Returns:
        X       : (N, G) float32 expression matrix
        ct_list : list[str]  cell-type label per cell
        cyt_list: list[str]  cytokine label per cell (tube-level)
        donor_list: list[str]
    """
    import anndata as ad

    with open(manifest_path) as f:
        manifest = json.load(f)

    gene_set = set(hvg_genes)
    Xs, cts, cyts, donors = [], [], [], []

    entries = manifest[:n_max_tubes] if n_max_tubes else manifest
    for entry in entries:
        adata = ad.read_h5ad(entry["path"])
        # Filter to HVG columns (intersection).
        available = [g for g in hvg_genes if g in adata.var_names]
        if not available:
            continue
        X_sub = adata[:, available].X
        if hasattr(X_sub, "toarray"):
            X_sub = X_sub.toarray()
        X_sub = X_sub.astype(np.float32)

        ct = adata.obs["cell_type"].tolist()
        cyt = [entry["cytokine"]] * adata.n_obs
        don = [entry["donor"]] * adata.n_obs
        Xs.append(X_sub)
        cts.extend(ct)
        cyts.extend(cyt)
        donors.extend(don)

    X = np.concatenate(Xs, axis=0)
    return X, cts, cyts, donors, available


# ---------------------------------------------------------------------------
# 1.  Expression-space variance decomposition (cell-type SNR per gene)
# ---------------------------------------------------------------------------

def compute_gene_variance_stats(X, ct_list, gene_names, log):
    """
    For each gene compute:
        within_ct_std  = mean over cell types of std(expr[cells_of_ct, g])
        between_ct_std = std of per-ct mean(expr[cells_of_ct, g])
        snr            = between / within
    """
    log("\n=== 1. Gene-level variance decomposition ===")

    cell_types = sorted(set(ct_list))
    ct_arr = np.array(ct_list)
    G = X.shape[1]

    ct_means = np.zeros((len(cell_types), G), dtype=np.float32)
    ct_stds  = np.zeros((len(cell_types), G), dtype=np.float32)
    ct_counts = []

    for ci, ct in enumerate(cell_types):
        mask = ct_arr == ct
        subset = X[mask]
        ct_counts.append(mask.sum())
        ct_means[ci] = subset.mean(axis=0)
        ct_stds[ci]  = subset.std(axis=0) + 1e-8

    within_ct_std  = ct_stds.mean(axis=0)    # (G,)
    between_ct_std = ct_means.std(axis=0)    # (G,)
    snr            = between_ct_std / within_ct_std

    log(f"  n_cell_types = {len(cell_types)}  |  n_cells = {len(ct_list)}")
    log(f"  n_genes = {G}")
    log(f"  Cells per type: {dict(zip(cell_types, ct_counts))}")
    log(f"\n  Per-gene statistics (across {G} HVGs):")
    log(f"    within-CT std   : mean={within_ct_std.mean():.3f}  "
        f"median={np.median(within_ct_std):.3f}  "
        f"p10={np.percentile(within_ct_std, 10):.3f}  "
        f"p90={np.percentile(within_ct_std, 90):.3f}")
    log(f"    between-CT std  : mean={between_ct_std.mean():.3f}  "
        f"median={np.median(between_ct_std):.3f}  "
        f"p10={np.percentile(between_ct_std, 10):.3f}  "
        f"p90={np.percentile(between_ct_std, 90):.3f}")
    log(f"    SNR (between/within): mean={snr.mean():.2f}  "
        f"median={np.median(snr):.2f}  "
        f"p90={np.percentile(snr, 90):.2f}")

    # Top vs bottom SNR genes
    top_snr_idx = np.argsort(snr)[-5:][::-1]
    log(f"\n  Top-5 SNR genes: "
        + ", ".join(f"{gene_names[i]} ({snr[i]:.2f})" for i in top_snr_idx))

    return {
        "within_ct_std": within_ct_std,
        "between_ct_std": between_ct_std,
        "snr": snr,
        "ct_means": ct_means,
        "ct_stds": ct_stds,
        "cell_types": cell_types,
        "gene_names_used": gene_names,
    }


# ---------------------------------------------------------------------------
# 2.  Per-cytokine program signal (delta from PBS in expression space)
# ---------------------------------------------------------------------------

def compute_cytokine_program_stats(X, ct_list, cyt_list, gene_names, log):
    """
    For each cytokine compute mean expression per gene, subtract matched PBS mean.
    Analyse:
        - delta_norm : L2 norm of mean delta vector (cytokine - PBS)
        - delta_snr  : per-gene delta / within-ct-std
        - n_sig_genes: genes with |delta| > 0.5 * within_ct_std
        - effective_rank: participation ratio of SVD on delta matrix
    """
    log("\n=== 2. Per-cytokine program signal (expression space) ===")

    ct_arr  = np.array(ct_list)
    cyt_arr = np.array(cyt_list)
    cytokines = sorted([c for c in set(cyt_arr) if c != "PBS"])
    G = X.shape[1]

    pbs_mask  = cyt_arr == "PBS"
    pbs_mean  = X[pbs_mask].mean(axis=0)  # global PBS mean

    delta_matrix = np.zeros((len(cytokines), G), dtype=np.float32)
    within_std = X[pbs_mask].std(axis=0) + 1e-8  # PBS within-all std as noise floor

    cyt_stats = {}
    for ci, cyt in enumerate(cytokines):
        mask = cyt_arr == cyt
        if mask.sum() == 0:
            continue
        cyt_mean = X[mask].mean(axis=0)
        delta    = cyt_mean - pbs_mean
        delta_matrix[ci] = delta

        n_sig = int((np.abs(delta) > 0.5 * within_std).sum())
        snr_vec = np.abs(delta) / within_std
        cyt_stats[cyt] = {
            "delta_l2":    float(np.linalg.norm(delta)),
            "delta_l1":    float(np.abs(delta).sum()),
            "n_sig_genes": n_sig,
            "top_delta":   float(np.max(np.abs(delta))),
            "snr_mean":    float(snr_vec.mean()),
            "snr_p90":     float(np.percentile(snr_vec, 90)),
        }

    delta_l2s = [v["delta_l2"] for v in cyt_stats.values()]
    n_sigs    = [v["n_sig_genes"] for v in cyt_stats.values()]
    snr_means = [v["snr_mean"] for v in cyt_stats.values()]

    log(f"  n_cytokines = {len(cytokines)}  (+ PBS)")
    log(f"\n  Cytokine delta L2 norm (vs PBS mean):")
    log(f"    mean={np.mean(delta_l2s):.3f}  median={np.median(delta_l2s):.3f}  "
        f"p10={np.percentile(delta_l2s, 10):.3f}  p90={np.percentile(delta_l2s, 90):.3f}")
    log(f"\n  n_significant_genes (|Δ| > 0.5 * within_std):")
    log(f"    mean={np.mean(n_sigs):.1f}  median={np.median(n_sigs):.1f}  "
        f"max={max(n_sigs)}  min={min(n_sigs)}")
    log(f"\n  Gene-level delta SNR (|Δ| / within_std), mean across genes:")
    log(f"    mean={np.mean(snr_means):.4f}  p90={np.percentile(snr_means, 90):.4f}")

    # Effective rank: participation ratio of singular values of delta matrix.
    U, s, Vt = np.linalg.svd(delta_matrix, full_matrices=False)
    eff_rank = float((s.sum() ** 2) / (s ** 2).sum())
    var_exp  = (s ** 2) / (s ** 2).sum()
    n_80pct  = int(np.searchsorted(np.cumsum(var_exp), 0.80)) + 1

    log(f"\n  SVD of cytokine delta matrix ({len(cytokines)} × {G}):")
    log(f"    effective rank (participation ratio) = {eff_rank:.2f}")
    log(f"    dims to explain 80% variance         = {n_80pct}")
    log(f"    top-5 singular values: "
        + "  ".join(f"{v:.3f}" for v in s[:5]))

    return {
        "per_cytokine": cyt_stats,
        "delta_matrix": delta_matrix,
        "cytokines_ordered": cytokines,
        "singular_values": s,
        "effective_rank": eff_rank,
        "n_dims_80pct": n_80pct,
        "pbs_within_std": within_std,
        "delta_l2_stats": {
            "mean": float(np.mean(delta_l2s)),
            "median": float(np.median(delta_l2s)),
            "p10": float(np.percentile(delta_l2s, 10)),
            "p90": float(np.percentile(delta_l2s, 90)),
        },
    }


# ---------------------------------------------------------------------------
# 3.  Cell-type separation in encoder embedding (latent space)
# ---------------------------------------------------------------------------

def compute_embedding_stats(encoder_path: Path, label_enc_path: Path,
                             hvg_genes: list, manifest_path: Path,
                             n_max_tubes: int, log):
    """
    Encode cells from a subset of tubes; compute within-CT scatter,
    between-CT separation, and PBS-RC signal norms in the embedding.
    """
    log("\n=== 3. Encoder embedding statistics (latent space) ===")

    # Load encoder weights + reconstruct architecture from label_encoder json.
    with open(label_enc_path) as f:
        lenc_data = json.load(f)
    n_genes = len(hvg_genes)

    # Infer n_cell_types from a sample tube.
    import anndata as ad
    with open(manifest_path) as f:
        manifest = json.load(f)
    sample_adata = ad.read_h5ad(manifest[0]["path"])
    n_cell_types = sample_adata.obs["cell_type"].nunique()
    log(f"  Detected {n_cell_types} cell types from sample tube")
    log(f"  n_input_genes = {n_genes}")

    encoder = build_encoder(n_input_genes=n_genes, n_cell_types=n_cell_types,
                            embed_dim=128)
    state = torch.load(encoder_path, map_location="cpu")
    encoder.load_state_dict(state, strict=False)
    encoder.eval()

    # Load cells from stage1 manifest (one tube per cytokine — manageable size).
    log(f"  Loading cells from {manifest_path.name} (up to {n_max_tubes} tubes)...")
    X, ct_list, cyt_list, donor_list, genes_used = _load_cells_from_manifest(
        manifest_path, hvg_genes, n_max_tubes=n_max_tubes
    )
    log(f"  Loaded {X.shape[0]} cells × {X.shape[1]} genes")

    # Align gene order to encoder's expected input.
    if len(genes_used) < n_genes:
        log(f"  [warn] Only {len(genes_used)}/{n_genes} HVGs found in tubes; "
            f"padding missing genes with 0.")
        gene_to_col = {g: i for i, g in enumerate(genes_used)}
        X_full = np.zeros((X.shape[0], n_genes), dtype=np.float32)
        for j, g in enumerate(hvg_genes):
            if g in gene_to_col:
                X_full[:, j] = X[:, gene_to_col[g]]
        X = X_full

    # Encode in batches.
    batch_size = 1024
    all_H = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.tensor(X[start:start + batch_size])
            H = encoder(batch)              # (B, 128)
            all_H.append(H.numpy())
    H = np.concatenate(all_H, axis=0)      # (N, 128)
    log(f"  Encoded {H.shape[0]} cells → H ∈ R^{H.shape[1]}")

    # Within-cell-type scatter.
    ct_arr  = np.array(ct_list)
    cyt_arr = np.array(cyt_list)
    cell_types = sorted(set(ct_arr))
    cytokines  = sorted([c for c in set(cyt_arr) if c != "PBS"])

    ct_centroids = {}
    within_ct_dists = []
    for ct in cell_types:
        mask   = ct_arr == ct
        subset = H[mask]
        centroid = subset.mean(axis=0)
        ct_centroids[ct] = centroid
        dists = np.linalg.norm(subset - centroid, axis=1)
        within_ct_dists.append(dists.mean())

    within_mean = float(np.mean(within_ct_dists))
    within_std  = float(np.std(within_ct_dists))

    # Between-cell-type separation.
    centroid_mat = np.stack(list(ct_centroids.values()), axis=0)  # (n_ct, 128)
    inter_dists  = []
    for i in range(len(cell_types)):
        for j in range(i + 1, len(cell_types)):
            inter_dists.append(
                float(np.linalg.norm(centroid_mat[i] - centroid_mat[j]))
            )
    between_mean = float(np.mean(inter_dists))

    log(f"\n  Embedding scatter (L2 distance to centroid):")
    log(f"    within-CT   : mean={within_mean:.3f}  std={within_std:.3f}")
    log(f"    between-CT  : mean pairwise centroid dist={between_mean:.3f}")
    log(f"    separation ratio = {between_mean / (within_mean + 1e-8):.2f}x")

    # PBS-RC signal: for PBS cells, compute per-cell-type centroid h̃
    # then measure L2 of (cyt_embedding - pbs_centroid_per_ct).
    pbs_ct_centroids = {}
    for ct in cell_types:
        mask = (ct_arr == ct) & (cyt_arr == "PBS")
        if mask.sum() > 0:
            pbs_ct_centroids[ct] = H[mask].mean(axis=0)

    pbs_rc_norms = {}
    for cyt in cytokines:
        cyt_mask = cyt_arr == cyt
        if cyt_mask.sum() == 0:
            continue
        pbs_rc_vecs = []
        for i in np.where(cyt_mask)[0]:
            ct_i = ct_arr[i]
            if ct_i in pbs_ct_centroids:
                pbs_rc_vecs.append(H[i] - pbs_ct_centroids[ct_i])
        if pbs_rc_vecs:
            pbs_rc_mat = np.stack(pbs_rc_vecs, axis=0)
            pbs_rc_norms[cyt] = float(np.linalg.norm(pbs_rc_mat, axis=1).mean())

    if pbs_rc_norms:
        norms = list(pbs_rc_norms.values())
        log(f"\n  PBS-RC signal norm (mean L2 of h̃ = h - µ_{{PBS,ct}}) per cytokine:")
        log(f"    mean={np.mean(norms):.3f}  median={np.median(norms):.3f}  "
            f"p10={np.percentile(norms, 10):.3f}  p90={np.percentile(norms, 90):.3f}")
        top5 = sorted(pbs_rc_norms.items(), key=lambda x: x[1], reverse=True)[:5]
        bot5 = sorted(pbs_rc_norms.items(), key=lambda x: x[1])[:5]
        log(f"    Top-5 cytokines (highest PBS-RC norm): "
            + ", ".join(f"{c} ({v:.3f})" for c, v in top5))
        log(f"    Bottom-5 cytokines (lowest PBS-RC norm): "
            + ", ".join(f"{c} ({v:.3f})" for c, v in bot5))

    # Effective dimensionality of the cytokine signal subspace.
    if len(cytokines) >= 3:
        cyt_means = []
        for cyt in cytokines:
            mask = cyt_arr == cyt
            if mask.sum() > 0:
                cyt_means.append(H[mask].mean(axis=0))
        cyt_mean_mat = np.stack(cyt_means, axis=0)  # (n_cyt, 128)
        _, s_emb, _ = np.linalg.svd(cyt_mean_mat, full_matrices=False)
        eff_rank_emb = float((s_emb.sum() ** 2) / (s_emb ** 2).sum())
        var_exp_emb  = (s_emb ** 2) / (s_emb ** 2).sum()
        n80_emb = int(np.searchsorted(np.cumsum(var_exp_emb), 0.80)) + 1
        log(f"\n  Cytokine centroid matrix in embedding space:")
        log(f"    effective rank (participation ratio) = {eff_rank_emb:.2f}")
        log(f"    dims to explain 80% variance         = {n80_emb}")
    else:
        eff_rank_emb, n80_emb = None, None

    return {
        "within_ct_mean_dist":  within_mean,
        "between_ct_mean_dist": between_mean,
        "separation_ratio":     between_mean / (within_mean + 1e-8),
        "pbs_rc_norms":         pbs_rc_norms,
        "emb_effective_rank":   eff_rank_emb,
        "emb_n_dims_80pct":     n80_emb,
    }


# ---------------------------------------------------------------------------
# 4.  Summary: recommended simulator parameters
# ---------------------------------------------------------------------------

def print_parameter_guide(gene_stats, cyt_stats, emb_stats, log):
    log("\n" + "=" * 70)
    log("CALIBRATION GUIDE — recommended SimConfig parameters")
    log("=" * 70)

    wc = gene_stats["within_ct_std"]
    bc = gene_stats["between_ct_std"]

    # Housekeeping genes: high expression, low SNR (near 1)
    low_snr_mask = gene_stats["snr"] < 1.5
    if low_snr_mask.any():
        hk_mu  = float(np.median(gene_stats["ct_means"].mean(axis=0)[low_snr_mask]))
        hk_sig = float(np.median(wc[low_snr_mask]))
    else:
        hk_mu, hk_sig = 2.0, 0.3

    # Marker genes: high expression in own type (top SNR quintile)
    hi_snr_mask = gene_stats["snr"] > np.percentile(gene_stats["snr"], 80)
    if hi_snr_mask.any():
        marker_hi_mu  = float(np.median(gene_stats["ct_means"].max(axis=0)[hi_snr_mask]))
        marker_lo_mu  = float(np.median(gene_stats["ct_means"].min(axis=0)[hi_snr_mask]))
        marker_hi_sig = float(np.median(wc[hi_snr_mask]))
    else:
        marker_hi_mu, marker_lo_mu, marker_hi_sig = 4.0, 0.2, 0.5

    # Cytokine program: real delta L2 norm
    real_delta_median = cyt_stats["delta_l2_stats"]["median"]
    # Each program has n_program_per_cytokine genes each with magnitude ~mag_mu.
    # Target: sqrt(n_prog) * mag_mu ≈ real_delta_median
    # → mag_mu ≈ real_delta_median / sqrt(n_prog)
    n_prog = 8  # current default
    implied_mag = real_delta_median / np.sqrt(n_prog)

    # PBS-RC signal vs within-CT scatter → what beta is detectable?
    if emb_stats and emb_stats.get("within_ct_mean_dist"):
        wc_emb = emb_stats["within_ct_mean_dist"]
        pbs_rc_median = float(np.median(list(emb_stats["pbs_rc_norms"].values()))) \
            if emb_stats.get("pbs_rc_norms") else None
        if pbs_rc_median:
            snr_emb = pbs_rc_median / wc_emb
            min_detectable_beta = 1.0 / snr_emb  # beta where secondary ≈ noise floor
        else:
            snr_emb = min_detectable_beta = None
    else:
        wc_emb = snr_emb = min_detectable_beta = None

    log(f"""
  Expression space:
    housekeeping_mu           ≈ {hk_mu:.2f}    (current: 2.0)
    housekeeping_sigma        ≈ {hk_sig:.3f}   (current: 0.3)
    marker_high_mu            ≈ {marker_hi_mu:.2f}   (current: 4.0)
    marker_low_mu             ≈ {marker_lo_mu:.3f}   (current: 0.2)
    marker_high_sigma         ≈ {marker_hi_sig:.3f}  (current: 0.5)
    program_magnitude_mu      ≈ {implied_mag:.3f}   (current: 1.5)
      [real delta L2={real_delta_median:.3f}, n_prog_genes={n_prog}]

  Cytokine program complexity:
    n_program_per_cytokine (matching real n_sig_genes)
      → target ≈ {cyt_stats['delta_l2_stats']['mean']:.0f} / (gene_mag)
      → effective rank of real delta matrix = {cyt_stats['effective_rank']:.1f}
         (use this as anchor: each synthetic cytokine should have ~{max(4, int(cyt_stats['effective_rank']))}-gene program)
    n_dims_80pct (expression)  = {cyt_stats['n_dims_80pct']}
""")
    if emb_stats:
        sep = emb_stats.get("separation_ratio", "N/A")
        sep_s = f"{sep:.1f}x" if isinstance(sep, float) else sep
        eff_r = emb_stats.get("emb_effective_rank", "N/A")
        eff_r_s = f"{eff_r:.1f}" if isinstance(eff_r, float) else eff_r
        log(f"""  Embedding space:
    within-CT scatter          = {wc_emb:.3f if wc_emb else 'N/A'}  (target: synthetic should match)
    between-CT / within-CT     = {sep_s}   (target: > 5x recommended)
""")
        if snr_emb:
            log(f"""  Cascade beta detectability:
    PBS-RC SNR (cyt signal / within-CT noise) = {snr_emb:.2f}
    Min detectable beta (secondary ≈ noise)   ≈ {min_detectable_beta:.2f}
    → beta should be > {min_detectable_beta:.2f} to be detectable in the embedding
      (current: 0.30–0.40, which is {'ABOVE' if 0.35 > min_detectable_beta else 'BELOW'} threshold)
""")
    log("=" * 70)


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------

def make_figures(gene_stats, cyt_stats, emb_stats, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. SNR histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    snr = gene_stats["snr"]
    ax.hist(snr[snr < 20], bins=60, color="steelblue", alpha=0.8, edgecolor="white", lw=0.4)
    ax.axvline(np.median(snr), color="tomato", ls="--", label=f"median={np.median(snr):.2f}")
    ax.set_xlabel("Cell-type SNR (between-CT std / within-CT std)")
    ax.set_ylabel("# HVGs")
    ax.set_title("Gene-level cell-type SNR — Oesinghaus HVGs")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "gene_snr_histogram.png", dpi=150); plt.close(fig)

    # 2. Cytokine delta L2 norms
    cyts = list(cyt_stats["per_cytokine"].keys())
    norms = [cyt_stats["per_cytokine"][c]["delta_l2"] for c in cyts]
    order = np.argsort(norms)[::-1]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(cyts)), [norms[i] for i in order], color="steelblue",
           edgecolor="white", lw=0.4)
    ax.set_xticks(range(len(cyts)))
    ax.set_xticklabels([cyts[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel("L2 norm of (mean_cyt − mean_PBS)")
    ax.set_title("Per-cytokine program strength (expression space)")
    plt.tight_layout()
    fig.savefig(fig_dir / "cytokine_delta_norms.png", dpi=150); plt.close(fig)

    # 3. SVD spectrum of delta matrix
    s = cyt_stats["singular_values"]
    var_exp = (s ** 2) / (s ** 2).sum()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, len(s) + 1), s, "o-", ms=4, color="steelblue")
    axes[0].set_xlabel("Rank"); axes[0].set_ylabel("Singular value")
    axes[0].set_title(f"SVD spectrum (eff. rank={cyt_stats['effective_rank']:.1f})")
    axes[1].plot(range(1, len(var_exp) + 1), np.cumsum(var_exp), "o-", ms=4, color="steelblue")
    axes[1].axhline(0.80, color="gray", ls="--", label="80%")
    axes[1].set_xlabel("# dimensions"); axes[1].set_ylabel("Cumulative var. explained")
    axes[1].set_title("Cytokine delta subspace")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "cytokine_svd_spectrum.png", dpi=150); plt.close(fig)

    # 4. PBS-RC signal norms in embedding (if available)
    if emb_stats and emb_stats.get("pbs_rc_norms"):
        pnorms = emb_stats["pbs_rc_norms"]
        sorted_items = sorted(pnorms.items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in sorted_items]
        vals   = [v for _, v in sorted_items]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(labels)), vals, color="darkorange", edgecolor="white", lw=0.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("Mean PBS-RC L2 norm (h̃ = h − µ_{PBS,ct})")
        ax.set_title("Cytokine signal strength in encoder embedding (PBS-RC)")
        plt.tight_layout()
        fig.savefig(fig_dir / "pbs_rc_signal_norms.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Extract real data statistics for simulator calibration.")
    p.add_argument("--encoder_run", type=str, default=str(DEFAULT_ENCODER_RUN),
                   help="Path to encoder run dir containing encoder_stage1.pt + label_encoder.json")
    p.add_argument("--n_max_tubes_expr", type=int, default=200,
                   help="Max tubes to load for expression stats (default 200, covering all conditions)")
    p.add_argument("--n_max_tubes_emb",  type=int, default=50,
                   help="Max tubes to encode for embedding stats (default 50 → ~24k cells)")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = RESULTS_DIR / "figures"
    log_path = RESULTS_DIR / "summary_report.txt"
    fh = open(log_path, "w")
    def log(msg=""):
        _log(msg, fh)

    log("=" * 70)
    log("Real data statistics — Oesinghaus PBMC cytokine dataset")
    log("=" * 70)
    log(f"Manifest   : {MANIFEST_PATH}")
    log(f"HVG list   : {HVG_PATH}")
    log(f"Encoder run: {args.encoder_run}")

    # Load HVG list.
    with open(HVG_PATH) as f:
        hvg_genes = json.load(f)
    log(f"HVGs loaded: {len(hvg_genes)}")

    # --------------------------------------------------------
    # 1 & 2: Expression-space stats from stage1 manifest
    # --------------------------------------------------------
    manifest_path = STAGE1_MANIFEST if STAGE1_MANIFEST.exists() else MANIFEST_PATH
    log(f"\nLoading expression data from {manifest_path.name}...")
    X, ct_list, cyt_list, donor_list, genes_used = _load_cells_from_manifest(
        manifest_path, hvg_genes, n_max_tubes=args.n_max_tubes_expr
    )
    log(f"  Loaded {X.shape[0]} cells × {X.shape[1]} genes")

    gene_stats = compute_gene_variance_stats(X, ct_list, genes_used, log)
    cyt_stats  = compute_cytokine_program_stats(X, ct_list, cyt_list, genes_used, log)

    np.savez(
        RESULTS_DIR / "gene_stats.npz",
        within_ct_std=gene_stats["within_ct_std"],
        between_ct_std=gene_stats["between_ct_std"],
        snr=gene_stats["snr"],
        ct_means=gene_stats["ct_means"],
    )
    np.savez(
        RESULTS_DIR / "cytokine_stats.npz",
        delta_matrix=cyt_stats["delta_matrix"],
        singular_values=cyt_stats["singular_values"],
    )

    # --------------------------------------------------------
    # 3: Embedding stats (needs encoder)
    # --------------------------------------------------------
    encoder_path = Path(args.encoder_run) / "encoder_stage1.pt"
    label_enc_path = Path(args.encoder_run) / "label_encoder.json"
    emb_stats = None

    if encoder_path.exists():
        try:
            emb_stats = compute_embedding_stats(
                encoder_path=encoder_path,
                label_enc_path=label_enc_path,
                hvg_genes=hvg_genes,
                manifest_path=manifest_path,
                n_max_tubes=args.n_max_tubes_emb,
                log=log,
            )
            np.savez(
                RESULTS_DIR / "embedding_stats.npz",
                pbs_rc_norms=list(emb_stats["pbs_rc_norms"].values()),
            )
        except Exception as e:
            log(f"\n  [warn] Embedding stats failed: {e}")
    else:
        log(f"\n  [warn] encoder_stage1.pt not found at {encoder_path} — skipping embedding stats")

    # --------------------------------------------------------
    # 4: Parameter guide
    # --------------------------------------------------------
    print_parameter_guide(gene_stats, cyt_stats, emb_stats, log)

    # --------------------------------------------------------
    # 5: Figures
    # --------------------------------------------------------
    make_figures(gene_stats, cyt_stats, emb_stats, fig_dir)
    log(f"\nAll outputs saved to {RESULTS_DIR}")
    fh.close()


if __name__ == "__main__":
    main()
