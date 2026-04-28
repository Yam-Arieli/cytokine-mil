"""
Aggregate PBS-RC latent geometry for the synthetic cascade dataset and compare
detected asymmetries against the known ground-truth cascade graph.

Ground truth (from cascade_ground_truth.json):
  True cascade edges   : cy1→cy2, cy3→cy4, cy4→cy5, cy6→cy7,
                         cy8→cy9, cy8→cy10, cy11→cy12, cy13→cy12
  Similar (no cascade) : (cy14,cy15), (cy16,cy17)
  Isolated             : cy18, cy19, cy20

Expected signal:
  - True cascade edges have higher asymmetry than non-cascade pairs
    (one-sided Mann-Whitney U, p < 0.05 after seeds are stable)
  - Reverse edges of true cascades have LOWER asymmetry than forward
  - Similar pairs show near-symmetric confusion (low |asymmetry|)
  - Isolated pairs show near-zero asymmetry against all others

Seeds used: 42, 123, 7, 1, 2, 3

Saves to results/synthetic_cascades/geo_summary/:
  summary.json
  report.txt
  rho_plot.png           — per-seed Spearman rho vs ensemble mean
  cascade_vs_null.png    — asymmetry distributions: true cascades vs null pairs
  ground_truth_ranks.png — rank of each true cascade edge across seeds
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

RESULTS_DIR    = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/synthetic_cascades")
SYNTHETIC_DIR  = Path("/cs/labs/mornitzan/yam.arieli/datasets/synthetic_cascades_v1")
OUT_DIR        = RESULTS_DIR / "geo_summary"

SEEDS = [42, 123, 7, 1, 2, 3]

STABLE_RHO_THRESHOLD = 0.7
TOP_N               = 50
STABLE_SEED_FRAC    = 0.6


# ---------------------------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------------------------

def _load_ground_truth() -> dict:
    with open(SYNTHETIC_DIR / "cascade_ground_truth.json") as f:
        return json.load(f)


def _build_ground_truth_sets(gt: dict):
    """Return (true_cascade_edges, similar_pairs, isolated_set)."""
    cascade_edges = {(e["src"], e["dst"]) for e in gt["cascades"]}
    similar_pairs = {frozenset(p) for p in gt["similar"]}
    isolated      = set(gt["isolated"])
    return cascade_edges, similar_pairs, isolated


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def _load_pkl(seed: int):
    pkl_path = RESULTS_DIR / f"seed{seed}" / "experiment_geo_pbs_rel" / "latent_geometry.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _extract_asym_vector(data: dict, pbs_label="PBS"):
    """Return (flat_vector, pairs, cytokine_names)."""
    asym_result = data["asymmetry"]
    matrix      = asym_result["asymmetry_matrix"]   # (K, K)
    names       = asym_result["cytokine_names"]
    K           = len(names)
    pbs_idx     = next((i for i, n in enumerate(names) if n == pbs_label), None)
    # Filter out empty-string placeholder classes (unused in synthetic).
    valid_idx   = [i for i, n in enumerate(names) if n and n != pbs_label]
    pairs, vals = [], []
    for a in valid_idx:
        for b in valid_idx:
            if a == b:
                continue
            pairs.append((names[a], names[b]))
            vals.append(float(matrix[a, b]))
    return np.array(vals), pairs, names


# ---------------------------------------------------------------------------
# Core analysis helpers
# ---------------------------------------------------------------------------

def _compute_rhos(loaded: dict):
    seeds = sorted(loaded)
    mat   = np.stack([loaded[s][0] for s in seeds], axis=0)
    ensemble_mean = mat.mean(axis=0)
    rhos = {}
    for i, s in enumerate(seeds):
        rho, _ = spearmanr(mat[i], ensemble_mean)
        rhos[s] = float(rho)
    return rhos, ensemble_mean, loaded[seeds[0]][1]


def _stable_ensemble(loaded, rhos):
    seeds = [s for s, r in rhos.items() if r >= STABLE_RHO_THRESHOLD]
    if not seeds:
        return None, seeds
    mat = np.stack([loaded[s][0] for s in seeds], axis=0)
    return mat.mean(axis=0), seeds


def _evaluate_ground_truth(ensemble_vec, pairs, cascade_edges, similar_pairs, isolated, log):
    """Compare detected asymmetry against known ground truth."""
    pair_to_idx = {p: i for i, p in enumerate(pairs)}
    n_pairs = len(pairs)

    # Rank all pairs by asymmetry.
    ranked = sorted(range(n_pairs), key=lambda i: ensemble_vec[i], reverse=True)
    rank_of = {idx: r + 1 for r, idx in enumerate(ranked)}

    # Collect scores by category.
    cascade_scores   = []  # forward cascade edges
    reverse_scores   = []  # reverse of cascade edges
    similar_scores   = []  # |asym| for similar pairs (should be ~0)
    isolated_scores  = []  # all edges touching isolated cytokines
    null_scores      = []  # everything else

    cascade_ranks = {}
    log("\nTrue cascade edge ranks:")
    for (src, dst) in sorted(cascade_edges):
        key = (src, dst)
        if key not in pair_to_idx:
            log(f"  {src} → {dst}  NOT FOUND in asymmetry matrix")
            continue
        idx   = pair_to_idx[key]
        score = float(ensemble_vec[idx])
        rank  = rank_of[idx]
        cascade_scores.append(score)
        cascade_ranks[key] = rank
        log(f"  {src:<10} → {dst:<10}  asym={score:+.4f}  rank={rank}/{n_pairs}")

    log("\nReverse edges (should rank lower):")
    for (src, dst) in sorted(cascade_edges):
        rev_key = (dst, src)
        if rev_key not in pair_to_idx:
            continue
        idx   = pair_to_idx[rev_key]
        score = float(ensemble_vec[idx])
        rank  = rank_of[idx]
        reverse_scores.append(score)
        log(f"  {dst:<10} → {src:<10}  asym={score:+.4f}  rank={rank}/{n_pairs}  (reverse)")

    log("\nSimilar-but-non-cascading pairs (expect near-zero |asym|):")
    for pair_set in sorted([sorted(p) for p in similar_pairs]):
        a, b = pair_set
        for src, dst in [(a, b), (b, a)]:
            key = (src, dst)
            if key not in pair_to_idx:
                continue
            idx   = pair_to_idx[key]
            score = float(ensemble_vec[idx])
            similar_scores.append(abs(score))
            log(f"  {src:<10} → {dst:<10}  asym={score:+.4f}  |asym|={abs(score):.4f}")

    log("\nIsolated cytokines (all edges, expect near-zero):")
    for (src, dst) in pairs:
        if src in isolated or dst in isolated:
            idx   = pair_to_idx[(src, dst)]
            score = float(ensemble_vec[idx])
            isolated_scores.append(abs(score))
    log(f"  {len(isolated_scores)} edges touching isolated cytokines  "
        f"mean|asym|={np.mean(isolated_scores):.4f}")

    # Null: all other edges.
    all_cascade_src_dst = cascade_edges | {(b, a) for a, b in cascade_edges}
    similar_flat = set()
    for (a, b) in pairs:
        if frozenset([a, b]) in similar_pairs:
            similar_flat.add((a, b))
    for (src, dst) in pairs:
        key = (src, dst)
        if key in all_cascade_src_dst:
            continue
        if key in similar_flat:
            continue
        if src in isolated or dst in isolated:
            continue
        null_scores.append(float(ensemble_vec[pair_to_idx[key]]))

    # Mann-Whitney U: true cascades vs null.
    log("\nMann-Whitney U (true cascade asym > null):")
    if cascade_scores and null_scores:
        stat, p = mannwhitneyu(cascade_scores, null_scores, alternative="greater")
        log(f"  cascade  n={len(cascade_scores)}  mean={np.mean(cascade_scores):.4f}")
        log(f"  null     n={len(null_scores)}  mean={np.mean(null_scores):.4f}")
        log(f"  U={stat:.0f}  p={p:.4f}  {'✓ SIGNIFICANT' if p < 0.05 else '✗ not significant'}")
    else:
        log("  Insufficient data for test.")
        p = 1.0

    # Directional check: forward > reverse for each cascade pair.
    log("\nDirectional check (forward asym > reverse asym):")
    n_correct = 0
    n_total   = 0
    for (src, dst) in sorted(cascade_edges):
        fwd_key = (src, dst)
        rev_key = (dst, src)
        if fwd_key not in pair_to_idx or rev_key not in pair_to_idx:
            continue
        fwd = float(ensemble_vec[pair_to_idx[fwd_key]])
        rev = float(ensemble_vec[pair_to_idx[rev_key]])
        correct = fwd > rev
        n_correct += int(correct)
        n_total   += 1
        log(f"  {src:<10} → {dst:<10}  fwd={fwd:+.4f}  rev={rev:+.4f}  "
            f"{'✓' if correct else '✗'}")
    if n_total:
        log(f"  Directional accuracy: {n_correct}/{n_total} = {n_correct/n_total:.0%}")

    return {
        "cascade_scores":    cascade_scores,
        "reverse_scores":    reverse_scores,
        "similar_abs_scores": similar_scores,
        "isolated_abs_scores": isolated_scores,
        "null_scores":       null_scores,
        "cascade_ranks":     {f"{a}→{b}": r for (a, b), r in cascade_ranks.items()},
        "mw_p_value":        p,
        "directional_accuracy": (n_correct / n_total) if n_total else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    gt = _load_ground_truth()
    cascade_edges, similar_pairs, isolated = _build_ground_truth_sets(gt)

    log("Loading ground truth...")
    log(f"  {len(cascade_edges)} cascade edges: {sorted(cascade_edges)}")
    log(f"  {len(similar_pairs)} similar pairs: {[sorted(p) for p in similar_pairs]}")
    log(f"  {len(isolated)} isolated: {sorted(isolated)}")

    log("\nLoading PBS-RC geometry results...")
    loaded = {}
    for seed in SEEDS:
        data = _load_pkl(seed)
        if data is None:
            log(f"  [MISSING] seed={seed}")
            continue
        vec, pairs, names = _extract_asym_vector(data)
        loaded[seed] = (vec, pairs, names)
        log(f"  [OK] seed={seed}  pairs={len(pairs)}")

    if not loaded:
        log("No results loaded — aborting."); sys.exit(1)

    # Per-seed stability.
    rhos, ensemble_mean, pairs = _compute_rhos(loaded)
    stable_mean, stable_seeds  = _stable_ensemble(loaded, rhos)

    log(f"\n{'='*62}")
    log("Per-seed Spearman rho (vs ensemble mean):")
    for s in sorted(rhos):
        flag = "✓ stable" if rhos[s] >= STABLE_RHO_THRESHOLD else "✗ unstable"
        log(f"  seed={s:4d}  rho={rhos[s]:.3f}  {flag}")
    log(f"\nStable seeds ({len(stable_seeds)}/{len(loaded)}): {stable_seeds}")

    gt_metrics = None
    if stable_mean is not None:
        log(f"\n{'='*62}")
        log("Ground-truth validation on stable ensemble:")
        log(f"{'='*62}")
        gt_metrics = _evaluate_ground_truth(
            stable_mean, pairs, cascade_edges, similar_pairs, isolated, log
        )

        # --- Plots ---

        # 1. Rho bar plot.
        fig, ax = plt.subplots(figsize=(8, 4))
        seeds_sorted = sorted(rhos)
        rho_vals = [rhos[s] for s in seeds_sorted]
        colors = ['#2166AC' if r >= STABLE_RHO_THRESHOLD else '#D6604D' for r in rho_vals]
        ax.bar([str(s) for s in seeds_sorted], rho_vals, color=colors, edgecolor='white', lw=0.5)
        ax.axhline(STABLE_RHO_THRESHOLD, color='gray', ls='--', lw=1,
                   label=f"threshold={STABLE_RHO_THRESHOLD}")
        ax.set_xlabel("Seed"); ax.set_ylabel("Spearman ρ (vs ensemble mean)")
        ax.set_title("Per-seed stability — synthetic cascade dataset")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "rho_plot.png", dpi=150); plt.close(fig)

        # 2. Cascade vs null asymmetry distribution.
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(
            min(min(gt_metrics["cascade_scores"] or [0]),
                min(gt_metrics["null_scores"] or [0])),
            max(max(gt_metrics["cascade_scores"] or [0]),
                max(gt_metrics["null_scores"] or [0])),
            30,
        )
        ax.hist(gt_metrics["null_scores"], bins=bins, alpha=0.5, color='steelblue',
                label=f"null (n={len(gt_metrics['null_scores'])})", density=True)
        ax.hist(gt_metrics["cascade_scores"], bins=bins, alpha=0.8, color='tomato',
                label=f"true cascades (n={len(gt_metrics['cascade_scores'])})", density=True)
        ax.axvline(0, color='gray', ls=':', lw=1)
        mw_p = gt_metrics.get("mw_p_value", 1.0)
        ax.set_xlabel("Asymmetry score (PBS-RC ensemble)")
        ax.set_ylabel("Density")
        ax.set_title(f"True cascade edges vs null\nMann-Whitney p={mw_p:.4f}")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "cascade_vs_null.png", dpi=150); plt.close(fig)

        # 3. Rank of each true cascade edge.
        cascade_rank_items = gt_metrics["cascade_ranks"]
        if cascade_rank_items:
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = list(cascade_rank_items.keys())
            ranks  = [cascade_rank_items[k] for k in labels]
            n_pairs = len(pairs)
            colors = ['#2ecc71' if r <= n_pairs * 0.1 else '#e74c3c' for r in ranks]
            ax.barh(labels, [n_pairs - r for r in ranks], color=colors, edgecolor='white', lw=0.5)
            ax.set_xlabel(f"Rank (higher bar = higher rank out of {n_pairs})")
            ax.set_title("True cascade edge ranks in stable ensemble (PBS-RC)")
            ax.axvline(n_pairs * 0.9, color='gray', ls='--', lw=1, label="top 10%")
            ax.legend(fontsize=9)
            plt.tight_layout()
            fig.savefig(OUT_DIR / "ground_truth_ranks.png", dpi=150); plt.close(fig)

    # --- Save summary JSON ---
    summary = {
        "seeds_loaded": sorted(loaded.keys()),
        "n_stable":     len(stable_seeds),
        "rhos":         {str(k): v for k, v in rhos.items()},
        "ground_truth": {
            "cascade_edges": [f"{a}→{b}" for a, b in sorted(cascade_edges)],
            "similar_pairs": [sorted(p) for p in sorted([sorted(q) for q in similar_pairs])],
            "isolated":      sorted(isolated),
        },
    }
    if gt_metrics:
        summary["validation"] = {
            "mw_p_value":           gt_metrics["mw_p_value"],
            "directional_accuracy": gt_metrics["directional_accuracy"],
            "cascade_ranks":        gt_metrics["cascade_ranks"],
            "mean_cascade_asym":    float(np.mean(gt_metrics["cascade_scores"])) if gt_metrics["cascade_scores"] else None,
            "mean_null_asym":       float(np.mean(gt_metrics["null_scores"])) if gt_metrics["null_scores"] else None,
            "mean_similar_abs":     float(np.mean(gt_metrics["similar_abs_scores"])) if gt_metrics["similar_abs_scores"] else None,
            "mean_isolated_abs":    float(np.mean(gt_metrics["isolated_abs_scores"])) if gt_metrics["isolated_abs_scores"] else None,
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_DIR / "report.txt", "w") as f:
        f.write("\n".join(lines))

    log(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
