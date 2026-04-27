"""
Aggregate PBS-RC latent geometry results for the offset-1 batch (seeds 11-22)
and cross-compare with the offset-0 batch (seeds 42,123,7,1-10).

For each seed in each batch:
  - Load experiment_geo_pbs_rel/latent_geometry.pkl
  - Extract the (K, K) asymmetry matrix
  - Flatten off-diagonal entries (PBS excluded) into a vector

Reports per batch:
  - Per-seed Spearman rho vs intra-batch ensemble mean
  - Stability classification (rho >= 0.7)
  - Pairs of interest ranks in the stable ensemble

Cross-offset comparison:
  - Spearman rho between offset-0 stable ensemble mean and offset-1 stable ensemble mean
  - Pairs of interest: rank in each batch, rank difference
  - Top-50 stable pairs in each batch: overlap fraction

Saves to results/oesinghaus_full/geo_offset1_summary/:
  summary.json
  report.txt
  rho_plot.png          — per-seed rho within offset-1 batch
  cross_offset_plot.png — scatter: offset-0 vs offset-1 asymmetry scores
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full")
OUT_DIR     = RESULTS_DIR / "geo_offset1_summary"

# Offset-0 seeds (existing runs)
OFFSET0_SEEDS = {
    42:  "run_20260412_161758_seed42",
    123: "run_20260412_161803_seed123",
    7:   "run_20260412_161803_seed7",
    **{s: f"new_seeds_seed{s}" for s in [1, 2, 3, 4, 5, 6, 8, 9, 10]},
}
# Offset-1 seeds (new batch)
OFFSET1_SEEDS = {s: f"offset1_seed{s}" for s in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]}

STABLE_RHO_THRESHOLD = 0.7
TOP_N_STABLE         = 50
STABLE_SEED_FRAC     = 0.6

PAIRS_OF_INTEREST = [
    ("IL-12",    "IFN-gamma",  "positive control (cascade)"),
    ("IFN-gamma","IL-12",      "reverse (should be weaker)"),
    ("IL-6",     "IL-10",      "negative control (shared STAT3)"),
    ("IL-10",    "IL-6",       "negative control reverse"),
]


def _load_pkl(run_dir: Path):
    pkl_path = run_dir / "experiment_geo_pbs_rel" / "latent_geometry.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _extract_asym_vector(data: dict, pbs_label="PBS"):
    """Return (flat_vector, cytokine_names, pbs_idx)."""
    asym_result = data["asymmetry"]
    matrix      = asym_result["asymmetry_matrix"]   # (K, K)
    names       = asym_result["cytokine_names"]
    K           = len(names)
    pbs_idx     = next((i for i, n in enumerate(names) if n == pbs_label), None)
    pairs, vals = [], []
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            if pbs_idx is not None and (a == pbs_idx or b == pbs_idx):
                continue
            pairs.append((names[a], names[b]))
            vals.append(float(matrix[a, b]))
    return np.array(vals), pairs, names, pbs_idx


def _load_batch(seed_map: dict) -> dict:
    """Load all available seeds; return {seed: (vec, pairs, names, pbs_idx)}."""
    loaded = {}
    for seed, dirname in seed_map.items():
        run_dir = RESULTS_DIR / dirname
        data = _load_pkl(run_dir)
        if data is None:
            print(f"  [MISSING] seed={seed}  ({run_dir})")
            continue
        vec, pairs, names, pbs_idx = _extract_asym_vector(data)
        loaded[seed] = (vec, pairs, names, pbs_idx)
        print(f"  [OK]      seed={seed}  K={len(names)}  pairs={len(pairs)}")
    return loaded


def _compute_rhos(loaded: dict):
    """Per-seed Spearman rho vs batch ensemble mean."""
    seeds = sorted(loaded)
    mat   = np.stack([loaded[s][0] for s in seeds], axis=0)   # (n_seeds, n_pairs)
    ensemble_mean = mat.mean(axis=0)
    rhos = {}
    for i, s in enumerate(seeds):
        rho, _ = spearmanr(mat[i], ensemble_mean)
        rhos[s] = float(rho)
    return rhos, ensemble_mean, loaded[seeds[0]][1]   # pairs list from first seed


def _stable_ensemble(loaded, rhos):
    seeds = [s for s, r in rhos.items() if r >= STABLE_RHO_THRESHOLD]
    if not seeds:
        return None, seeds
    mat = np.stack([loaded[s][0] for s in seeds], axis=0)
    return mat.mean(axis=0), seeds


def _top_pairs(ensemble_vec, pairs, n=TOP_N_STABLE):
    ranked = sorted(zip(ensemble_vec, pairs), reverse=True)
    return [(p, float(v)) for v, p in ranked[:n]]


def _report_batch(label, loaded, rhos, stable_mean, stable_seeds, pairs, log):
    log(f"\n{'='*62}")
    log(f"Batch: {label}  ({len(loaded)} seeds loaded)")
    log(f"{'='*62}")
    log(f"\nPer-seed Spearman rho (vs batch ensemble mean):")
    for s in sorted(rhos):
        flag = "✓ stable" if rhos[s] >= STABLE_RHO_THRESHOLD else "✗ unstable"
        log(f"  seed={s:4d}  rho={rhos[s]:.3f}  {flag}")
    log(f"\nStable seeds ({len(stable_seeds)}/{len(loaded)}): {stable_seeds}")

    if stable_mean is None:
        log("  No stable seeds — cannot compute stable ensemble.")
        return
    name_to_local = {(a, b): i for i, (a, b) in enumerate(pairs)}
    log(f"\nPairs of interest (stable ensemble rank out of {len(pairs)}):")
    all_ranked = sorted(range(len(pairs)),
                        key=lambda i: stable_mean[i], reverse=True)
    rank_map = {idx: r+1 for r, idx in enumerate(all_ranked)}
    for src, tgt, note in PAIRS_OF_INTEREST:
        key = (src, tgt)
        if key not in name_to_local:
            log(f"  {src:<20} → {tgt:<20}  NOT FOUND  [{note}]")
            continue
        idx  = name_to_local[key]
        rank = rank_map[idx]
        log(f"  {src:<20} → {tgt:<20}  asym={stable_mean[idx]:.4f}  "
            f"rank={rank}/{len(pairs)}  [{note}]")

    log(f"\nTop-{TOP_N_STABLE} stable pairs (ensemble):")
    for (a, b), v in _top_pairs(stable_mean, pairs):
        log(f"  {a:<22} → {b:<22}  asym={v:.4f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log("Loading offset-0 batch...")
    loaded0 = _load_batch(OFFSET0_SEEDS)
    log("\nLoading offset-1 batch...")
    loaded1 = _load_batch(OFFSET1_SEEDS)

    if not loaded0 or not loaded1:
        log("Not enough data to compare — aborting.")
        sys.exit(1)

    # ── per-batch analysis ────────────────────────────────────────────────────
    rhos0, ensemble0, pairs0 = _compute_rhos(loaded0)
    rhos1, ensemble1, pairs1 = _compute_rhos(loaded1)

    stable_mean0, stable_seeds0 = _stable_ensemble(loaded0, rhos0)
    stable_mean1, stable_seeds1 = _stable_ensemble(loaded1, rhos1)

    _report_batch("offset-0 (original rotation)",
                  loaded0, rhos0, stable_mean0, stable_seeds0, pairs0, log)
    _report_batch("offset-1 (shifted rotation)",
                  loaded1, rhos1, stable_mean1, stable_seeds1, pairs1, log)

    # ── cross-offset comparison ───────────────────────────────────────────────
    log(f"\n{'='*62}")
    log("Cross-offset comparison")
    log(f"{'='*62}")

    if stable_mean0 is None or stable_mean1 is None:
        log("  Cannot cross-compare — at least one batch has no stable seeds.")
    else:
        assert pairs0 == pairs1, "Pair ordering differs between batches — cannot compare."
        rho_cross, p_cross = spearmanr(stable_mean0, stable_mean1)
        log(f"\n  Spearman rho(offset-0 stable ensemble, offset-1 stable ensemble) = "
            f"{rho_cross:.3f}  (p={p_cross:.2e})")
        if rho_cross >= 0.7:
            log("  ✓ CONSISTENT: same cascade pairs emerge from disjoint training cells.")
        else:
            log("  ✗ INCONSISTENT: geometry differs between donor rotations.")

        # Top-50 overlap
        top0 = {p for p, _ in _top_pairs(stable_mean0, pairs0)}
        top1 = {p for p, _ in _top_pairs(stable_mean1, pairs1)}
        overlap = top0 & top1
        log(f"\n  Top-{TOP_N_STABLE} overlap: {len(overlap)}/{TOP_N_STABLE} pairs appear in both batches")

        log(f"\n  Pairs of interest — rank in offset-0 vs offset-1:")
        name_to_local = {(a, b): i for i, (a, b) in enumerate(pairs0)}
        ranked0 = sorted(range(len(pairs0)), key=lambda i: stable_mean0[i], reverse=True)
        ranked1 = sorted(range(len(pairs1)), key=lambda i: stable_mean1[i], reverse=True)
        rank_map0 = {idx: r+1 for r, idx in enumerate(ranked0)}
        rank_map1 = {idx: r+1 for r, idx in enumerate(ranked1)}
        for src, tgt, note in PAIRS_OF_INTEREST:
            key = (src, tgt)
            if key not in name_to_local:
                continue
            idx   = name_to_local[key]
            r0    = rank_map0[idx]
            r1    = rank_map1[idx]
            delta = r1 - r0
            log(f"  {src:<20} → {tgt:<20}  rank0={r0:4d}  rank1={r1:4d}  "
                f"Δ={delta:+d}  [{note}]")

        # cross-offset scatter plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(stable_mean0, stable_mean1, s=4, alpha=0.4, color='steelblue')
        lim = max(abs(stable_mean0).max(), abs(stable_mean1).max()) * 1.05
        ax.axline((0, 0), slope=1, color='red', lw=0.8, ls='--', label='y=x')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Asymmetry score — offset-0 ensemble", fontsize=11)
        ax.set_ylabel("Asymmetry score — offset-1 ensemble", fontsize=11)
        ax.set_title(f"Cross-offset asymmetry agreement\nSpearman ρ = {rho_cross:.3f}", fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "cross_offset_plot.png", dpi=150)
        plt.close(fig)
        log(f"\n  Saved cross_offset_plot.png")

    # ── rho bar plot for offset-1 ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    seeds_sorted = sorted(rhos1)
    rho_vals     = [rhos1[s] for s in seeds_sorted]
    colors       = ['#2166AC' if r >= STABLE_RHO_THRESHOLD else '#D6604D' for r in rho_vals]
    ax.bar([str(s) for s in seeds_sorted], rho_vals, color=colors, edgecolor='white', lw=0.5)
    ax.axhline(STABLE_RHO_THRESHOLD, color='gray', ls='--', lw=1, label=f'threshold={STABLE_RHO_THRESHOLD}')
    ax.set_xlabel("Seed", fontsize=11)
    ax.set_ylabel("Spearman ρ (vs batch mean)", fontsize=11)
    ax.set_title("Per-seed stability — offset-1 batch", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rho_plot.png", dpi=150)
    plt.close(fig)

    # ── save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "offset0": {"n_loaded": len(loaded0), "rhos": {str(k): v for k, v in rhos0.items()},
                    "n_stable": len(stable_seeds0)},
        "offset1": {"n_loaded": len(loaded1), "rhos": {str(k): v for k, v in rhos1.items()},
                    "n_stable": len(stable_seeds1)},
    }
    if stable_mean0 is not None and stable_mean1 is not None:
        summary["cross_offset_rho"] = rho_cross
        summary["top50_overlap"]    = len(overlap)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT_DIR / "report.txt", "w") as f:
        f.write("\n".join(lines))

    log(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
