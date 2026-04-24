"""
Aggregate PBS-RC latent geometry results across all seeds.

Loads experiment_geo_pbs_rel/latent_geometry.pkl from:
  - 3 existing seeds (42, 123, 7)
  - 9 new seeds     (1, 2, 3, 4, 5, 6, 8, 9, 10)

For each seed:
  - Extracts the (K, K) asymmetry matrix
  - Flattens off-diagonal entries (PBS excluded) into a vector
  - Computes Spearman rho vs ensemble mean

Reports:
  - Per-seed Spearman rho and stability classification (rho >= 0.7 = stable)
  - Pairs of interest (IL-12→IFN-γ, IL-6→IL-10, etc.) rank in each seed and stable ensemble
  - Top stable pairs (appearing in top-50 across >= 60% of stable seeds)
  - Per-seed run_summary.json metrics (final train/val accuracy)

Saves to results/oesinghaus_full/geo_ensemble_summary/:
  summary.json        <- all key metrics (machine-readable)
  report.txt          <- human-readable report
  rho_plot.png        <- Spearman rho per seed
  poi_ranks_plot.png  <- pairs of interest rank per seed
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
OUT_DIR     = RESULTS_DIR / "geo_ensemble_summary"

# Existing seeds → their run directories (fixed names)
EXISTING_SEEDS = {
    42:  "run_20260412_161758_seed42",
    123: "run_20260412_161803_seed123",
    7:   "run_20260412_161803_seed7",
}
# New seeds → fixed output dirs from run_new_seeds_train.slurm
NEW_SEEDS = [1, 2, 3, 4, 5, 6, 8, 9, 10]

STABLE_RHO_THRESHOLD = 0.7
TOP_N_STABLE         = 50       # define "top" as top-50 pairs per seed
STABLE_SEED_FRAC     = 0.6      # pair must appear in top-50 in >= 60% of stable seeds

PAIRS_OF_INTEREST = [
    ("IL-12",    "IFN-gamma",  "positive control (cascade)"),
    ("IFN-gamma","IL-12",      "reverse (should be weaker)"),
    ("IL-6",     "IL-10",      "negative control (shared STAT3)"),
    ("IL-10",    "IL-6",       "negative control reverse"),
]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_pkl(run_dir: Path) -> dict | None:
    pkl_path = run_dir / "experiment_geo_pbs_rel" / "latent_geometry.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _load_run_summary(run_dir: Path) -> dict | None:
    p = run_dir / "run_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _get_asym_vector(asym_matrix: np.ndarray, cyt_names: list[str]) -> np.ndarray:
    """Flatten off-diagonal asymmetry matrix entries, excluding PBS rows/cols."""
    K = asym_matrix.shape[0]
    pbs_idx = next((i for i, n in enumerate(cyt_names) if n == "PBS"), None)
    vals = []
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            if pbs_idx is not None and (a == pbs_idx or b == pbs_idx):
                continue
            vals.append(asym_matrix[a, b])
    return np.array(vals)


def _get_pair_rank(asym_matrix: np.ndarray, cyt_names: list[str],
                   src: str, tgt: str) -> tuple[float, int, int]:
    """Return (asym_value, rank_1indexed, n_total) for a pair, PBS-excluded."""
    name_to_idx = {n: i for i, n in enumerate(cyt_names)}
    pbs_idx = next((i for i, n in enumerate(cyt_names) if n == "PBS"), None)
    si = name_to_idx.get(src)
    ti = name_to_idx.get(tgt)
    if si is None or ti is None:
        return float("nan"), -1, -1
    val = float(asym_matrix[si, ti])
    # Collect all off-diagonal non-PBS values
    K = asym_matrix.shape[0]
    all_vals = [asym_matrix[a, b]
                for a in range(K) for b in range(K)
                if a != b and (pbs_idx is None or (a != pbs_idx and b != pbs_idx))]
    all_vals.sort(reverse=True)
    rank = sum(1 for v in all_vals if v > val) + 1
    return val, rank, len(all_vals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = []

    def log(msg=""):
        print(msg, flush=True)
        report_lines.append(msg)

    log("=" * 70)
    log("PBS-RC Latent Geometry — Full Ensemble Aggregation")
    log(f"Results dir : {RESULTS_DIR}")
    log(f"Stable rho  : >= {STABLE_RHO_THRESHOLD}")
    log("=" * 70)

    # ------------------------------------------------------------------
    # 1. Collect all seeds and load their data
    # ------------------------------------------------------------------
    all_seeds = {}  # seed -> {"run_dir", "pkl", "run_summary"}

    for seed, dirname in EXISTING_SEEDS.items():
        rd = RESULTS_DIR / dirname
        all_seeds[seed] = {
            "run_dir":    rd,
            "pkl":        _load_pkl(rd),
            "run_summary": _load_run_summary(rd),
            "kind":       "existing",
        }

    for seed in NEW_SEEDS:
        rd = RESULTS_DIR / f"new_seeds_seed{seed}"
        all_seeds[seed] = {
            "run_dir":    rd,
            "pkl":        _load_pkl(rd),
            "run_summary": _load_run_summary(rd),
            "kind":       "new",
        }

    # Filter to seeds where pkl loaded successfully
    valid_seeds = {s: d for s, d in all_seeds.items() if d["pkl"] is not None}
    failed_seeds = [s for s, d in all_seeds.items() if d["pkl"] is None]

    log(f"\nLoaded {len(valid_seeds)}/{len(all_seeds)} seeds successfully.")
    if failed_seeds:
        log(f"  MISSING: seeds {sorted(failed_seeds)}")

    if len(valid_seeds) < 2:
        log("ERROR: Need at least 2 seeds. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Build asymmetry vectors; compute Spearman rho vs ensemble mean
    # ------------------------------------------------------------------
    seed_list  = sorted(valid_seeds.keys())
    cyt_names  = valid_seeds[seed_list[0]]["pkl"]["asymmetry"]["cytokine_names"]
    asym_mats  = {}
    asym_vecs  = {}

    for s in seed_list:
        mat = valid_seeds[s]["pkl"]["asymmetry"]["asymmetry_matrix"]
        asym_mats[s] = mat
        asym_vecs[s] = _get_asym_vector(mat, cyt_names)

    vec_matrix   = np.stack([asym_vecs[s] for s in seed_list])   # (n_seeds, n_pairs)
    ensemble_mean = vec_matrix.mean(axis=0)

    rho_per_seed = {}
    for i, s in enumerate(seed_list):
        rho, _ = spearmanr(asym_vecs[s], ensemble_mean)
        rho_per_seed[s] = float(rho)

    # Stable seeds
    stable_seeds  = [s for s in seed_list if rho_per_seed[s] >= STABLE_RHO_THRESHOLD]
    unstable_seeds = [s for s in seed_list if rho_per_seed[s] < STABLE_RHO_THRESHOLD]

    log(f"\n{'Seed':>6}  {'Kind':>8}  {'Rho':>6}  {'Stable':>7}  "
        f"{'train_final':>12}  {'val_final':>10}")
    log("-" * 65)
    for s in seed_list:
        rs = valid_seeds[s]["run_summary"] or {}
        tf = rs.get("final_train_p_correct")
        vf = rs.get("final_val_p_correct")
        tf_str = f"{tf:.4f}" if tf is not None else "  N/A "
        vf_str = f"{vf:.4f}" if vf is not None else "  N/A "
        stable_str = "✓" if rho_per_seed[s] >= STABLE_RHO_THRESHOLD else "✗"
        log(f"  {s:>4}  {valid_seeds[s]['kind']:>8}  {rho_per_seed[s]:>6.3f}  "
            f"{stable_str:>7}  {tf_str:>12}  {vf_str:>10}")

    log(f"\nStable seeds ({len(stable_seeds)}/{len(seed_list)}): {stable_seeds}")
    log(f"Unstable seeds: {unstable_seeds}")

    # ------------------------------------------------------------------
    # 3. Pairs of interest — rank in each seed and stable ensemble
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("PAIRS OF INTEREST")
    log("=" * 70)

    poi_results = {}
    for src, tgt, note in PAIRS_OF_INTEREST:
        log(f"\n  {src} → {tgt}  [{note}]")
        per_seed = {}
        for s in seed_list:
            mat = asym_mats[s]
            val, rank, n_total = _get_pair_rank(mat, cyt_names, src, tgt)
            pct = 100.0 * rank / n_total if n_total > 0 else float("nan")
            stable_mark = "*" if s in stable_seeds else " "
            log(f"    {stable_mark}seed {s:>4}: asym={val:6.4f}  rank={rank:5}/{n_total}"
                f"  ({pct:.1f}th %ile)")
            per_seed[s] = {"asym": val, "rank": rank, "n_total": n_total, "pct": pct}

        # Stable ensemble stats
        stable_vals = [per_seed[s]["asym"] for s in stable_seeds if not np.isnan(per_seed[s]["asym"])]
        stable_rnks = [per_seed[s]["pct"]  for s in stable_seeds if not np.isnan(per_seed[s]["pct"])]
        if stable_vals:
            log(f"    → Stable ensemble: mean_asym={np.mean(stable_vals):.4f}±{np.std(stable_vals):.4f}"
                f"  mean_rank_pct={np.mean(stable_rnks):.1f}th %ile")
        poi_results[f"{src}→{tgt}"] = {"note": note, "per_seed": per_seed,
                                         "stable_mean_asym": float(np.mean(stable_vals)) if stable_vals else None,
                                         "stable_mean_pct":  float(np.mean(stable_rnks)) if stable_rnks else None}

    # ------------------------------------------------------------------
    # 4. Top stable pairs across stable seeds
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("TOP STABLE PAIRS (stable seeds only, top-50 per seed)")
    log("=" * 70)

    K = len(cyt_names)
    pbs_idx = next((i for i, n in enumerate(cyt_names) if n == "PBS"), None)

    pair_counts  = {}   # (a_name, b_name) -> count of stable seeds where pair is in top-N
    pair_means   = {}   # (a_name, b_name) -> mean asym across stable seeds

    for s in stable_seeds:
        mat = asym_mats[s]
        # Collect all off-diagonal PBS-excluded pairs sorted by asym
        pairs_sorted = []
        for a in range(K):
            for b in range(K):
                if a == b:
                    continue
                if pbs_idx is not None and (a == pbs_idx or b == pbs_idx):
                    continue
                pairs_sorted.append((float(mat[a, b]), cyt_names[a], cyt_names[b]))
        pairs_sorted.sort(reverse=True)
        top_n_set = {(an, bn) for _, an, bn in pairs_sorted[:TOP_N_STABLE]}

        for _, an, bn in pairs_sorted:
            key = (an, bn)
            pair_means.setdefault(key, []).append(float(mat[cyt_names.index(an), cyt_names.index(bn)]))
            if key in top_n_set:
                pair_counts[key] = pair_counts.get(key, 0) + 1

    min_stable_count = max(1, int(np.ceil(STABLE_SEED_FRAC * len(stable_seeds))))
    stable_pairs = [(cnt, key) for key, cnt in pair_counts.items() if cnt >= min_stable_count]
    stable_pairs.sort(key=lambda x: (x[0], np.mean(pair_means[x[1]])), reverse=True)

    log(f"\n  Threshold: >= {min_stable_count}/{len(stable_seeds)} stable seeds  "
        f"(>= {STABLE_SEED_FRAC*100:.0f}%)")
    log(f"  Found {len(stable_pairs)} stable pairs\n")
    log(f"  {'Pair':<45}  {'Seeds':>5}  {'Mean Asym':>10}  {'Std':>8}")
    log("  " + "-" * 75)
    for cnt, (an, bn) in stable_pairs[:30]:
        vals = pair_means[(an, bn)]
        log(f"  {an:<22} → {bn:<22}  {cnt:>3}/{len(stable_seeds)}"
            f"  {np.mean(vals):>10.4f}  {np.std(vals):>8.4f}")

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    _plot_rho(seed_list, rho_per_seed, stable_seeds, valid_seeds)
    _plot_poi_ranks(seed_list, stable_seeds, poi_results)

    # ------------------------------------------------------------------
    # 6. Save summary JSON
    # ------------------------------------------------------------------
    summary = {
        "n_seeds_total":    len(all_seeds),
        "n_seeds_loaded":   len(valid_seeds),
        "failed_seeds":     failed_seeds,
        "stable_seeds":     stable_seeds,
        "unstable_seeds":   unstable_seeds,
        "rho_per_seed":     rho_per_seed,
        "run_summaries":    {s: valid_seeds[s]["run_summary"] for s in seed_list},
        "pairs_of_interest": poi_results,
        "n_stable_pairs":   len(stable_pairs),
        "top_stable_pairs": [
            {
                "src": an, "tgt": bn,
                "n_stable_seeds": cnt,
                "mean_asym": float(np.mean(pair_means[(an, bn)])),
                "std_asym":  float(np.std(pair_means[(an, bn)])),
            }
            for cnt, (an, bn) in stable_pairs[:50]
        ],
        "config": {
            "stable_rho_threshold":  STABLE_RHO_THRESHOLD,
            "top_n_per_seed":        TOP_N_STABLE,
            "stable_seed_frac":      STABLE_SEED_FRAC,
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved: {OUT_DIR / 'summary.json'}")

    with open(OUT_DIR / "report.txt", "w") as f:
        f.write("\n".join(report_lines))
    log(f"Saved: {OUT_DIR / 'report.txt'}")

    log("\nDone.")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_rho(seed_list, rho_per_seed, stable_seeds, valid_seeds):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ecc71" if s in stable_seeds else "#e74c3c" for s in seed_list]
    bars = ax.bar([str(s) for s in seed_list],
                  [rho_per_seed[s] for s in seed_list],
                  color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.axhline(0.7, ls="--", lw=1.5, color="#333", alpha=0.7,
               label="ρ = 0.7 stability threshold")
    for bar, s in zip(bars, seed_list):
        rho = rho_per_seed[s]
        ax.text(bar.get_x() + bar.get_width() / 2, rho + 0.01,
                f"{rho:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("MIL Seed")
    ax.set_ylabel("Spearman ρ vs ensemble mean")
    ax.set_title("PBS-RC Ensemble: Seed Stability (Spearman ρ)\n"
                 "Green = stable (ρ ≥ 0.7),  Red = unstable")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rho_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'rho_plot.png'}")


def _plot_poi_ranks(seed_list, stable_seeds, poi_results):
    fig, ax = plt.subplots(figsize=(11, 5))
    pair_labels = list(poi_results.keys())
    n_pairs = len(pair_labels)
    x = np.arange(n_pairs)
    bar_w = 0.7 / len(seed_list)

    cmap = plt.cm.RdYlGn_r
    seed_colors = {s: cmap(i / max(len(seed_list) - 1, 1))
                   for i, s in enumerate(seed_list)}

    for vi, s in enumerate(seed_list):
        pcts = []
        for key in pair_labels:
            ps = poi_results[key]["per_seed"].get(s, {})
            pcts.append(ps.get("pct", float("nan")))
        xs = x + vi * bar_w - (len(seed_list) - 1) * bar_w / 2
        alpha = 0.9 if s in stable_seeds else 0.35
        ax.bar(xs, pcts, bar_w, color=seed_colors[s], alpha=alpha,
               edgecolor="white", linewidth=0.5,
               label=f"seed {s}" + (" *" if s in stable_seeds else ""))

    ax.axhline(50, ls="--", lw=1.5, color="#999", alpha=0.7, label="50th %ile (random)")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=9)
    ax.set_ylabel("Rank percentile (lower = stronger signal)")
    ax.set_title("PBS-RC: Pairs of Interest — Rank per Seed\n"
                 "Faded bars = unstable seeds (ρ < 0.7)  |  * = stable")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "poi_ranks_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'poi_ranks_plot.png'}")


if __name__ == "__main__":
    main()
