"""
Compare refined per-donor Wilcoxon PBS-RC method vs legacy asymmetry method.

Reads both fields from latent_geometry.pkl files (which contain both
`asymmetry` from the legacy path and `refined` from the new Wilcoxon path).

Metrics computed for both methods:
  - Within-batch seed stability (Spearman rho vs batch ensemble mean)
  - Cross-batch reproducibility (Spearman rho Batch 0 vs Batch 1)
  - Pre-registered control ranks / cascade calls
  - Top-N overlap between batches

Saves to results/oesinghaus_full/geo_refined_comparison/
  report.txt
  summary.json
  stability_plot.png   -- per-seed rho for both methods
  scatter_plot.png     -- cross-batch scatter for both methods
"""

import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full")
OUT_DIR     = RESULTS_DIR / "geo_refined_comparison"

OFFSET0_SEEDS = {
    42:  "run_20260412_161758_seed42",
    123: "run_20260412_161803_seed123",
    7:   "run_20260412_161803_seed7",
    **{s: f"new_seeds_seed{s}" for s in [1, 2, 3, 4, 5, 6, 8, 9, 10]},
}
OFFSET1_SEEDS = {s: f"offset1_seed{s}" for s in range(11, 23)}

STABLE_RHO_THRESHOLD = 0.70
TOP_N = 50

PAIRS_OF_INTEREST = [
    ("IL-12",     "IFN-gamma", "pos ctrl (cascade)"),
    ("IFN-gamma", "IL-12",     "pos ctrl (reverse)"),
    ("IL-6",      "IL-10",     "neg ctrl (shared STAT3)"),
    ("IL-10",     "IL-6",      "neg ctrl (reverse)"),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_pkl(run_dir: Path):
    p = run_dir / "experiment_geo_pbs_rel" / "latent_geometry.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _legacy_vector(data: dict, pbs_label="PBS"):
    """Flatten off-diagonal (non-PBS) asymmetry matrix → (pairs, values)."""
    asym   = data["asymmetry"]
    matrix = asym["asymmetry_matrix"]
    names  = asym["cytokine_names"]
    K      = len(names)
    pbs    = next((i for i, n in enumerate(names) if n == pbs_label), None)
    pairs, vals = [], []
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            if pbs is not None and (a == pbs or b == pbs):
                continue
            pairs.append((names[a], names[b]))
            vals.append(float(matrix[a, b]))
    return pairs, np.array(vals)


def _refined_vector(data: dict, pbs_label="PBS"):
    """
    -log10(min_T p_fwd_bonf) per ordered pair → (pairs, values).

    Bonferroni-corrected per-cell-type p-values are already stored in the pkl;
    we derive the per-pair min without rerunning the model or BH correction.
    This avoids the over-correction that collapsed all q_pair_fwd values to 1.0.
    """
    if data.get("refined") is None:
        return None, None
    sig = data["refined"]["significance"]

    asym  = data["asymmetry"]
    names = asym["cytokine_names"]
    K     = len(names)
    pbs   = next((i for i, n in enumerate(names) if n == pbs_label), None)

    # p_fwd_bonf is keyed by (A, B, ct); compute per-pair minimum.
    # Also accept p_pair_fwd if already stored (new pkl format).
    p_pair = sig.get("p_pair_fwd")
    if p_pair is None:
        p_fwd_bonf = sig.get("p_fwd_bonf", {})
        p_pair = {}
        for (a, b, ct), p in p_fwd_bonf.items():
            if p < p_pair.get((a, b), 1.0):
                p_pair[(a, b)] = p

    pairs, vals = [], []
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            if pbs is not None and (a == pbs or b == pbs):
                continue
            pair = (names[a], names[b])
            p    = p_pair.get(pair, 1.0)
            pairs.append(pair)
            vals.append(-np.log10(max(p, 1e-300)))
    return pairs, np.array(vals)


def _refined_cascade_bonf(data: dict, alpha: float = 0.05):
    """
    Derive Bonferroni-only cascade calls from stored p_fwd_bonf/p_rev_bonf.
    Does NOT apply BH across pairs — this is the corrected scoring.
    """
    if data.get("refined") is None:
        return {}
    sig = data["refined"]["significance"]

    p_fwd_bonf = sig.get("p_fwd_bonf", {})
    p_rev_bonf = sig.get("p_rev_bonf", {})

    pair_min_fwd: dict = {}
    pair_min_rev: dict = {}
    for (a, b, ct), p in p_fwd_bonf.items():
        if p < pair_min_fwd.get((a, b), 1.0):
            pair_min_fwd[(a, b)] = p
    for (a, b, ct), p in p_rev_bonf.items():
        if p < pair_min_rev.get((a, b), 1.0):
            pair_min_rev[(a, b)] = p

    cascade_call = {}
    for (a, b) in pair_min_fwd:
        fwd_sig = pair_min_fwd[(a, b)] <= alpha
        rev_sig = pair_min_rev.get((b, a), 1.0) <= alpha
        if fwd_sig and not rev_sig:
            cascade_call[(a, b)] = "A->B"
        elif rev_sig and not fwd_sig:
            cascade_call[(a, b)] = "B->A"
        elif fwd_sig and rev_sig:
            cascade_call[(a, b)] = "shared"
        else:
            cascade_call[(a, b)] = "none"
    return cascade_call



# ---------------------------------------------------------------------------
# Per-batch analysis
# ---------------------------------------------------------------------------

def _analyze_batch(seed_map: dict, batch_name: str):
    """
    Returns dict with keys:
      legacy_rhos, refined_rhos  — {seed: rho}
      pairs                      — list of (A, B) pairs (same order for all seeds)
      legacy_ensemble            — mean legacy vector over stable seeds
      refined_ensemble           — mean refined vector over stable seeds
      stable_legacy              — list of stable seed numbers (legacy criterion)
      stable_refined             — list of stable seed numbers (refined criterion)
      cascade_calls              — {seed: {(A, B): call}}
    """
    print(f"\n{'='*60}")
    print(f"Batch: {batch_name}")
    print(f"{'='*60}")

    per_seed_legacy  = {}
    per_seed_refined = {}
    per_seed_calls   = {}
    pairs_ref        = None

    for seed, dirname in seed_map.items():
        run_dir = RESULTS_DIR / dirname
        data    = _load_pkl(run_dir)
        if data is None:
            print(f"  [SKIP] seed {seed}: no pkl found at {run_dir}")
            continue
        if data.get("refined") is None:
            print(f"  [SKIP] seed {seed}: no 'refined' field — need to re-run geo")
            continue

        lpairs, lvals = _legacy_vector(data)
        rpairs, rvals = _refined_vector(data)

        if lpairs is None or rpairs is None:
            print(f"  [SKIP] seed {seed}: extraction failed")
            continue

        if pairs_ref is None:
            pairs_ref = lpairs
        per_seed_legacy[seed]  = lvals
        per_seed_refined[seed] = rvals
        per_seed_calls[seed]   = _refined_cascade_bonf(data)

    if not per_seed_legacy:
        print(f"  No valid seeds found for {batch_name}")
        return None

    # Stack into matrices: (n_seeds, n_pairs)
    seeds_l = sorted(per_seed_legacy.keys())
    L = np.stack([per_seed_legacy[s]  for s in seeds_l])
    R = np.stack([per_seed_refined[s] for s in seeds_l])

    def _stability(mat, seeds):
        """Per-seed Spearman rho vs ensemble mean (leave-one-out)."""
        rhos = {}
        for i, s in enumerate(seeds):
            others = np.delete(mat, i, axis=0)
            mean   = others.mean(axis=0)
            rho, _ = spearmanr(mat[i], mean)
            rhos[s] = float(rho)
        return rhos

    rhos_l = _stability(L, seeds_l)
    rhos_r = _stability(R, seeds_l)

    stable_l = [s for s in seeds_l if rhos_l[s] >= STABLE_RHO_THRESHOLD]
    stable_r = [s for s in seeds_l if rhos_r[s] >= STABLE_RHO_THRESHOLD]

    print(f"\n  Seeds evaluated  : {len(seeds_l)}")
    print(f"  Legacy stable    : {len(stable_l)}/{len(seeds_l)}  "
          f"({[s for s in seeds_l if rhos_l[s] < STABLE_RHO_THRESHOLD]} unstable)")
    print(f"  Refined stable   : {len(stable_r)}/{len(seeds_l)}  "
          f"({[s for s in seeds_l if rhos_r[s] < STABLE_RHO_THRESHOLD]} unstable)")

    print(f"\n  Per-seed rho (legacy / refined):")
    for s in seeds_l:
        mark_l = " " if rhos_l[s] >= STABLE_RHO_THRESHOLD else "*"
        mark_r = " " if rhos_r[s] >= STABLE_RHO_THRESHOLD else "*"
        print(f"    seed {s:4d}: legacy={rhos_l[s]:.3f}{mark_l}  refined={rhos_r[s]:.3f}{mark_r}")

    # Ensemble vectors from stable seeds
    stable_idx_l = [seeds_l.index(s) for s in stable_l]
    stable_idx_r = [seeds_l.index(s) for s in stable_r]
    ens_legacy  = L[stable_idx_l].mean(axis=0) if stable_idx_l else L.mean(axis=0)
    ens_refined = R[stable_idx_r].mean(axis=0) if stable_idx_r else R.mean(axis=0)

    # Pre-registered control ranks
    pair_to_idx = {p: i for i, p in enumerate(pairs_ref)}
    n_pairs = len(pairs_ref)

    print(f"\n  Pre-registered controls (ranks out of {n_pairs}):")
    print(f"  {'Pair':<35} {'Legacy':>12} {'Refined':>12}  {'Calls (mode)'}")
    for (src, tgt, role) in PAIRS_OF_INTEREST:
        pair = (src, tgt)
        idx  = pair_to_idx.get(pair)
        if idx is None:
            print(f"  {src}→{tgt} NOT FOUND")
            continue
        l_rank  = int(np.sum(ens_legacy  > ens_legacy[idx])  + 1)
        r_rank  = int(np.sum(ens_refined > ens_refined[idx]) + 1)
        # Most common cascade call across seeds
        calls   = [per_seed_calls.get(s, {}).get(pair, "n/a") for s in seeds_l]
        call_ctr = defaultdict(int)
        for c in calls:
            call_ctr[c] += 1
        mode_call = max(call_ctr, key=call_ctr.get)
        print(f"  {src:>10}→{tgt:<22}  {l_rank:>6}/{n_pairs}  {r_rank:>6}/{n_pairs}  "
              f"{mode_call} {dict(call_ctr)}")

    return {
        "pairs":           pairs_ref,
        "seeds":           seeds_l,
        "legacy_rhos":     rhos_l,
        "refined_rhos":    rhos_r,
        "stable_legacy":   stable_l,
        "stable_refined":  stable_r,
        "legacy_ensemble": ens_legacy,
        "refined_ensemble": ens_refined,
        "cascade_calls":   per_seed_calls,
    }


# ---------------------------------------------------------------------------
# Cross-batch comparison
# ---------------------------------------------------------------------------

def _cross_batch(b0, b1):
    print(f"\n{'='*60}")
    print("Cross-batch reproducibility")
    print(f"{'='*60}")

    # Align pairs
    pairs0 = b0["pairs"]
    pairs1 = b1["pairs"]
    shared = [p for p in pairs0 if p in set(pairs1)]
    idx0   = [pairs0.index(p) for p in shared]
    idx1   = [pairs1.index(p) for p in shared]

    l0 = b0["legacy_ensemble"][idx0]
    l1 = b1["legacy_ensemble"][idx1]
    r0 = b0["refined_ensemble"][idx0]
    r1 = b1["refined_ensemble"][idx1]

    rho_l, _ = spearmanr(l0, l1)
    rho_r, _ = spearmanr(r0, r1)

    print(f"\n  Shared pairs: {len(shared)}")
    print(f"  Cross-batch Spearman rho — legacy : {rho_l:.3f}")
    print(f"  Cross-batch Spearman rho — refined: {rho_r:.3f}")

    # Top-N overlap
    top_l0 = set(pairs0[i] for i in np.argsort(-b0["legacy_ensemble"])[:TOP_N])
    top_l1 = set(pairs1[i] for i in np.argsort(-b1["legacy_ensemble"])[:TOP_N])
    top_r0 = set(pairs0[i] for i in np.argsort(-b0["refined_ensemble"])[:TOP_N])
    top_r1 = set(pairs1[i] for i in np.argsort(-b1["refined_ensemble"])[:TOP_N])

    overlap_l = len(top_l0 & top_l1)
    overlap_r = len(top_r0 & top_r1)
    print(f"\n  Top-{TOP_N} overlap (out of {TOP_N}) — legacy : {overlap_l} pairs")
    print(f"  Top-{TOP_N} overlap (out of {TOP_N}) — refined: {overlap_r} pairs")

    shared_top_r = sorted(top_r0 & top_r1, key=lambda p: -b0["refined_ensemble"][pairs0.index(p)])
    if shared_top_r:
        print(f"\n  Top pairs stable in BOTH batches (refined, ranked by Batch-0 score):")
        for p in shared_top_r[:20]:
            i0 = pairs0.index(p)
            i1 = pairs1.index(p)
            print(f"    {p[0]:>20} → {p[1]:<20}  "
                  f"B0={b0['refined_ensemble'][i0]:.3f}  B1={b1['refined_ensemble'][i1]:.3f}")

    return {
        "rho_legacy":    float(rho_l),
        "rho_refined":   float(rho_r),
        "overlap_legacy":  overlap_l,
        "overlap_refined": overlap_r,
        "top_refined_both": [list(p) for p in shared_top_r],
        "l0": l0.tolist(), "l1": l1.tolist(),
        "r0": r0.tolist(), "r1": r1.tolist(),
        "pairs_aligned": [list(p) for p in shared],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_stability(b0, b1, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Within-batch seed stability: Legacy vs Refined", fontsize=13)

    for ax, batch, label in zip(axes, [b0, b1], ["Batch 0 (δ=0)", "Batch 1 (δ=1)"]):
        seeds   = batch["seeds"]
        rhos_l  = [batch["legacy_rhos"][s]  for s in seeds]
        rhos_r  = [batch["refined_rhos"][s] for s in seeds]
        x       = np.arange(len(seeds))
        w       = 0.35
        bars_l  = ax.bar(x - w/2, rhos_l, w, label="Legacy asymmetry", color="#4878d0", alpha=0.85)
        bars_r  = ax.bar(x + w/2, rhos_r, w, label="Refined Wilcoxon", color="#ee854a", alpha=0.85)
        ax.axhline(STABLE_RHO_THRESHOLD, color="k", ls="--", lw=1, label=f"ρ={STABLE_RHO_THRESHOLD}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds], fontsize=8)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Spearman ρ vs batch ensemble")
        ax.set_title(label)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "stability_plot.png", dpi=150)
    plt.close()


def _plot_scatter(cross, out_dir: Path):
    l0 = np.array(cross["l0"])
    l1 = np.array(cross["l1"])
    r0 = np.array(cross["r0"])
    r1 = np.array(cross["r1"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, x, y, rho, title, xlabel, ylabel in [
        (axes[0], l0, l1, cross["rho_legacy"],
         f"Legacy asymmetry — ρ={cross['rho_legacy']:.3f}",
         "Batch 0 asymmetry score", "Batch 1 asymmetry score"),
        (axes[1], r0, r1, cross["rho_refined"],
         f"Refined Wilcoxon (−log₁₀ p_bonf) — ρ={cross['rho_refined']:.3f}",
         "Batch 0  −log₁₀(p_bonf_fwd)", "Batch 1  −log₁₀(p_bonf_fwd)"),
    ]:
        ax.scatter(x, y, s=4, alpha=0.3, color="#4878d0")
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="y=x")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Cross-batch reproducibility: Legacy vs Refined", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_plot.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    b0 = _analyze_batch(OFFSET0_SEEDS, "Batch 0 (δ=0, seeds 1-10, 42, 123, 7)")
    b1 = _analyze_batch(OFFSET1_SEEDS, "Batch 1 (δ=1, seeds 11-22)")

    if b0 is None or b1 is None:
        print("\nERROR: one or both batches have no valid refined data. "
              "Re-run run_experiment_geo.py on all seeds first.")
        sys.exit(1)

    cross = _cross_batch(b0, b1)

    _plot_stability(b0, b1, OUT_DIR)
    _plot_scatter(cross, OUT_DIR)

    summary = {
        "batch0": {
            "stable_legacy":  len(b0["stable_legacy"]),
            "stable_refined": len(b0["stable_refined"]),
            "n_seeds":        len(b0["seeds"]),
            "legacy_rhos":    {str(k): v for k, v in b0["legacy_rhos"].items()},
            "refined_rhos":   {str(k): v for k, v in b0["refined_rhos"].items()},
        },
        "batch1": {
            "stable_legacy":  len(b1["stable_legacy"]),
            "stable_refined": len(b1["stable_refined"]),
            "n_seeds":        len(b1["seeds"]),
            "legacy_rhos":    {str(k): v for k, v in b1["legacy_rhos"].items()},
            "refined_rhos":   {str(k): v for k, v in b1["refined_rhos"].items()},
        },
        "cross_batch": {
            "rho_legacy":    cross["rho_legacy"],
            "rho_refined":   cross["rho_refined"],
            "overlap_legacy_top50":  cross["overlap_legacy"],
            "overlap_refined_top50": cross["overlap_refined"],
        },
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Batch 0 — legacy stable : {len(b0['stable_legacy'])}/{len(b0['seeds'])}"
          f"   refined stable : {len(b0['stable_refined'])}/{len(b0['seeds'])}")
    print(f"  Batch 1 — legacy stable : {len(b1['stable_legacy'])}/{len(b1['seeds'])}"
          f"   refined stable : {len(b1['stable_refined'])}/{len(b1['seeds'])}")
    print(f"  Cross-batch ρ — legacy  : {cross['rho_legacy']:.3f}")
    print(f"  Cross-batch ρ — refined : {cross['rho_refined']:.3f}")
    print(f"\n  Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
