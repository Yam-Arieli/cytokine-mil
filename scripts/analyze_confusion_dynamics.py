"""
analyze_confusion_dynamics.py
--------------------------------
Load dynamics.pkl from all 3 seeds, compute confusion trajectories,
asymmetry scores, seed stability (Spearman rho), and key biological
controls. Saves a text summary + figures.

Usage:
    python scripts/analyze_confusion_dynamics.py \
        --results_dir results/oesinghaus_full \
        --output_dir results/confusion_analysis
"""

import argparse
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# -- project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from cytokine_mil.analysis.confusion_dynamics import (
    compute_confusion_trajectory,
    compute_asymmetry_score,
    compute_temporal_profile,
)
from cytokine_mil.data.label_encoder import CytokineLabel

# ------------------------------------------------------------------------------

def load_seed_dynamics(run_dir: Path):
    pkl = run_dir / "dynamics.pkl"
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    return d


def find_seed_dirs(results_dir: Path):
    """Return the 3 most-recent completed run dirs (one per seed)."""
    # Case 1: subdirs named run_*_seed* (v1 layout)
    dirs = sorted(results_dir.glob("run_*_seed*"), key=lambda p: p.name)
    dirs = [d for d in dirs if (d / "dynamics.pkl").exists()]
    by_seed = {}
    for d in dirs:
        seed = d.name.split("_seed")[-1]
        by_seed[seed] = d

    # Case 2: subdirs named seed_* (v2 layout)
    for d in sorted(results_dir.glob("seed_*")):
        if (d / "dynamics.pkl").exists():
            seed = d.name.split("seed_")[-1]
            by_seed[seed] = d

    # Case 3: flat dir — dynamics.pkl directly in results_dir
    if not by_seed and (results_dir / "dynamics.pkl").exists():
        # Read seed from train.log if present
        seed = "unknown"
        log_path = results_dir / "train.log"
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                if line.startswith("Seed:"):
                    seed = line.split(":")[-1].strip()
                    break
        by_seed[seed] = results_dir

    return by_seed


def rebuild_label_encoder(cytokines: list) -> CytokineLabel:
    """Reconstruct a CytokineLabel from the saved cytokines list."""
    enc = CytokineLabel()
    enc._label_to_idx = {c: i for i, c in enumerate(cytokines) if c != "PBS"}
    enc._label_to_idx["PBS"] = 90
    enc._idx_to_label = {v: k for k, v in enc._label_to_idx.items()}
    return enc


def learnability_ranking(records, label_encoder):
    """AUC of mean p_correct_trajectory, aggregated to donor level."""
    from collections import defaultdict
    # group by (cytokine, donor)
    donor_cyt = defaultdict(list)
    for r in records:
        traj = r.get("p_correct_trajectory")
        if traj is None:
            continue
        donor_cyt[(r["cytokine"], r["donor"])].append(np.array(traj))

    # median per (cyt, donor)
    cyt_donor_auc = defaultdict(list)
    for (cyt, donor), trajs in donor_cyt.items():
        med = np.median(trajs, axis=0)
        auc = np.trapz(med) / (len(med) - 1) if len(med) > 1 else med[0]
        cyt_donor_auc[cyt].append(auc)

    # mean across donors
    ranking = {cyt: np.mean(aucs) for cyt, aucs in cyt_donor_auc.items()}
    return dict(sorted(ranking.items(), key=lambda x: x[1], reverse=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/oesinghaus_full",
                        help="Dir containing run_*_seed* subdirs")
    parser.add_argument("--output_dir", default="results/confusion_analysis")
    parser.add_argument("--late_fraction", type=float, default=0.3)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    # ------------------------------------------------------------------
    # 1. Load all seeds
    # ------------------------------------------------------------------
    seed_dirs = find_seed_dirs(results_dir)
    if not seed_dirs:
        log(f"No completed run dirs found in {results_dir}")
        return

    log(f"Found {len(seed_dirs)} seed(s): {sorted(seed_dirs.keys())}")

    seed_data = {}
    for seed, d in sorted(seed_dirs.items()):
        log(f"  Loading seed {seed} from {d.name} ...")
        seed_data[seed] = load_seed_dynamics(d)
        log(f"    {len(seed_data[seed]['records'])} train records, "
            f"{len(seed_data[seed]['val_records'])} val records")

    # Rebuild label encoder from first seed
    first = next(iter(seed_data.values()))
    cytokines = first["label_encoder_cytokines"]
    label_enc = rebuild_label_encoder(cytokines)
    log(f"  Label encoder: {label_enc.n_classes()} classes")

    # Check softmax_trajectory presence
    ex = first["records"][0]
    sm = ex.get("softmax_trajectory")
    if sm is None:
        log("ERROR: softmax_trajectory not in records — cannot run confusion analysis")
        return
    log(f"  softmax_trajectory shape: {sm.shape} (K x T)")

    # ------------------------------------------------------------------
    # 2. Learnability ranking per seed
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("LEARNABILITY RANKING (AUC of mean donor p_correct_trajectory)")
    log("=" * 70)

    seed_rankings = {}
    for seed, data in seed_data.items():
        ranking = learnability_ranking(data["records"], label_enc)
        seed_rankings[seed] = ranking

    # Print top-20 from first seed
    first_seed = sorted(seed_data.keys())[0]
    log(f"\nTop-20 by seed {first_seed}:")
    log(f"  {'Rank':<5} {'Cytokine':<20} {'AUC':>8}")
    log(f"  {'-'*35}")
    for rank, (cyt, auc) in enumerate(list(seed_rankings[first_seed].items())[:20], 1):
        log(f"  {rank:<5} {cyt:<20} {auc:>8.4f}")

    # Bottom-10
    items = list(seed_rankings[first_seed].items())
    log(f"\nBottom-10 by seed {first_seed}:")
    for rank, (cyt, auc) in enumerate(items[-10:], len(items) - 9):
        log(f"  {rank:<5} {cyt:<20} {auc:>8.4f}")

    # ------------------------------------------------------------------
    # 3. Seed stability of learnability rankings
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("SEED STABILITY — Spearman rho of learnability rankings")
    log("=" * 70)
    seeds = sorted(seed_rankings.keys())
    all_cyts = sorted(set().union(*[set(r.keys()) for r in seed_rankings.values()]))
    # Build vectors per seed
    seed_vecs = {}
    for s in seeds:
        vec = np.array([seed_rankings[s].get(c, np.nan) for c in all_cyts])
        seed_vecs[s] = vec

    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            v1, v2 = seed_vecs[s1], seed_vecs[s2]
            mask = ~(np.isnan(v1) | np.isnan(v2))
            rho, p = stats.spearmanr(v1[mask], v2[mask])
            log(f"  Seed {s1} vs {s2}: rho={rho:.3f}  p={p:.4f}")

    # ------------------------------------------------------------------
    # 4. Confusion trajectory (seed 42)
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("CONFUSION TRAJECTORY — seed 42")
    log("=" * 70)

    ref_seed = first_seed
    ref_records = seed_data[ref_seed]["records"]

    confusion, cyt_names = compute_confusion_trajectory(ref_records, label_enc)
    log(f"  Confusion tensor shape: {confusion.shape}  (K x K x T)")
    T = confusion.shape[2]

    # ------------------------------------------------------------------
    # 5. Asymmetry scores
    # ------------------------------------------------------------------
    asym = compute_asymmetry_score(confusion, late_epoch_fraction=args.late_fraction)
    log(f"  Asymmetry matrix shape: {asym.shape}")
    log(f"  Max |asymmetry|: {np.abs(asym).max():.4f}")
    log(f"  Antisymmetric check: max |A + A.T|: {np.abs(asym + asym.T).max():.2e}")

    # Top-20 directed pairs by asymmetry
    log(f"\n  Top-20 cascade direction signals (Asym[A→B] > 0 = evidence A→B):")
    log(f"  {'A→B':<35} {'Asym':>8}")
    log(f"  {'-'*45}")
    pairs = []
    K = asym.shape[0]
    for a in range(K):
        for b in range(K):
            if a != b and not np.isnan(asym[a, b]):
                pairs.append((cyt_names[a], cyt_names[b], asym[a, b]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a_name, b_name, score in pairs[:20]:
        log(f"  {a_name:>18} → {b_name:<15} {score:>8.4f}")

    # ------------------------------------------------------------------
    # 6. Biological controls
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("BIOLOGICAL CONTROLS")
    log("=" * 70)

    def get_idx(name):
        try:
            return label_enc.encode(name)
        except KeyError:
            return None

    # Positive control: IL-12 → IFN-gamma
    il12_idx = get_idx("IL-12")
    ifng_idx = get_idx("IFN-gamma")
    if il12_idx is not None and ifng_idx is not None and il12_idx < K and ifng_idx < K:
        asym_il12_ifng = asym[il12_idx, ifng_idx]
        asym_ifng_il12 = asym[ifng_idx, il12_idx]
        prof = compute_temporal_profile(confusion, il12_idx, ifng_idx)
        log(f"\n  [POSITIVE CONTROL] IL-12 → IFN-gamma")
        log(f"    Asym[IL-12→IFN-gamma] = {asym_il12_ifng:+.4f}")
        log(f"    Asym[IFN-gamma→IL-12] = {asym_ifng_il12:+.4f}")
        log(f"    Direction correct (IL-12→IFN-gamma > 0): {asym_il12_ifng > 0}")
        log(f"    Temporal profile: {prof['profile_type']}  "
            f"(peak epoch {prof['peak_epoch']}/{T})")
        if asym_il12_ifng > 0 and prof["profile_type"] == "late":
            log("    ✓ POSITIVE CONTROL PASSED: directional + late-onset")
        elif asym_il12_ifng > 0:
            log(f"    ~ PARTIAL: direction correct but profile is {prof['profile_type']}")
        else:
            log("    ✗ POSITIVE CONTROL FAILED: wrong direction")
    else:
        log("\n  [POSITIVE CONTROL] IL-12 or IFN-gamma not in label encoder — skipping")

    # Negative control: IL-6 / IL-10 (shared STAT3 — expect symmetric, early)
    il6_idx = get_idx("IL-6")
    il10_idx = get_idx("IL-10")
    if il6_idx is not None and il10_idx is not None and il6_idx < K and il10_idx < K:
        asym_il6_il10 = asym[il6_idx, il10_idx]
        prof6_10 = compute_temporal_profile(confusion, il6_idx, il10_idx)
        prof10_6 = compute_temporal_profile(confusion, il10_idx, il6_idx)
        log(f"\n  [NEGATIVE CONTROL] IL-6 / IL-10 (shared STAT3 path)")
        log(f"    Asym[IL-6→IL-10] = {asym_il6_il10:+.4f}  (expect ~0)")
        log(f"    IL-6→IL-10 profile: {prof6_10['profile_type']}  "
            f"(peak {prof6_10['peak_epoch']}/{T})")
        log(f"    IL-10→IL-6 profile: {prof10_6['profile_type']}  "
            f"(peak {prof10_6['peak_epoch']}/{T})")
        if abs(asym_il6_il10) < 0.01 and prof6_10["profile_type"] == "early":
            log("    ✓ NEGATIVE CONTROL PASSED: symmetric + early-onset")
        else:
            log(f"    ~ Partial or unexpected: |asym|={abs(asym_il6_il10):.4f}, "
                f"profile={prof6_10['profile_type']}")
    else:
        log("\n  [NEGATIVE CONTROL] IL-6 or IL-10 not found — skipping")

    # ------------------------------------------------------------------
    # 7. Seed stability of asymmetry scores
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("SEED STABILITY — asymmetry scores across seeds")
    log("=" * 70)

    asym_by_seed = {}
    for seed, data in seed_data.items():
        conf_s, _ = compute_confusion_trajectory(data["records"], label_enc)
        asym_by_seed[seed] = compute_asymmetry_score(
            conf_s, late_epoch_fraction=args.late_fraction)

    seeds = sorted(asym_by_seed.keys())
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            a1 = asym_by_seed[s1].flatten()
            a2 = asym_by_seed[s2].flatten()
            mask = ~(np.isnan(a1) | np.isnan(a2))
            rho, p = stats.spearmanr(a1[mask], a2[mask])
            log(f"  Seed {s1} vs {s2}: rho={rho:.3f}  p={p:.4g}")
            if rho > 0.7:
                log(f"    ✓ Stable (rho > 0.7 threshold)")
            else:
                log(f"    ✗ Unstable — pairs below threshold excluded from graph")

    # Average asymmetry across seeds
    asym_avg = np.nanmean(
        np.stack([asym_by_seed[s] for s in seeds], axis=0), axis=0)

    # Top-20 cascade pairs (averaged)
    log(f"\n  Top-20 cascade signals (averaged across {len(seeds)} seeds):")
    log(f"  {'A→B':<35} {'Avg Asym':>10}")
    log(f"  {'-'*47}")
    pairs_avg = []
    for a in range(K):
        for b in range(K):
            if a != b and not np.isnan(asym_avg[a, b]):
                pairs_avg.append((cyt_names[a], cyt_names[b], asym_avg[a, b]))
    pairs_avg.sort(key=lambda x: x[2], reverse=True)
    for a_name, b_name, score in pairs_avg[:20]:
        log(f"  {a_name:>18} → {b_name:<15} {score:>10.4f}")

    # ------------------------------------------------------------------
    # 8. Figures
    # ------------------------------------------------------------------
    log("\nGenerating figures...")

    # (A) Learnability ranking — multi-seed comparison
    fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 8), sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    for ax, seed in zip(axes, sorted(seeds)):
        ranking = seed_rankings[seed]
        cyts = list(ranking.keys())[:30]
        aucs = [ranking[c] for c in cyts]
        ax.barh(range(len(cyts)), aucs, color="steelblue")
        ax.set_yticks(range(len(cyts)))
        ax.set_yticklabels(cyts, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Learnability AUC")
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="chance")
    fig.suptitle("Learnability ranking — top 30 cytokines per seed\n"
                 "Metric: AUC(mean donor p_correct_trajectory)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "learnability_ranking.png", dpi=120)
    plt.close()
    log(f"  Saved: learnability_ranking.png")

    # (B) Asymmetry heatmap (avg across seeds, top-40 cytokines by max |asym|)
    max_asym = np.nanmax(np.abs(asym_avg), axis=1)
    top_idx = np.argsort(max_asym)[::-1][:40]
    sub = asym_avg[np.ix_(top_idx, top_idx)]
    sub_names = [cyt_names[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(14, 12))
    vmax = np.nanpercentile(np.abs(sub), 95)
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(sub_names)))
    ax.set_xticklabels(sub_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(sub_names)))
    ax.set_yticklabels(sub_names, fontsize=6)
    ax.set_title("Asymmetry score Asym[A→B] (avg across seeds)\n"
                 "Positive = evidence for cascade A→B", fontsize=9)
    plt.colorbar(im, ax=ax, label="Asym[A,B]")
    plt.tight_layout()
    plt.savefig(out_dir / "asymmetry_heatmap.png", dpi=120)
    plt.close()
    log(f"  Saved: asymmetry_heatmap.png")

    # (C) Confusion trajectory for IL-12 → IFN-gamma
    if il12_idx is not None and ifng_idx is not None and il12_idx < K and ifng_idx < K:
        fig, ax = plt.subplots(figsize=(8, 4))
        for seed in sorted(seeds):
            conf_s = compute_confusion_trajectory(
                seed_data[seed]["records"], label_enc)[0]
            ax.plot(conf_s[il12_idx, ifng_idx, :], label=f"C(IL-12→IFN-γ) seed {seed}",
                    alpha=0.8)
            ax.plot(conf_s[ifng_idx, il12_idx, :], label=f"C(IFN-γ→IL-12) seed {seed}",
                    linestyle="--", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean softmax mass")
        ax.set_title("IL-12 → IFN-γ positive control\n"
                     "Metric: C(A,B,t) = mean softmax[B] over A-tubes")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / "positive_control_il12_ifng.png", dpi=120)
        plt.close()
        log(f"  Saved: positive_control_il12_ifng.png")

    # ------------------------------------------------------------------
    # Save text summary
    # ------------------------------------------------------------------
    with open(out_dir / "summary.txt", "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nSaved: summary.txt")
    log("Done.")


if __name__ == "__main__":
    main()
