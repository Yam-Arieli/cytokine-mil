"""
Method 3 Analysis: Trajectory shape as cross-seed stability filter.

Instead of comparing only the final asymmetry value across seeds (as we do in the
existing rho filter), compare the FULL TRAJECTORY SHAPE of ASYM(A,B,t).

A pair with:
  - high final asymmetry AND consistent trajectory shape → strong cascade evidence
  - high final asymmetry BUT erratic trajectory shape   → likely noise
  - consistent late-onset profile across all seeds      → most trustworthy

Metric: for each pair (A,B), compute Spearman rho between each pair of seeds'
full ASYM(A,B,t) trajectories. Mean rho across all seed pairs = trajectory stability.

Combined score: pairs ranked by (trajectory_rho × final_asym_mean)

Output: results/instrumented_analysis/method3_trajectory_stability/report.txt + summary.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE    = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full")
OUT_DIR = BASE / "instrumented_analysis" / "method3_trajectory_stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_DIRS = {s: BASE / f"instrumented_seed{s}" for s in [11, 12, 13, 14, 15]}

PAIRS_OF_INTEREST = [
    ("IL-12",    "IFN-gamma"),
    ("IFN-gamma","IL-12"),
    ("IL-6",     "IL-10"),
    ("IL-10",    "IL-6"),
    ("IFN-beta", "IL-27"),
    ("IL-27",    "IL-22"),
    ("IFN-beta", "TNF-alpha"),
    ("IL-27",    "IL-35"),
]

TRAJ_RHO_THRESHOLD = 0.7   # minimum trajectory Spearman rho to call a pair stable


def find_idx(cytokine_names, name):
    exact = [i for i, c in enumerate(cytokine_names) if c.lower() == name.lower()]
    if exact: return exact[0]
    partial = [i for i, c in enumerate(cytokine_names) if name.lower() in c.lower()]
    return partial[0] if partial else None


def main():
    lines = []
    W = 72

    # Load all seeds
    seed_data = {}
    for seed, sdir in SEED_DIRS.items():
        pkl = sdir / "geo_trajectory.pkl"
        if not pkl.exists():
            print(f"  Seed {seed}: geo_trajectory.pkl not found, skipping.")
            continue
        with open(pkl, "rb") as f:
            seed_data[seed] = pickle.load(f)
        print(f"  Seed {seed}: loaded, shape={seed_data[seed]['asymmetry_traj'].shape}")

    if len(seed_data) < 2:
        print("Need at least 2 seeds. Exiting.")
        return

    seeds = sorted(seed_data.keys())
    ref   = seed_data[seeds[0]]
    cytokine_names = ref["cytokine_names"]
    K = len(cytokine_names)
    pbs_idx = next((i for i, c in enumerate(cytokine_names) if c == "PBS"), None)

    lines += ["=" * W,
              "Method 3: Trajectory Shape Stability Filter",
              f"Seeds: {seeds}  |  Trajectory rho threshold: {TRAJ_RHO_THRESHOLD}",
              "=" * W, ""]

    # ── Pairs of interest ─────────────────────────────────────────────────────
    lines += ["PAIRS OF INTEREST", "-" * W]

    for src, tgt in PAIRS_OF_INTEREST:
        si = find_idx(cytokine_names, src)
        ti = find_idx(cytokine_names, tgt)
        if si is None or ti is None:
            lines.append(f"  {src} → {tgt}: not found")
            continue

        trajs = {seed: data["asymmetry_traj"][:, si, ti]
                 for seed, data in seed_data.items()}
        epochs = ref["epochs"]

        rhos = []
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i+1:]:
                r, _ = spearmanr(trajs[s1], trajs[s2])
                if not np.isnan(r):
                    rhos.append(r)
        mean_rho    = float(np.mean(rhos)) if rhos else 0.0
        finals      = [float(trajs[s][-1]) for s in seeds]
        mean_final  = float(np.mean(finals))
        std_final   = float(np.std(finals))
        combined    = mean_rho * mean_final

        lines.append(f"\n  {src} → {tgt}")
        lines.append(f"    Trajectory rho:  {mean_rho:.3f}  "
                     f"({'STABLE' if mean_rho >= TRAJ_RHO_THRESHOLD else 'unstable'})")
        lines.append(f"    Final asym:      mean={mean_final:.4f}  std={std_final:.4f}")
        lines.append(f"    Combined score:  {combined:.4f}")
        for seed in seeds:
            t = trajs[seed]
            lines.append(f"    Seed {seed}: {[f'{v:.3f}' for v in t]}  "
                         f"peak_ep={epochs[int(np.argmax(t))]}")

    # ── Full scan: all pairs ──────────────────────────────────────────────────
    lines += ["", "TOP STABLE PAIRS BY TRAJECTORY RHO × FINAL ASYMMETRY", "-" * W]

    all_pair_scores = []
    for a in range(K):
        if a == pbs_idx:
            continue
        for b in range(K):
            if a == b or b == pbs_idx:
                continue

            trajs = {seed: data["asymmetry_traj"][:, a, b]
                     for seed, data in seed_data.items()}

            rhos = []
            for i, s1 in enumerate(seeds):
                for s2 in seeds[i+1:]:
                    r, _ = spearmanr(trajs[s1], trajs[s2])
                    if not np.isnan(r):
                        rhos.append(r)
            if not rhos:
                continue

            mean_rho   = float(np.mean(rhos))
            finals     = [float(trajs[s][-1]) for s in seeds]
            mean_final = float(np.mean(finals))

            if mean_final <= 0:
                continue

            peak_indices = [int(np.argmax(trajs[s])) for s in seeds]
            mean_peak    = float(np.mean(peak_indices))
            n_epochs_total = ref["asymmetry_traj"].shape[0]
            ptype = ("late" if mean_peak / max(n_epochs_total - 1, 1) > 0.7 else
                     "early" if mean_peak / max(n_epochs_total - 1, 1) < 0.3 else "mid")

            all_pair_scores.append({
                "src": cytokine_names[a], "tgt": cytokine_names[b],
                "mean_rho": mean_rho, "mean_final": mean_final,
                "combined": mean_rho * mean_final,
                "profile": ptype, "mean_peak_idx": mean_peak,
                "n_seeds": len(seeds),
            })

    all_pair_scores.sort(key=lambda x: x["combined"], reverse=True)
    stable_scores = [p for p in all_pair_scores if p["mean_rho"] >= TRAJ_RHO_THRESHOLD]

    lines.append(f"  Total pairs: {len(all_pair_scores)}  "
                 f"Stable (rho ≥ {TRAJ_RHO_THRESHOLD}): {len(stable_scores)}\n")
    lines.append(f"  {'Source':<22} {'Target':<22} {'TrajRho':>7} {'Final':>7} "
                 f"{'Score':>7} {'Profile':<7}")
    lines.append("  " + "-" * 72)
    for p in stable_scores[:30]:
        lines.append(f"  {p['src']:<22} {p['tgt']:<22} "
                     f"{p['mean_rho']:>7.3f} {p['mean_final']:>7.4f} "
                     f"{p['combined']:>7.4f} {p['profile']:<7}")

    # Late-onset stable pairs (cascade candidates)
    late_stable = [p for p in stable_scores if p["profile"] == "late"]
    lines += ["", f"  Late-onset stable pairs (cascade candidates): {len(late_stable)}"]
    for p in late_stable[:15]:
        lines.append(f"  {p['src']:<22} → {p['tgt']:<22}  "
                     f"rho={p['mean_rho']:.3f}  final={p['mean_final']:.4f}")

    report_str = "\n".join(lines)
    print(report_str)
    with open(OUT_DIR / "report.txt", "w") as f:
        f.write(report_str + "\n")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "top_stable": stable_scores[:100],
            "late_stable": late_stable[:50],
            "seeds": seeds,
            "traj_rho_threshold": TRAJ_RHO_THRESHOLD,
        }, f, indent=2)

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
