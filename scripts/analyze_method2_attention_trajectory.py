"""
Method 2 Analysis: Per-cell-type attention trajectories over training epochs.

For each cytokine and each cell type, tracks mean attention weight over training.
Key question: do specific cell types gain attention in specific cytokine contexts,
consistently across seeds? That cell type is a candidate cascade relay.

Biological prediction:
  - IL-12 tubes: NK cell attention should rise as model learns IL-12 classification
  - IFN-γ tubes: NK + CD14 Mono attention should dominate
  - IFN-β tubes: cDC or pDC attention should be prominent (innate sensing)

Stability: cross-seed Spearman rho of attention-trajectory slopes per (cytokine, cell_type).

Output: results/instrumented_analysis/method2_attention_trajectory/report.txt + summary.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE    = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full")
OUT_DIR = BASE / "instrumented_analysis" / "method2_attention_trajectory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_DIRS = {s: BASE / f"instrumented_seed{s}" for s in [11, 12, 13, 14, 15]}

# Expected dominant cell types per cytokine (for validation)
EXPECTED_DOMINANT = {
    "IL-12":    ["NK", "NK CD56bright"],
    "IFN-gamma":["NK", "CD14 Mono"],
    "IL-4":     ["B cell", "CD4 T"],
    "IL-2":     ["CD4 T", "CD8 T"],
    "TNF-alpha":["CD14 Mono"],
    "IFN-beta": ["pDC", "cDC"],
    "IL-27":    ["NK", "CD4 T"],
}

# Top cascade source cytokines from geo ensemble
CASCADE_SOURCES = ["IFN-beta", "IL-27", "IL-12"]
CASCADE_PAIRS = [
    ("IFN-beta",  "IL-27"),
    ("IL-27",     "IL-22"),
    ("IL-12",     "IFN-gamma"),
    ("IL-6",      "IL-10"),   # negative control
]


def find_cyt(trajectory, name):
    """Find cytokine key by case-insensitive partial match."""
    exact = [k for k in trajectory if k.lower() == name.lower()]
    if exact: return exact[0]
    partial = [k for k in trajectory if name.lower() in k.lower()]
    return partial[0] if partial else None


def top_n_cell_types(traj_dict, n=3):
    """Return top-n cell types by mean attention across epochs."""
    scores = {ct: float(np.mean(vals)) for ct, vals in traj_dict.items()}
    return sorted(scores, key=lambda k: scores[k], reverse=True)[:n]


def main():
    lines = []
    W = 72

    # Load all seeds
    seed_data = {}
    for seed, sdir in SEED_DIRS.items():
        pkl = sdir / "attention_trajectory.pkl"
        if not pkl.exists():
            print(f"  Seed {seed}: attention_trajectory.pkl not found, skipping.")
            continue
        with open(pkl, "rb") as f:
            seed_data[seed] = pickle.load(f)
        epochs = seed_data[seed]["epochs"]
        n_cyts = len(seed_data[seed]["cytokines"])
        n_cts  = len(seed_data[seed]["cell_types"])
        print(f"  Seed {seed}: epochs={epochs}, {n_cyts} cytokines, {n_cts} cell types")

    if not seed_data:
        print("No data found. Exiting.")
        return

    lines += ["=" * W, "Method 2: Attention Trajectory — Per-Cell-Type Dynamics", "=" * W, ""]
    lines += [f"Seeds: {sorted(seed_data.keys())}", ""]

    # ── 1. Validation against known biology ──────────────────────────────────
    lines += ["1. VALIDATION: Top cell types by attention (cross-seed mean)", "-" * W]
    for cyt_key, expected in EXPECTED_DOMINANT.items():
        lines.append(f"\n  {cyt_key}  (expected: {expected})")
        seed_tops = []
        for seed, data in seed_data.items():
            cyt = find_cyt(data["trajectory"], cyt_key)
            if cyt is None:
                lines.append(f"    Seed {seed}: not found")
                continue
            traj_dict = data["trajectory"][cyt]
            tops = top_n_cell_types(traj_dict, n=3)
            seed_tops.append(tops)
            hit = any(e.lower() in t.lower() or t.lower() in e.lower()
                      for t in tops for e in expected)
            mark = "✓" if hit else "✗"
            lines.append(f"    Seed {seed}: {tops}  {mark}")
        # Cross-seed consistency: do the top cells agree?
        if len(seed_tops) >= 2:
            all_tops = set(seed_tops[0]) if seed_tops else set()
            for tops in seed_tops[1:]:
                all_tops &= set(tops)
            lines.append(f"    Consistent top cells across seeds: {sorted(all_tops) or 'none'}")

    # ── 2. Cascade source cytokines: rising cell types ────────────────────────
    lines += ["", "2. CASCADE SOURCE CYTOKINES: Rising attention cell types", "-" * W]
    lines.append("  (cell types with largest attention increase from epoch 1 to last)")

    for src in CASCADE_SOURCES:
        lines.append(f"\n  {src}:")
        for seed, data in seed_data.items():
            cyt = find_cyt(data["trajectory"], src)
            if cyt is None:
                continue
            traj_dict = data["trajectory"][cyt]
            # Slope: last epoch minus first epoch
            slopes = {ct: float(vals[-1] - vals[0]) for ct, vals in traj_dict.items()}
            top_rising = sorted(slopes, key=lambda k: slopes[k], reverse=True)[:3]
            top_vals   = [f"{ct}(Δ{slopes[ct]:+.4f})" for ct in top_rising]
            lines.append(f"    Seed {seed}: {top_vals}")

    # ── 3. Cross-seed attention stability per (cytokine, cell type) ───────────
    lines += ["", "3. CROSS-SEED STABILITY: Spearman rho of attention trajectories", "-" * W]
    lines.append("  For each (cytokine, cell_type): compute Spearman rho of the")
    lines.append("  attention time-series between every pair of seeds.")
    lines.append("  High rho = consistently reproducible attention dynamics.\n")

    if len(seed_data) >= 2:
        ref_seed = next(iter(seed_data.values()))
        all_cyts = ref_seed["cytokines"]
        all_cts  = ref_seed["cell_types"]

        stable_pairs = []
        for cyt_name in all_cyts:
            for ct in all_cts:
                seed_trajs = []
                for data in seed_data.values():
                    cyt = find_cyt(data["trajectory"], cyt_name)
                    if cyt and ct in data["trajectory"][cyt]:
                        seed_trajs.append(data["trajectory"][cyt][ct])

                if len(seed_trajs) < 2:
                    continue
                rhos = []
                for i in range(len(seed_trajs)):
                    for j in range(i+1, len(seed_trajs)):
                        r, _ = spearmanr(seed_trajs[i], seed_trajs[j])
                        if not np.isnan(r):
                            rhos.append(r)
                if rhos:
                    mean_rho = float(np.mean(rhos))
                    # Also compute the slope (trend)
                    all_slopes = [float(t[-1] - t[0]) for t in seed_trajs]
                    mean_slope = float(np.mean(all_slopes))
                    stable_pairs.append({
                        "cytokine": cyt_name, "cell_type": ct,
                        "mean_rho": mean_rho, "mean_slope": mean_slope,
                        "n_seeds": len(seed_trajs),
                    })

        stable_pairs.sort(key=lambda x: x["mean_rho"], reverse=True)

        lines.append("  Top-20 most stable (cytokine, cell_type) attention pairs:")
        lines.append(f"  {'Cytokine':<22} {'Cell Type':<20} {'Rho':>6} {'Slope':>8}")
        lines.append("  " + "-" * 58)
        for entry in stable_pairs[:20]:
            lines.append(f"  {entry['cytokine']:<22} {entry['cell_type']:<20} "
                         f"{entry['mean_rho']:>6.3f} {entry['mean_slope']:>+8.4f}")

        # Filter: high rho + rising + cascade source cytokines
        cascade_stable = [p for p in stable_pairs
                          if p["mean_rho"] >= 0.8
                          and p["mean_slope"] > 0
                          and any(src.lower() in p["cytokine"].lower()
                                  for src in CASCADE_SOURCES)]
        lines += ["", "  Stable rising cell types in cascade source cytokines (rho ≥ 0.8):"]
        for entry in cascade_stable[:15]:
            lines.append(f"  {entry['cytokine']:<22} {entry['cell_type']:<20} "
                         f"rho={entry['mean_rho']:.3f}  slope={entry['mean_slope']:+.4f}")

        # Save summary
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump({
                "top_stable": stable_pairs[:100],
                "cascade_stable": cascade_stable,
                "seeds": sorted(seed_data.keys()),
            }, f, indent=2)

    report_str = "\n".join(lines)
    print(report_str)
    with open(OUT_DIR / "report.txt", "w") as f:
        f.write(report_str + "\n")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
