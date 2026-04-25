"""
Method 1 Analysis: Temporal profiles of PBS-RC asymmetry trajectories.

For each cytokine pair (A→B):
  - Extract ASYM(A,B,t) across epochs from geo_trajectory.pkl
  - Characterise: onset_epoch, peak_epoch, profile_type (early/mid/late)
  - Cross-seed: compare trajectory shapes (Spearman rho of peak_epoch across seeds)

Biological prediction:
  - True cascade pair: asymmetry emerges LATE (model learns direct signal first)
  - Shared pathway: asymmetry emerges EARLY (both cytokines activate same program)
  - Controls: IL-12→IFN-γ should be late-onset; IL-6→IL-10 should be early-onset

Output: results/instrumented_analysis/method1_geo_trajectory/report.txt + summary.json
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE     = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full")
OUT_DIR  = BASE / "instrumented_analysis" / "method1_geo_trajectory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_DIRS = {s: BASE / f"instrumented_seed{s}" for s in [11, 12, 13, 14, 15]}

PAIRS_OF_INTEREST = [
    ("IL-12",    "IFN-gamma",  "positive control"),
    ("IFN-gamma","IL-12",      "reverse"),
    ("IL-6",     "IL-10",      "negative control"),
    ("IL-10",    "IL-6",       "neg ctrl reverse"),
    ("IFN-beta", "IL-27",      "top stable pair"),
    ("IL-27",    "IL-22",      "top stable pair"),
    ("IFN-beta", "TNF-alpha",  "top stable pair"),
]

LATE_FRACTION  = 0.3   # fraction of epochs considered "late"
EARLY_FRACTION = 0.3   # fraction of epochs considered "early"


def onset_epoch(traj):
    """First epoch where ASYM > 5% of its max."""
    mx = traj.max()
    if mx <= 0:
        return len(traj) - 1
    threshold = 0.05 * mx
    idx = np.argmax(traj >= threshold)
    return int(idx)


def profile_type(peak_idx, n_epochs):
    frac = peak_idx / max(n_epochs - 1, 1)
    if frac < EARLY_FRACTION:
        return "early"
    if frac > (1 - LATE_FRACTION):
        return "late"
    return "mid"


def find_pair_indices(cytokine_names, src, tgt):
    """Case-insensitive partial match."""
    def find(name):
        exact = [i for i, c in enumerate(cytokine_names) if c.lower() == name.lower()]
        if exact: return exact[0]
        partial = [i for i, c in enumerate(cytokine_names) if name.lower() in c.lower()]
        return partial[0] if partial else None
    return find(src), find(tgt)


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
        print(f"  Seed {seed}: epochs={seed_data[seed]['epochs']}, "
              f"shape={seed_data[seed]['asymmetry_traj'].shape}")

    if not seed_data:
        print("No data found. Exiting.")
        return

    lines += ["=" * W, "Method 1: Geo Trajectory — Temporal Profile Analysis", "=" * W, ""]
    lines += [f"Seeds loaded: {sorted(seed_data.keys())}", ""]

    # ── Pairs of interest analysis ────────────────────────────────────────────
    lines += ["PAIRS OF INTEREST", "-" * W]
    poi_summary = []

    for src, tgt, label in PAIRS_OF_INTEREST:
        lines.append(f"\n  {src} → {tgt}  [{label}]")
        seed_peaks = []
        seed_onsets = []

        for seed, data in seed_data.items():
            si, ti = find_pair_indices(data["cytokine_names"], src, tgt)
            if si is None or ti is None:
                lines.append(f"    Seed {seed}: pair not found")
                continue

            traj = data["asymmetry_traj"][:, si, ti]   # (n_epochs,)
            epochs = data["epochs"]
            n = len(traj)

            peak_idx  = int(np.argmax(traj))
            peak_ep   = epochs[peak_idx]
            onset_idx = onset_epoch(traj)
            onset_ep  = epochs[onset_idx]
            ptype     = profile_type(peak_idx, n)
            final_val = float(traj[-1])
            max_val   = float(traj.max())

            seed_peaks.append(peak_idx)
            seed_onsets.append(onset_idx)

            lines.append(f"    Seed {seed}: onset=ep{onset_ep}  peak=ep{peak_ep}  "
                         f"type={ptype}  max={max_val:.4f}  final={final_val:.4f}")

        if len(seed_peaks) >= 2:
            peak_std  = float(np.std(seed_peaks))
            onset_std = float(np.std(seed_onsets))
            # Peak consistency (low std = consistent trajectory shape)
            peak_range = max(seed_peaks) - min(seed_peaks) if seed_peaks else 0
            lines.append(f"    Cross-seed: peak_std={peak_std:.1f} epochs  "
                         f"onset_std={onset_std:.1f} epochs  "
                         f"peak_range={peak_range} epochs")
            poi_summary.append({
                "pair": f"{src} → {tgt}",
                "label": label,
                "mean_peak_idx": float(np.mean(seed_peaks)),
                "peak_std": peak_std,
                "mean_onset_idx": float(np.mean(seed_onsets)),
                "onset_std": onset_std,
                "n_seeds": len(seed_peaks),
            })

    # ── Top-20 most stable late-onset pairs (cross-seed peak_std < 1 epoch) ──
    lines += ["", "TOP STABLE LATE-ONSET PAIRS (cascade candidates)", "-" * W]
    pair_profiles = []

    # Use first seed's cytokine_names as reference
    ref_seed = next(iter(seed_data.values()))
    cytokine_names = ref_seed["cytokine_names"]
    K = len(cytokine_names)
    pbs_idx = next((i for i, c in enumerate(cytokine_names) if c == "PBS"), None)

    for a in range(K):
        if a == pbs_idx:
            continue
        for b in range(K):
            if a == b or b == pbs_idx:
                continue

            peaks_all = []
            finals_all = []
            for data in seed_data.values():
                traj = data["asymmetry_traj"][:, a, b]
                peaks_all.append(int(np.argmax(traj)))
                finals_all.append(float(traj[-1]))

            if len(peaks_all) < 2:
                continue

            mean_peak = float(np.mean(peaks_all))
            std_peak  = float(np.std(peaks_all))
            n_epochs  = ref_seed["asymmetry_traj"].shape[0]
            ptype     = profile_type(int(round(mean_peak)), n_epochs)
            mean_final = float(np.mean(finals_all))

            pair_profiles.append({
                "a": a, "b": b,
                "src": cytokine_names[a], "tgt": cytokine_names[b],
                "mean_peak_idx": mean_peak,
                "peak_std": std_peak,
                "profile_type": ptype,
                "mean_final": mean_final,
                "n_seeds": len(peaks_all),
            })

    # Filter: late-onset, low peak_std, positive final asymmetry
    late_stable = [p for p in pair_profiles
                   if p["profile_type"] == "late"
                   and p["peak_std"] <= 1.0
                   and p["mean_final"] > 0
                   and p["n_seeds"] == len(seed_data)]
    late_stable.sort(key=lambda p: p["mean_final"], reverse=True)

    lines.append(f"  (late-onset, peak_std ≤ 1 epoch, all {len(seed_data)} seeds)")
    for entry in late_stable[:20]:
        lines.append(f"  {entry['src']:<22} → {entry['tgt']:<22}  "
                     f"mean_peak_ep_idx={entry['mean_peak_idx']:.1f}  "
                     f"peak_std={entry['peak_std']:.2f}  "
                     f"final_asym={entry['mean_final']:.4f}")

    # Save
    report_str = "\n".join(lines)
    print(report_str)
    with open(OUT_DIR / "report.txt", "w") as f:
        f.write(report_str + "\n")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "poi": poi_summary,
            "top_late_stable": late_stable[:50],
            "seeds": sorted(seed_data.keys()),
        }, f, indent=2)

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
