"""
Learnability ordering analysis — uses existing dynamics.pkl files, no new training.

For each seed:
  - Load dynamics["records"]
  - Per cytokine: aggregate p_correct_trajectory to donor level
    (median across tubes per donor, then mean across donors)
  - Compute AUC of the donor-aggregated trajectory
  - Rank cytokines by AUC

Across seeds:
  - Spearman rho between every pair of seed rankings
  - Stable cytokines: those whose rank is consistent (low std) across stable seeds
  - Check whether top cascade-source cytokines (IFN-beta, IL-27) appear as
    early learners or late learners relative to their cascade targets

Output: results/oesinghaus_full/learnability_analysis/
  - per_seed_rankings.json     per-cytokine AUC and rank for every seed
  - cross_seed_rho.json        Spearman rho matrix between seeds
  - stable_rankings.json       mean rank ± std for stable seeds
  - report.txt                 human-readable summary
"""

import pickle, json, os, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full")
OUT_DIR = BASE / "learnability_analysis"
OUT_DIR.mkdir(exist_ok=True)

SEED_DIRS = {
    42:  BASE / "run_20260412_161758_seed42",
    123: BASE / "run_20260412_161803_seed123",
    7:   BASE / "run_20260412_161803_seed7",
    **{s: BASE / f"new_seeds_seed{s}" for s in [1,2,3,4,5,6,8,9,10]},
}

# Stable seeds from geo ensemble (rho >= 0.7)
STABLE_SEEDS = {1, 2, 5, 6, 7, 8, 9, 10, 42, 123}

# Cascade pairs of interest for cross-referencing
CASCADE_PAIRS = [
    ("IFN-beta",  "IL-27"),    # top hit
    ("IL-27",     "IL-22"),
    ("IL-27",     "IL-35"),
    ("IFN-beta",  "TNF-alpha"),
    ("IL-12",     "IFN-gamma"), # positive control
    ("IL-6",      "IL-10"),     # negative control
]

# ── helpers ───────────────────────────────────────────────────────────────────
def auc_trapz(trajectory):
    """Normalised AUC: trapezoid / (n_steps - 1), so result in [0, 1]."""
    t = np.asarray(trajectory, dtype=float)
    if len(t) < 2:
        return float(t[0]) if len(t) == 1 else 0.0
    return float(np.trapz(t) / (len(t) - 1))


def aggregate_to_donor_level(records):
    """
    For each cytokine:
      1. Group records by donor
      2. Median p_correct_trajectory across tubes within each donor
      3. Mean across donors
    Returns dict: cytokine_name -> donor-aggregated trajectory (np.array)
    """
    # group: cytokine -> donor -> list of trajectories
    cyt_donor = defaultdict(lambda: defaultdict(list))
    for rec in records:
        cyt  = rec.get("cytokine_name") or rec.get("cytokine")
        donor = rec.get("donor")
        traj  = rec.get("p_correct_trajectory")
        if cyt is None or donor is None or traj is None:
            continue
        cyt_donor[cyt][donor].append(np.asarray(traj, dtype=float))

    result = {}
    for cyt, donors in cyt_donor.items():
        donor_medians = []
        for donor, trajs in donors.items():
            mat = np.stack(trajs)           # (n_tubes, n_epochs)
            donor_medians.append(np.median(mat, axis=0))
        result[cyt] = np.mean(np.stack(donor_medians), axis=0)
    return result


def load_dynamics(seed_dir):
    pkl_path = seed_dir / "dynamics.pkl"
    if not pkl_path.exists():
        return None
    print(f"  Loading {pkl_path} ({pkl_path.stat().st_size / 1e9:.2f} GB) ...", flush=True)
    with open(pkl_path, "rb") as fh:
        dyn = pickle.load(fh)
    return dyn


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    per_seed_rankings = {}   # seed -> {cytokine: {auc, rank}}
    seed_auc_vectors  = {}   # seed -> np.array aligned to common cytokine list

    # pass 1: compute per-cytokine AUC for every seed
    for seed, seed_dir in SEED_DIRS.items():
        print(f"\n=== Seed {seed} ===", flush=True)
        if not seed_dir.exists():
            print(f"  Directory not found: {seed_dir}", flush=True)
            continue

        dyn = load_dynamics(seed_dir)
        if dyn is None:
            print(f"  dynamics.pkl not found.", flush=True)
            continue

        records = dyn.get("records", [])
        if not records:
            print(f"  No records found.", flush=True)
            continue

        print(f"  {len(records)} records loaded.", flush=True)
        agg = aggregate_to_donor_level(records)
        print(f"  {len(agg)} cytokines aggregated.", flush=True)

        aucs = {cyt: auc_trapz(traj) for cyt, traj in agg.items()}

        # rank: 1 = highest AUC = most learnable
        sorted_cyts = sorted(aucs, key=lambda c: aucs[c], reverse=True)
        ranks = {cyt: rank+1 for rank, cyt in enumerate(sorted_cyts)}

        per_seed_rankings[seed] = {
            cyt: {"auc": round(aucs[cyt], 6), "rank": ranks[cyt]}
            for cyt in aucs
        }
        print(f"  Top 5: {sorted_cyts[:5]}", flush=True)
        print(f"  Bottom 5: {sorted_cyts[-5:]}", flush=True)

        # free memory
        del dyn, records, agg

    # ── common cytokine set across all loaded seeds ───────────────────────────
    all_cyt_sets = [set(r.keys()) for r in per_seed_rankings.values()]
    common_cyts  = sorted(set.intersection(*all_cyt_sets))
    n_cyts       = len(common_cyts)
    print(f"\nCommon cytokines across all seeds: {n_cyts}", flush=True)

    loaded_seeds = sorted(per_seed_rankings.keys())

    # align rank vectors
    for seed in loaded_seeds:
        seed_auc_vectors[seed] = np.array(
            [per_seed_rankings[seed][c]["auc"] for c in common_cyts])

    # ── cross-seed Spearman rho ───────────────────────────────────────────────
    rho_matrix = {}
    for i, s1 in enumerate(loaded_seeds):
        rho_matrix[s1] = {}
        for s2 in loaded_seeds:
            rho, _ = spearmanr(seed_auc_vectors[s1], seed_auc_vectors[s2])
            rho_matrix[s1][s2] = round(float(rho), 4)

    # ── stable-seed aggregate ranking ────────────────────────────────────────
    stable_loaded = [s for s in loaded_seeds if s in STABLE_SEEDS]
    rank_matrix   = np.stack([
        [per_seed_rankings[s][c]["rank"] for c in common_cyts]
        for s in stable_loaded
    ])  # (n_stable_seeds, n_cyts)

    mean_ranks = rank_matrix.mean(axis=0)
    std_ranks  = rank_matrix.std(axis=0)

    stable_order = np.argsort(mean_ranks)  # index of most-learnable first
    stable_ranking = [
        {
            "cytokine":   common_cyts[i],
            "mean_rank":  round(float(mean_ranks[i]), 2),
            "std_rank":   round(float(std_ranks[i]), 2),
            "mean_auc":   round(float(np.mean([
                per_seed_rankings[s][common_cyts[i]]["auc"] for s in stable_loaded
            ])), 6),
        }
        for i in stable_order
    ]

    # ── cascade pairs cross-reference ─────────────────────────────────────────
    cascade_report = []
    for src, tgt in CASCADE_PAIRS:
        # find matching cytokine names (case-insensitive partial match)
        def find(name):
            exact = [c for c in common_cyts if c.lower() == name.lower()]
            if exact:
                return exact[0]
            partial = [c for c in common_cyts if name.lower() in c.lower()]
            return partial[0] if partial else None

        src_cyt = find(src)
        tgt_cyt = find(tgt)
        if src_cyt is None or tgt_cyt is None:
            cascade_report.append({
                "pair": f"{src} → {tgt}",
                "note": f"Not found: {src if src_cyt is None else tgt}"
            })
            continue

        src_rank = next(r["mean_rank"] for r in stable_ranking if r["cytokine"] == src_cyt)
        tgt_rank = next(r["mean_rank"] for r in stable_ranking if r["cytokine"] == tgt_cyt)
        src_auc  = next(r["mean_auc"]  for r in stable_ranking if r["cytokine"] == src_cyt)
        tgt_auc  = next(r["mean_auc"]  for r in stable_ranking if r["cytokine"] == tgt_cyt)

        cascade_report.append({
            "pair":     f"{src_cyt} → {tgt_cyt}",
            "src_rank": src_rank, "src_auc": round(src_auc, 4),
            "tgt_rank": tgt_rank, "tgt_auc": round(tgt_auc, 4),
            "rank_diff": round(tgt_rank - src_rank, 2),
            "note": "src learned BEFORE tgt (consistent with cascade)" if src_rank < tgt_rank
                    else "src learned AFTER tgt (inconsistent)" if src_rank > tgt_rank
                    else "same rank",
        })

    # ── save JSON outputs ─────────────────────────────────────────────────────
    with open(OUT_DIR / "per_seed_rankings.json", "w") as f:
        json.dump({str(k): v for k, v in per_seed_rankings.items()}, f, indent=2)

    with open(OUT_DIR / "cross_seed_rho.json", "w") as f:
        json.dump({str(k): {str(k2): v2 for k2, v2 in v.items()}
                   for k, v in rho_matrix.items()}, f, indent=2)

    with open(OUT_DIR / "stable_rankings.json", "w") as f:
        json.dump(stable_ranking, f, indent=2)

    # ── human-readable report ─────────────────────────────────────────────────
    lines = []
    W = 70
    lines += ["=" * W,
              "Learnability Ordering Analysis",
              f"Stable seeds: {stable_loaded}",
              f"Common cytokines: {n_cyts}",
              "=" * W, ""]

    lines += ["TOP 20 MOST LEARNABLE CYTOKINES (stable-seed ensemble)", "-" * W]
    for entry in stable_ranking[:20]:
        lines.append(f"  {entry['mean_rank']:6.1f} ± {entry['std_rank']:4.1f}  "
                     f"AUC={entry['mean_auc']:.4f}  {entry['cytokine']}")

    lines += ["", "BOTTOM 10 LEAST LEARNABLE CYTOKINES", "-" * W]
    for entry in stable_ranking[-10:]:
        lines.append(f"  {entry['mean_rank']:6.1f} ± {entry['std_rank']:4.1f}  "
                     f"AUC={entry['mean_auc']:.4f}  {entry['cytokine']}")

    lines += ["", "CROSS-SEED SPEARMAN RHO (learnability ranking)", "-" * W]
    header = "       " + "  ".join(f"s{s:>3}" for s in loaded_seeds)
    lines.append(header)
    for s1 in loaded_seeds:
        row = f"s{s1:>3}  " + "  ".join(
            f"{rho_matrix[s1][s2]:+.3f}" for s2 in loaded_seeds)
        lines.append(row)

    lines += ["", "CASCADE PAIRS: SOURCE vs TARGET LEARNABILITY", "-" * W]
    for entry in cascade_report:
        if "note" in entry and "Not found" in entry["note"]:
            lines.append(f"  {entry['pair']}: {entry['note']}")
        else:
            lines.append(
                f"  {entry['pair']}\n"
                f"    src rank={entry['src_rank']:.1f} (AUC={entry['src_auc']:.4f})  "
                f"tgt rank={entry['tgt_rank']:.1f} (AUC={entry['tgt_auc']:.4f})  "
                f"Δrank={entry['rank_diff']:+.1f}\n"
                f"    → {entry['note']}"
            )

    report_str = "\n".join(lines)
    print("\n" + report_str, flush=True)
    with open(OUT_DIR / "report.txt", "w") as f:
        f.write(report_str + "\n")

    print(f"\nSaved to: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
