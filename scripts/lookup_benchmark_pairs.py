"""
lookup_benchmark_pairs.py

Reads existing latent_geometry.pkl files (no retraining) and reports
legacy asymmetry rank/score + refined Wilcoxon -log10(p) for the
ground-truth benchmark pairs defined in CLAUDE.md Section 21.

Output: results/oesinghaus_full/geo_refined_comparison/benchmark_pairs.json
"""

import json
import pickle
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full")
OUT_FILE    = RESULTS_DIR / "geo_refined_comparison" / "benchmark_pairs.json"

OFFSET0_SEEDS = {
    42:  "run_20260412_161758_seed42",
    123: "run_20260412_161803_seed123",
    7:   "run_20260412_161803_seed7",
    **{s: f"new_seeds_seed{s}" for s in [1, 2, 3, 4, 5, 6, 8, 9, 10]},
}
OFFSET1_SEEDS = {s: f"offset1_seed{s}" for s in range(11, 23)}

# (A, B, role, confidence, expected_direction)
#   expected_direction: "A->B" = A should rank higher than B->A
#                       "symmetric" = both directions should be similar
BENCHMARK_PAIRS = [
    # Existing controls
    ("IL-12",     "IFN-gamma", "positive control",          "HIGH",   "A->B"),
    ("IL-6",      "IL-10",     "negative control (STAT3)",  "NEG",    "symmetric"),
    # New benchmark pairs
    ("IL-1beta",  "IL-6",      "monocyte cascade",          "HIGH",   "A->B"),
    ("TNF-alpha", "IL-6",      "monocyte cascade",          "HIGH",   "A->B"),
    ("IL-32-beta","TNF-alpha", "monocyte cascade",          "HIGH",   "A->B"),
    ("IL-15",     "IFN-gamma", "NK cascade",                "HIGH",   "A->B"),
    ("IL-18",     "IFN-gamma", "NK cascade (synergy)",      "HIGH",   "A->B"),
    ("IFN-gamma", "IL-12",     "positive feedback",         "HIGH",   "A->B"),
    ("IL-33",     "IL-13",     "ILC2 cascade",              "HIGH*",  "A->B"),
    ("IL-27",     "IL-10",     "Tr1 induction",             "MEDIUM", "A->B"),
    ("IL-2",      "IFN-gamma", "NK cascade (noisy)",        "MEDIUM", "A->B"),
]

PBS_LABEL = "PBS"


def _load_pkl(run_dir: Path):
    p = run_dir / "experiment_geo_pbs_rel" / "latent_geometry.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _find_name(names, query):
    """Case-insensitive fuzzy match for cytokine name."""
    query_lower = query.lower().replace("-", "").replace("_", "").replace(" ", "")
    for n in names:
        n_norm = n.lower().replace("-", "").replace("_", "").replace(" ", "")
        if n_norm == query_lower:
            return n
    # Try partial match
    for n in names:
        n_norm = n.lower().replace("-", "").replace("_", "").replace(" ", "")
        if query_lower in n_norm or n_norm in query_lower:
            return n
    return None


def _query_seed(data, pairs_ab):
    """
    For each (A, B) pair, return:
      legacy_asym, legacy_rank, legacy_pct,
      refined_neg_log_p_fwd, refined_neg_log_p_rev,
      cascade_call (Bonferroni-only, alpha=0.05)
    """
    names  = data["asymmetry"]["cytokine_names"]
    matrix = data["asymmetry"]["asymmetry_matrix"]
    K      = len(names)
    pbs_idx = next((i for i, n in enumerate(names) if n == PBS_LABEL), None)

    # Build flat legacy vector (non-PBS pairs only) for rank computation
    flat_vals = []
    flat_keys = []
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            if pbs_idx is not None and (a == pbs_idx or b == pbs_idx):
                continue
            flat_keys.append((names[a], names[b]))
            flat_vals.append(float(matrix[a, b]))
    flat_vals = np.array(flat_vals)
    n_total   = len(flat_vals)

    # Refined: build per-pair min Bonferroni p
    p_pair_fwd = {}
    p_pair_rev = {}
    if data.get("refined") is not None:
        sig = data["refined"]["significance"]
        p_fwd_bonf = sig.get("p_fwd_bonf", {})
        p_rev_bonf = sig.get("p_rev_bonf", {})
        for (a, b, ct), p in p_fwd_bonf.items():
            if p < p_pair_fwd.get((a, b), 1.0):
                p_pair_fwd[(a, b)] = p
        for (a, b, ct), p in p_rev_bonf.items():
            if p < p_pair_rev.get((a, b), 1.0):
                p_pair_rev[(a, b)] = p

    results = {}
    for (A_query, B_query) in pairs_ab:
        A = _find_name(names, A_query)
        B = _find_name(names, B_query)
        if A is None or B is None:
            results[(A_query, B_query)] = {
                "error": f"name not found: A={A_query}->{A}, B={B_query}->{B}",
                "available_names_sample": names[:10],
            }
            continue

        a_idx = names.index(A)
        b_idx = names.index(B)

        # Legacy
        asym_fwd = float(matrix[a_idx, b_idx])
        asym_rev = float(matrix[b_idx, a_idx])
        rank_fwd = int(np.sum(flat_vals > asym_fwd)) + 1
        rank_rev = int(np.sum(flat_vals > asym_rev)) + 1
        pct_fwd  = round(rank_fwd / n_total * 100, 2)
        pct_rev  = round(rank_rev / n_total * 100, 2)

        # Refined
        p_fwd = p_pair_fwd.get((A, B), 1.0)
        p_rev = p_pair_fwd.get((B, A), 1.0)  # forward test for (B,A) is the reverse probe
        neg_log_p_fwd = float(-np.log10(max(p_fwd, 1e-300)))
        neg_log_p_rev = float(-np.log10(max(p_rev, 1e-300)))

        # Cascade call (Bonferroni-only, alpha=0.05)
        alpha = 0.05
        fwd_sig = p_fwd <= alpha
        rev_sig = p_rev <= alpha
        if fwd_sig and not rev_sig:
            call = "A->B"
        elif rev_sig and not fwd_sig:
            call = "B->A"
        elif fwd_sig and rev_sig:
            call = "shared"
        else:
            call = "none"

        results[(A_query, B_query)] = {
            "resolved_A": A, "resolved_B": B,
            "legacy_asym_fwd":  round(asym_fwd, 4),
            "legacy_asym_rev":  round(asym_rev, 4),
            "legacy_rank_fwd":  rank_fwd,
            "legacy_rank_rev":  rank_rev,
            "legacy_pct_fwd":   pct_fwd,
            "legacy_pct_rev":   pct_rev,
            "n_total_pairs":    n_total,
            "refined_neg_log_p_fwd": round(neg_log_p_fwd, 4),
            "refined_neg_log_p_rev": round(neg_log_p_rev, 4),
            "cascade_call_bonf": call,
        }
    return results


def _process_batch(seed_map, batch_name, pairs_ab):
    print(f"\n{'='*60}")
    print(f"  {batch_name}")
    print(f"{'='*60}")
    batch_results = {}
    for seed, dirname in sorted(seed_map.items()):
        run_dir = RESULTS_DIR / dirname
        data = _load_pkl(run_dir)
        if data is None:
            print(f"  [SKIP] seed {seed}: pkl not found at {run_dir}")
            continue
        print(f"  seed {seed} ... ", end="", flush=True)
        seed_res = _query_seed(data, pairs_ab)
        batch_results[str(seed)] = seed_res
        print("ok")
    return batch_results


def _summarize(all_seeds_results, pairs_ab):
    """Compute mean rank/pct and fraction of seeds calling A->B per pair."""
    summary = {}
    for (A, B) in pairs_ab:
        key = (A, B)
        per_seed = []
        for seed, seed_res in all_seeds_results.items():
            r = seed_res.get(key)
            if r is None or "error" in r:
                continue
            per_seed.append(r)
        if not per_seed:
            summary[f"{A}->{B}"] = {"error": "no valid seeds"}
            continue

        mean_legacy_pct_fwd = np.mean([r["legacy_pct_fwd"] for r in per_seed])
        mean_legacy_pct_rev = np.mean([r["legacy_pct_rev"] for r in per_seed])
        mean_refined_fwd    = np.mean([r["refined_neg_log_p_fwd"] for r in per_seed])
        mean_refined_rev    = np.mean([r["refined_neg_log_p_rev"] for r in per_seed])
        n_call_AB   = sum(1 for r in per_seed if r["cascade_call_bonf"] == "A->B")
        n_call_BA   = sum(1 for r in per_seed if r["cascade_call_bonf"] == "B->A")
        n_call_shared = sum(1 for r in per_seed if r["cascade_call_bonf"] == "shared")
        n_call_none = sum(1 for r in per_seed if r["cascade_call_bonf"] == "none")
        n_seeds     = len(per_seed)

        summary[f"{A}->{B}"] = {
            "n_seeds": n_seeds,
            "mean_legacy_pct_fwd":  round(float(mean_legacy_pct_fwd), 2),
            "mean_legacy_pct_rev":  round(float(mean_legacy_pct_rev), 2),
            "mean_refined_neg_log_p_fwd": round(float(mean_refined_fwd), 4),
            "mean_refined_neg_log_p_rev": round(float(mean_refined_rev), 4),
            "call_counts": {
                "A->B": n_call_AB,
                "B->A": n_call_BA,
                "shared": n_call_shared,
                "none": n_call_none,
            },
            "dominant_call": max(["A->B", "B->A", "shared", "none"],
                                  key=lambda c: {"A->B": n_call_AB, "B->A": n_call_BA,
                                                 "shared": n_call_shared, "none": n_call_none}[c]),
        }
    return summary


def main():
    pairs_ab = [(A, B) for (A, B, *_) in BENCHMARK_PAIRS]

    b0 = _process_batch(OFFSET0_SEEDS, "Batch 0 (δ=0)", pairs_ab)
    b1 = _process_batch(OFFSET1_SEEDS, "Batch 1 (δ=1)", pairs_ab)

    all_seeds = {**b0, **b1}
    summary   = _summarize(all_seeds, pairs_ab)

    # Also build per-batch summaries
    summary_b0 = _summarize(b0, pairs_ab)
    summary_b1 = _summarize(b1, pairs_ab)

    output = {
        "benchmark_pairs": [
            {"A": A, "B": B, "role": role, "confidence": conf, "expected": exp}
            for (A, B, role, conf, exp) in BENCHMARK_PAIRS
        ],
        "summary_all_seeds": summary,
        "summary_batch0":    summary_b0,
        "summary_batch1":    summary_b1,
        "per_seed_batch0":   {s: {f"{A}->{B}": v for (A, B), v in res.items()}
                              for s, res in b0.items()},
        "per_seed_batch1":   {s: {f"{A}->{B}": v for (A, B), v in res.items()}
                              for s, res in b1.items()},
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nSummary across all seeds:")
    print(f"{'Pair':<30} {'Exp':>8} {'LegacyFwd%':>12} {'LegacyRev%':>12} "
          f"{'RefinedFwd':>12} {'DomCall':>10}")
    print("-" * 90)
    for A, B, role, conf, exp in BENCHMARK_PAIRS:
        s = summary.get(f"{A}->{B}", {})
        if "error" in s:
            print(f"  {A}->{B}: ERROR - {s['error']}")
            continue
        print(f"  {A}->{B:<22} {exp:>8}  "
              f"{s['mean_legacy_pct_fwd']:>10.1f}%  "
              f"{s['mean_legacy_pct_rev']:>10.1f}%  "
              f"{s['mean_refined_neg_log_p_fwd']:>10.3f}  "
              f"{s['dominant_call']:>10}  "
              f"(n={s['n_seeds']})")

    print(f"\nSaved to: {OUT_FILE}")


if __name__ == "__main__":
    main()
