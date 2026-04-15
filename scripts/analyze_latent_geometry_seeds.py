"""
Cross-seed stability analysis for latent geometry results.

Loads latent_geometry_results.pkl from 3 seed run directories,
computes Spearman rho of asymmetry vectors across seeds, and
prints a final cascade candidate report with stability filtering.

Usage:
    python scripts/analyze_latent_geometry_seeds.py \
        --result_dirs results/oesinghaus_full/run_20260412_161758_seed42 \
                      results/oesinghaus_full/run_20260412_161803_seed123 \
                      results/oesinghaus_full/run_20260412_161803_seed7
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

MIN_SEED_RHO = 0.7  # per CLAUDE.md Section 13

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dirs", nargs="+", required=True)
    return p.parse_args()


def _load(result_dir: str) -> dict:
    path = Path(result_dir) / "latent_geometry_results.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _asym_vector(result: dict) -> np.ndarray:
    """Flatten asymmetry matrix to a 1-D vector (upper triangle, excl. diagonal)."""
    A = result["asym_result"]["asymmetry_matrix"]
    K = A.shape[0]
    idx = np.triu_indices(K, k=1)
    return A[idx]


def _pairwise_rho(vectors: list) -> list:
    """Spearman rho for all pairs in a list of vectors."""
    n = len(vectors)
    rhos = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, pval = stats.spearmanr(vectors[i], vectors[j])
            rhos.append((i, j, rho, pval))
    return rhos


def main():
    args = _parse_args()
    results = [_load(d) for d in args.result_dirs]
    n_seeds = len(results)

    print(f"Loaded {n_seeds} seed results.")
    for i, (d, r) in enumerate(zip(args.result_dirs, results)):
        gate = "PASS" if r["gate_pass"] else "FAIL"
        print(f"  [{i}] {d}  gate={gate}")

    # ------------------------------------------------------------------
    # Gate check: all seeds must pass Exp 0
    # ------------------------------------------------------------------
    gates = [r["gate_pass"] for r in results]
    if not all(gates):
        failed = [args.result_dirs[i] for i, g in enumerate(gates) if not g]
        print(f"\nWARNING: {len(failed)} seed(s) failed Exp 0 gate:")
        for d in failed:
            print(f"  {d}")
        print("  Cross-seed analysis continues but results may not be meaningful.")
    else:
        print("\nAll seeds passed Exp 0 gate. Cytokine geometry confirmed.")

    # ------------------------------------------------------------------
    # Cross-seed Spearman rho of asymmetry vectors
    # ------------------------------------------------------------------
    vectors = [_asym_vector(r) for r in results]
    rhos = _pairwise_rho(vectors)

    print()
    print("=" * 55)
    print("Cross-seed Spearman rho — Asymmetry Matrix")
    print("=" * 55)
    for i, j, rho, pval in rhos:
        stable = "STABLE" if rho >= MIN_SEED_RHO else "UNSTABLE"
        print(f"  Seed {i} vs Seed {j}: rho={rho:.3f}  p={pval:.3e}  → {stable}")
    mean_rho = np.mean([r for _, _, r, _ in rhos])
    print(f"\n  Mean pairwise rho: {mean_rho:.3f} (threshold: {MIN_SEED_RHO})")

    # ------------------------------------------------------------------
    # Consensus asymmetry: mean across seeds
    # ------------------------------------------------------------------
    names = results[0]["asym_result"]["cytokine_names"]
    K = len(names)
    all_A = np.stack([r["asym_result"]["asymmetry_matrix"] for r in results], axis=0)
    mean_A = all_A.mean(axis=0)      # (K, K)
    std_A  = all_A.std(axis=0)       # (K, K)

    # ------------------------------------------------------------------
    # Consensus cascade candidates: high mean ASYM, low std, FDR-sig in all seeds
    # ------------------------------------------------------------------
    fdr_alpha = results[0]["fdr_alpha"]

    # For each pair, find if FDR-significant in all seeds
    def all_seeds_sig(a_name, b_name):
        for r in results:
            q = r["bias_result"]["q_values"]
            cell_types = {k[2] for k in q if k[0] == a_name and k[1] == b_name}
            if not cell_types:
                return False
            min_q = min(q.get((a_name, b_name, ct), 1.0) for ct in cell_types)
            if min_q > fdr_alpha:
                return False
        return True

    print()
    print("=" * 65)
    print("Consensus Cascade Candidates (mean ASYM > 0, all-seed FDR < 0.05)")
    print("=" * 65)
    print(f"{'Source':<20} {'Target':<20} {'MeanASYM':>9} {'StdASYM':>8} {'AllSig':>7}")
    print("-" * 67)

    candidates = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            if mean_A[i, j] > 0:
                sig = all_seeds_sig(names[i], names[j])
                candidates.append((names[i], names[j],
                                    float(mean_A[i, j]), float(std_A[i, j]), sig))

    candidates.sort(key=lambda x: x[2], reverse=True)
    for src, tgt, m, s, sig in candidates[:25]:
        flag = "YES" if sig else "no"
        print(f"  {src:<20} {tgt:<20} {m:>9.4f} {s:>8.4f} {flag:>7}")

    # ------------------------------------------------------------------
    # Controls summary
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Controls Summary")
    print("=" * 55)

    def idx(name):
        return names.index(name) if name in names else None

    il12 = idx("IL-12")
    ifng = idx("IFN-gamma")
    il6  = idx("IL-6")
    il10 = idx("IL-10")

    if il12 is not None and ifng is not None:
        fwd = mean_A[il12, ifng]
        rev = mean_A[ifng, il12]
        correct = fwd > 0 and fwd > rev
        print(f"  IL-12 → IFN-γ: mean ASYM = {fwd:+.4f}  (reverse: {rev:+.4f})  "
              f"direction_correct={correct}")

    if il6 is not None and il10 is not None:
        fwd = mean_A[il6, il10]
        rev = mean_A[il10, il6]
        symmetric = abs(fwd - rev) < 0.01
        print(f"  IL-6 / IL-10:  ASYM(6→10)={fwd:+.4f}  ASYM(10→6)={rev:+.4f}  "
              f"near_symmetric={symmetric}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
