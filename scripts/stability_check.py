"""Quick stability check: Spearman rho of asymmetry matrices across seeds."""
import pickle
import numpy as np
from scipy.stats import spearmanr

RESULTS = "/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full"

paths = {
    42:  f"{RESULTS}/run_20260412_161758_seed42/experiment3/latent_geometry.pkl",
    123: f"{RESULTS}/run_20260412_161803_seed123/experiment3/latent_geometry.pkl",
    7:   f"{RESULTS}/run_20260412_161803_seed7/experiment3/latent_geometry.pkl",
}

mats = {}
cyt_names = None
for seed, path in paths.items():
    with open(path, "rb") as f:
        r = pickle.load(f)
    m = r["asymmetry"]["asymmetry_matrix"]
    cyt_names = r["asymmetry"]["cytokine_names"]
    K = m.shape[0]
    mask = ~np.eye(K, dtype=bool)
    mats[seed] = m
    off = m[mask]
    print(f"seed={seed}  shape={m.shape}  off-diag: [{off.min():.2f}, {off.max():.2f}]  mean={off.mean():.3f}")

K = list(mats.values())[0].shape[0]
mask = ~np.eye(K, dtype=bool)

print("\nSpearman rho (off-diagonal asymmetry, flattened):")
pairs = [(42, 123), (42, 7), (123, 7)]
for s1, s2 in pairs:
    v1 = mats[s1][mask]
    v2 = mats[s2][mask]
    rho, pval = spearmanr(v1, v2)
    print(f"  seed {s1} vs {s2}: rho={rho:.4f}  p={pval:.2e}")

print("\nTop-50 pair overlap:")
top50s = {}
for seed, m in mats.items():
    flat = m[mask]
    # Get row, col indices of top 50 off-diagonal entries
    rows, cols = np.where(mask)
    order = np.argsort(flat)[-50:]
    top_pairs = set(zip(rows[order].tolist(), cols[order].tolist()))
    top50s[seed] = top_pairs
for s1, s2 in pairs:
    overlap = len(top50s[s1] & top50s[s2])
    print(f"  seed {s1} vs {s2}: {overlap}/50")

print("\nPairs in ALL three top-50:")
common = top50s[42] & top50s[123] & top50s[7]
print(f"  count: {len(common)}")
for (a, b) in sorted(common):
    print(f"  {cyt_names[a]:<25} -> {cyt_names[b]:<25}  "
          f"asym: {mats[42][a,b]:.2f} / {mats[123][a,b]:.2f} / {mats[7][a,b]:.2f}")
