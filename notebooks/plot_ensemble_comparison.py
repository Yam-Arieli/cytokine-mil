"""
Ensemble comparison plot: v2 (h64, g_i) vs v3 (h256, g_i) vs v4 (h256, h_i+decoder weights).
Summarizes seed stability, positive/negative control recovery, and top stable pairs.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# ── Data from aggregate_ensemble.py ─────────────────────────────────────────

# Spearman rho per run vs ensemble mean
rho = {
    "v2\n(64-dim g_i)": [0.5280, 0.5566, 0.5494, 0.6283, 0.6036,
                          0.5254, 0.5667, 0.7585, 0.4539, 0.6941,
                          0.7007, 0.7381, 0.6802, 0.6449, 0.5528],
    "v3\n(256-dim g_i)": [0.7862, 0.7563, 0.6668, 0.6171, 0.5577,
                           0.7995, 0.7919, 0.7566, 0.7679, 0.7583,
                           0.8227, 0.7816, 0.8058, 0.8010, 0.8462],
    "v4\n(256-dim h_i\n+decoder wts)": [0.3951, 0.3248, 0.3793, 0.2516, 0.2469,
                                          0.4011, 0.3830, 0.3395, 0.2745, 0.3008,
                                          0.3420, 0.3085, 0.3685, 0.3945, 0.3279],
}

# Stable pairs in top-50 (appearing in >=7/15 runs)
stable_pair_counts = {"v2\n(64-dim g_i)": 1,
                      "v3\n(256-dim g_i)": 29,
                      "v4\n(256-dim h_i\n+decoder wts)": 0}

# Pairs of interest: (mean_asym, std, snr, rank) out of 8010
poi = {
    "IL-12→IFN-γ\n(positive control)": {
        "v2": (1.84, 1.49, 1.24, 2235),
        "v3": (14.04, 3.11, 4.52, 5135),
        "v4": (29.09, 4.03, 7.22, 2297),
    },
    "IFN-γ→IL-12\n(should be lower)": {
        "v2": (0.93, 0.61, 1.54, 5963),
        "v3": (15.33, 4.17, 3.67, 4109),
        "v4": (25.67, 6.50, 3.95, 6388),
    },
    "IL-6→IL-10\n(negative control,\nshared STAT3)": {
        "v2": (1.92, 1.03, 1.87, 2011),
        "v3": (14.45, 5.51, 2.62, 878),
        "v4": (31.02, 7.67, 4.05, 878),
    },
}

# Top stable pairs in v3 (best version)
v3_stable = [
    ("OX40L", "CD30L",      10, 40.06),
    ("OX40L", "IL-36-alpha",10, 38.87),
    ("LT-α1β2", "IL-18",   10, 38.20),
    ("APRIL", "IL-36-alpha",10, 38.17),
    ("TGF-β1", "IFN-λ1",   10, 39.32),
    ("LT-α1β2", "IL-9",    10, 38.13),
    ("HGF", "IFN-λ1",       8, 38.86),
    ("APRIL", "CD30L",      8, 36.07),
    ("TWEAK", "RANKL",      10, 37.66),
    ("IL-17A", "RANKL",     8, 36.77),
    ("LT-α1β2", "RANKL",    9, 37.07),
    ("HGF", "EGF",          9, 36.10),
]

# ── Figure layout ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor("#f8f9fa")

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.50, wspace=0.38,
                       left=0.07, right=0.97,
                       top=0.93, bottom=0.06)

COLORS = {
    "v2\n(64-dim g_i)":              "#4878cf",
    "v3\n(256-dim g_i)":             "#6acc65",
    "v4\n(256-dim h_i\n+decoder wts)": "#d65f5f",
}
SHORT = {"v2\n(64-dim g_i)": "v2", "v3\n(256-dim g_i)": "v3",
         "v4\n(256-dim h_i\n+decoder wts)": "v4"}

# ── Panel A: Spearman rho boxplots ───────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :2])
labels = list(rho.keys())
data   = [rho[k] for k in labels]
cols   = [COLORS[k] for k in labels]

bp = ax_a.boxplot(data, patch_artist=True, widths=0.45,
                  medianprops=dict(color="white", linewidth=2),
                  whiskerprops=dict(linewidth=1.4),
                  capprops=dict(linewidth=1.4),
                  flierprops=dict(marker="o", markersize=4, alpha=0.5))
for patch, col in zip(bp["boxes"], cols):
    patch.set_facecolor(col)
    patch.set_alpha(0.85)

# Jitter individual points
rng = np.random.default_rng(0)
for i, (d, col) in enumerate(zip(data, cols), 1):
    jitter = rng.uniform(-0.18, 0.18, len(d))
    ax_a.scatter([i + j for j in jitter], d, color=col, s=22,
                 zorder=5, alpha=0.9, edgecolors="white", linewidths=0.5)

ax_a.axhline(0.7, ls="--", lw=1.5, color="#666", alpha=0.6,
             label="ρ = 0.7 stability threshold")
ax_a.set_xticks([1, 2, 3])
ax_a.set_xticklabels([SHORT[k] for k in labels], fontsize=11)
ax_a.set_ylabel("Spearman ρ\n(run vs ensemble mean)", fontsize=10)
ax_a.set_title("A  Seed-to-seed stability of asymmetry rankings", fontsize=12, fontweight="bold", loc="left")
ax_a.set_ylim(0.15, 0.92)
ax_a.legend(fontsize=9, loc="lower right")
ax_a.set_facecolor("white")
ax_a.grid(axis="y", alpha=0.3)

# ── Panel B: Stable pair count ───────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])
vers = [SHORT[k] for k in labels]
cnts = [stable_pair_counts[k] for k in labels]
bars = ax_b.bar(vers, cnts, color=[COLORS[k] for k in labels], alpha=0.85,
                edgecolor="white", linewidth=1.5, width=0.55)
for bar, cnt in zip(bars, cnts):
    ax_b.text(bar.get_x() + bar.get_width()/2, cnt + 0.4,
              str(cnt), ha="center", va="bottom", fontsize=13, fontweight="bold")
ax_b.set_ylabel("# pairs stable in\n≥7/15 runs (top-50)", fontsize=10)
ax_b.set_title("B  Stable pair count", fontsize=12, fontweight="bold", loc="left")
ax_b.set_ylim(0, 35)
ax_b.set_facecolor("white")
ax_b.grid(axis="y", alpha=0.3)

# ── Panel C: Rank of pairs of interest (% rank out of 8010) ─────────────────
ax_c = fig.add_subplot(gs[1, :2])
n_total = 8010
pair_labels = list(poi.keys())
n_pairs = len(pair_labels)
x = np.arange(n_pairs)
bar_w = 0.22
vers_keys = list(labels)

for vi, vk in enumerate(vers_keys):
    ranks_pct = [100.0 * poi[pl][SHORT[vk]][3] / n_total for pl in pair_labels]
    ax_c.bar(x + vi * bar_w - bar_w, ranks_pct,
             bar_w, color=COLORS[vk], alpha=0.85,
             edgecolor="white", linewidth=0.8,
             label=SHORT[vk])

ax_c.axhline(50, ls="--", lw=1.5, color="#999", alpha=0.7, label="50th percentile (random)")
ax_c.set_xticks(x)
ax_c.set_xticklabels(pair_labels, fontsize=9.5)
ax_c.set_ylabel("Rank percentile (lower = stronger)", fontsize=10)
ax_c.set_title("C  Pair-of-interest rank in full ensemble (lower rank = stronger signal)",
               fontsize=12, fontweight="bold", loc="left")
ax_c.set_ylim(0, 100)
ax_c.legend(fontsize=9, loc="upper right")
ax_c.set_facecolor("white")
ax_c.grid(axis="y", alpha=0.3)

# Highlight negative control failure note
ax_c.annotate("Negative control\n(IL-6→IL-10) ranks\ntoo high!",
              xy=(2.22, 100*878/8010), xytext=(2.55, 30),
              fontsize=8, color="#c0392b",
              arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

# ── Panel D: SNR of pairs of interest ───────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 2])
for vi, vk in enumerate(vers_keys):
    snrs = [poi[pl][SHORT[vk]][2] for pl in pair_labels]
    ax_d.bar(np.arange(n_pairs) + vi * bar_w - bar_w, snrs,
             bar_w, color=COLORS[vk], alpha=0.85,
             edgecolor="white", linewidth=0.8,
             label=SHORT[vk])

ax_d.axhline(2.0, ls="--", lw=1.5, color="#666", alpha=0.6, label="SNR = 2")
ax_d.set_xticks(np.arange(n_pairs))
ax_d.set_xticklabels(["IL-12→\nIFN-γ", "IFN-γ→\nIL-12", "IL-6→\nIL-10"], fontsize=8.5)
ax_d.set_ylabel("SNR\n(mean / std across seeds)", fontsize=10)
ax_d.set_title("D  Signal-to-noise ratio\nof pairs of interest", fontsize=12, fontweight="bold", loc="left")
ax_d.legend(fontsize=8, loc="upper left")
ax_d.set_facecolor("white")
ax_d.grid(axis="y", alpha=0.3)

# ── Panel E: Top stable pairs from v3 (horizontal lollipop) ─────────────────
ax_e = fig.add_subplot(gs[2, :])
# Sort by frequency then mean_asym
v3_stable_s = sorted(v3_stable, key=lambda x: (x[2], x[3]), reverse=True)
y_pos = np.arange(len(v3_stable_s))
pair_strs = [f"{a} → {b}" for a, b, _, _ in v3_stable_s]
run_counts = [r for _, _, r, _ in v3_stable_s]
mean_asym  = [m for _, _, _, m in v3_stable_s]

# Color by run_count
cmap = plt.cm.YlOrRd
norm = plt.Normalize(vmin=6, vmax=10)
bar_colors = [cmap(norm(r)) for r in run_counts]

hbars = ax_e.barh(y_pos, mean_asym, color=bar_colors, height=0.65,
                   edgecolor="white", linewidth=0.8)
ax_e.scatter(mean_asym, y_pos, color=[cmap(norm(r)) for r in run_counts],
             s=45, zorder=5, edgecolors="#333", linewidths=0.5)

for yp, rc, ma in zip(y_pos, run_counts, mean_asym):
    ax_e.text(ma + 0.1, yp, f"{rc}/15 runs", va="center", fontsize=7.5,
              color="#333")

ax_e.set_yticks(y_pos)
ax_e.set_yticklabels(pair_strs, fontsize=9)
ax_e.set_xlabel("Mean asymmetry score (ensemble)", fontsize=10)
ax_e.set_title("E  v3 top stable cascade pairs (appear in ≥7/15 runs, sorted by stability then score)",
               fontsize=12, fontweight="bold", loc="left")
ax_e.set_facecolor("white")
ax_e.grid(axis="x", alpha=0.3)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_e, orientation="vertical", fraction=0.015, pad=0.01)
cbar.set_label("# runs\n(of 15)", fontsize=8)

# ── Supertitle ───────────────────────────────────────────────────────────────
fig.suptitle("Ensemble Asymmetry Results: v2 vs v3 vs v4\n"
             "v2 = 64-dim decoder (g_i space)  |  v3 = 256-dim decoder (g_i space)  |  "
             "v4 = 256-dim decoder weights on frozen h_i (Approach F)",
             fontsize=11, y=0.975)

out_path = "/Users/yam/my-packages/cytokine_mil/notebooks/ensemble_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
