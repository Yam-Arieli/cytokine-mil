#!/usr/bin/env python
"""
Candidate figure drafts for the thesis Results "Coupling" subsection.

Message 1: cascade (upstream/downstream) pairs tend to be coupled.
Message 2: the donor-level-null + degree/hub correction is what makes that
           discrimination possible (stricter AND more accurate; provably
           safe -- symmetric, doesn't touch cross_asym direction).

Data sources (all IG_vsPBS, the current adopted default signature variant --
NOT IG_vsPanel, which appears in some older reports and is superseded):
  - donor_coupling_hub_IG_vsPBS.csv / donor_coupling_raw_IG_vsPBS.csv
    (pulled from cluster: results/sig_ablation/oes/donor_coupling/, 276
    Oesinghaus pairs, 10 donors, donor-level sign-flip null)
  - Sheu 3hr numbers hardcoded from results/sig_ablation/sheu3hr/cell_degree/
    cell_degree_report.md (21 pairs, read in full via cluster_cmd this session)
  - 3-stage progression numbers from the same donor_coupling_report.md

Output: PNGs only (exploratory drafts, not final thesis embeds) under this
directory: msg1_oes_violin.png, msg1_sheu_bars.png, msg1_enrichment_summary.png,
msg2_progression.png, msg2_il15_hub_shrink.png, msg2_heatmap_before_after.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    }
)

GREEN = "#2a9d5c"
RED = "#d1495b"
GREY = "#9e9e9e"
BLUE = "#3a6ea5"
ORANGE = "#e08e2a"

hub = pd.read_csv(HERE / "donor_coupling_hub_IG_vsPBS.csv")
raw = pd.read_csv(HERE / "donor_coupling_raw_IG_vsPBS.csv")

DIRECTIONAL = {"DIRECTIONAL_a_to_b", "DIRECTIONAL_b_to_a"}
n_benchmark = hub["pair_status"].isin(DIRECTIONAL).sum()
print(f"Sanity check: {n_benchmark} DIRECTIONAL benchmark pairs found (expect 17).")


def pair_lookup(df: pd.DataFrame, a: str, b: str) -> float:
    row = df[((df.axis_a == a) & (df.axis_b == b)) | ((df.axis_a == b) & (df.axis_b == a))]
    if row.empty:
        raise KeyError(f"pair not found: {a} / {b}")
    return float(row["excess_mean"].iloc[0])


# ============================================================================
# Message 1 — cascade pairs tend to be coupled
# ============================================================================

# ---- 1. Oesinghaus strip/violin: benchmark (17) vs other (259) ----
is_bench = hub["pair_status"].isin(DIRECTIONAL)
bench_vals = hub.loc[is_bench, "excess_mean"].to_numpy()
other_vals = hub.loc[~is_bench, "excess_mean"].to_numpy()

fig, ax = plt.subplots(figsize=(6.5, 5))
parts = ax.violinplot([other_vals], positions=[0], showmedians=True, widths=0.7)
for pc in parts["bodies"]:
    pc.set_facecolor(GREY)
    pc.set_alpha(0.4)
for key in ("cmedians", "cbars", "cmins", "cmaxes"):
    parts[key].set_color(GREY)

rng = np.random.default_rng(0)
jitter = rng.uniform(-0.12, 0.12, size=len(bench_vals))
ax.scatter(
    np.full_like(bench_vals, 1.0) + jitter, bench_vals,
    s=55, color=ORANGE, edgecolor="black", linewidths=0.5, zorder=5,
    label=f"17 known cascade pairs (mean={bench_vals.mean():+.3f})",
)
ax.scatter([], [], s=55, color=GREY, alpha=0.4, label=f"other 259 pairs (mean={other_vals.mean():+.3f})")

ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["other 259 pairs\n(violin = distribution)", "17 known cascade pairs\n(dots, jittered)"])
ax.set_ylabel(r"coupling, hub-corrected  $R_{ab}=C_{ab}-d_a-d_b+g$")
ax.set_title("Oesinghaus: known cascade pairs skew higher\n(hub-corrected coupling, IG$_{vsPBS}$)")
ax.legend(loc="upper left", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(HERE / "msg1_oes_violin.png")
plt.close(fig)
print("Wrote msg1_oes_violin.png")


# ---- 2. Sheu per-pair horizontal bar chart (all 21 pairs, IG_vsPBS, hub) ----
sheu_rows = [
    ("IFNb", "PIC", "MUST/SHOULD", 0.8142, 0.0000),
    ("LPSlo", "P3CSK", "MUST/SHOULD", 0.4002, 0.0000),
    ("CpG", "LPSlo", None, 0.3572, 0.0000),
    ("CpG", "P3CSK", "MUST/SHOULD", 0.2760, 0.0000),
    ("LPSlo", "TNF", None, 0.2469, 0.0000),
    ("P3CSK", "TNF", None, 0.2307, 0.0000),
    ("IFNb", "LPS", "MUST/SHOULD", 0.2208, 0.0000),
    ("CpG", "TNF", None, 0.1011, 0.0000),
    ("LPS", "P3CSK", None, 0.0071, 0.2920),
    ("LPS", "PIC", None, -0.0118, 0.4760),
    ("LPS", "TNF", "MUST/SHOULD", -0.0479, 0.9520),
    ("LPS", "LPSlo", None, -0.0911, 1.0000),
    ("CpG", "LPS", None, -0.1003, 1.0000),
    ("LPSlo", "PIC", None, -0.1553, 1.0000),
    ("IFNb", "TNF", "MUST-NOT", -0.1935, 1.0000),
    ("PIC", "TNF", None, -0.1950, 1.0000),
    ("P3CSK", "PIC", None, -0.1963, 1.0000),
    ("CpG", "PIC", None, -0.2828, 1.0000),
    ("CpG", "IFNb", "MUST-NOT", -0.3529, 1.0000),
    ("IFNb", "LPSlo", None, -0.4860, 1.0000),
    ("IFNb", "P3CSK", "MUST-NOT", -0.5413, 1.0000),
]
sheu = pd.DataFrame(sheu_rows, columns=["a", "b", "label", "coupling_hub", "null_p_hub"])
sheu["pair"] = sheu["a"] + " – " + sheu["b"]
sheu = sheu.sort_values("coupling_hub")
color_map = {"MUST/SHOULD": GREEN, "MUST-NOT": RED, None: GREY}
colors = [color_map[l] for l in sheu["label"]]

fig, ax = plt.subplots(figsize=(7, 6.5))
ax.barh(sheu["pair"], sheu["coupling_hub"], color=colors, edgecolor="black", linewidth=0.4)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel(r"coupling, hub-corrected  ($M_{ab}+M_{ba}$, degree-centered)")
ax.set_title("Sheu 3hr: MUST cascades couple, MUST-NOT pairs don't\n(IG$_{vsPBS}$, all 21 pairs)")
handles = [
    plt.Rectangle((0, 0), 1, 1, color=GREEN, label="MUST/SHOULD (known cascade)"),
    plt.Rectangle((0, 0), 1, 1, color=RED, label="MUST-NOT (no known relationship)"),
    plt.Rectangle((0, 0), 1, 1, color=GREY, label="unlabeled"),
]
ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(HERE / "msg1_sheu_bars.png")
plt.close(fig)
print("Wrote msg1_sheu_bars.png")


# ---- 3. Plain enrichment summary (2 bars) ----
recall_frac = 10 / 17
overcall_frac = 0.3188
fig, ax = plt.subplots(figsize=(5, 5.5))
bars = ax.bar(
    ["known cascade pairs\n(recall)", "any random pair\n(base over-call)"],
    [recall_frac * 100, overcall_frac * 100],
    color=[ORANGE, GREY], edgecolor="black", linewidth=0.6, width=0.55,
)
for b, v in zip(bars, [recall_frac * 100, overcall_frac * 100]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
ax.set_ylabel("% of pairs flagged \"coupled\"")
ax.set_ylim(0, 78)
ax.set_title(f"Known cascade pairs are ≈{recall_frac/overcall_frac:.2f}×\nmore likely to be flagged coupled\n(Oesinghaus, hub-corrected, IG$_{{vsPBS}}$)")
fig.tight_layout()
fig.savefig(HERE / "msg1_enrichment_summary.png")
plt.close(fig)
print("Wrote msg1_enrichment_summary.png")


# ============================================================================
# Message 2 — the hub/degree fix is what makes discrimination possible
# ============================================================================

# ---- 4. Operating-point trajectory (ROC-like): x=recall, y=over-call ----
stages = ["cell-level (no fix)", "+ donor-level null", "+ degree/hub correction"]
overcall = [76.5, 57.6, 31.9]
recall = [8 / 17 * 100, 8 / 17 * 100, 10 / 17 * 100]
point_colors = [GREY, BLUE, GREEN]
# (dx, dy) label offset in points, and horizontal alignment, tuned per point
label_offsets = [(-12, -22, "right"), (-12, 14, "right"), (12, 12, "left")]

fig, ax = plt.subplots(figsize=(7, 6.5))
ax.plot(recall, overcall, "-", color="0.5", linewidth=1.5, alpha=0.6, zorder=1)
for i in range(len(stages) - 1):
    ax.annotate(
        "", xy=(recall[i + 1], overcall[i + 1]), xytext=(recall[i], overcall[i]),
        arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.5, alpha=0.8), zorder=1,
    )
ax.scatter(recall, overcall, s=170, color=point_colors, edgecolor="black", linewidths=1.0, zorder=3)

for i, stage in enumerate(stages):
    dx, dy, ha = label_offsets[i]
    label = f"{stage}\nrecall {recall[i]:.0f}%, over-call {overcall[i]:.0f}%"
    ax.annotate(
        label, (recall[i], overcall[i]), textcoords="offset points", xytext=(dx, dy),
        ha=ha, fontsize=9, fontweight="bold" if i == 2 else "normal",
    )

ax.set_xlim(40, 68)
ax.set_ylim(24, 91)
ax.invert_yaxis()  # up = lower over-call = better, so up-and-right = better
ax.set_xlabel("recall  (% of 17 known cascade pairs recovered)")
ax.set_ylabel("over-call  (% of ALL pairs flagged \"coupled\")")
ax.set_title(
    "Fixing the gate moves toward high-recall, low-over-call\n"
    "(Oesinghaus, IG$_{vsPBS}$; up-right = better)"
)
fig.tight_layout()
fig.savefig(HERE / "msg2_progression.png")
plt.close(fig)
print("Wrote msg2_progression.png")


# ---- 5. IL-15 hub-shrinkage grouped bar chart ----
partners = ["IL-2", "IFN-beta", "IL-1-beta", "IFN-gamma", "IFN-omega", "IL-12"]
raw_vals = [pair_lookup(raw, "IL-15", p) for p in partners]
hub_vals = [pair_lookup(hub, "IL-15", p) for p in partners]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
xw = np.arange(len(partners))
width = 0.35
ax.bar(xw - width / 2, raw_vals, width, color=GREY, edgecolor="black", linewidth=0.4, label="raw (donor-null only)")
ax.bar(xw + width / 2, hub_vals, width, color=BLUE, edgecolor="black", linewidth=0.4, label="hub-corrected")
ax.set_xticks(xw)
ax.set_xticklabels([f"IL-15–{p}" for p in partners], rotation=20, ha="right")
ax.set_ylabel("coupling (excess over random-gene baseline)")
ax.set_title("IL-15 (the hub): every edge shrinks under degree correction\n(IG$_{vsPBS}$)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(HERE / "msg2_il15_hub_shrink.png")
plt.close(fig)
print("Wrote msg2_il15_hub_shrink.png")


# ---- 6. Small real-data heatmap, before/after degree correction ----
cyts = ["IL-15", "IL-2", "IFN-beta", "IL-1-beta", "IFN-gamma", "IFN-omega", "IL-12"]
n = len(cyts)


def build_matrix(df: pd.DataFrame) -> np.ndarray:
    M = np.full((n, n), np.nan)
    for i, a in enumerate(cyts):
        for j, b in enumerate(cyts):
            if i == j:
                continue
            try:
                M[i, j] = pair_lookup(df, a, b)
            except KeyError:
                pass
    return M


M_raw = build_matrix(raw)
M_hub = build_matrix(hub)
vmax = np.nanmax(np.abs(np.concatenate([M_raw.ravel(), M_hub.ravel()])))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, M, title in zip(axes, [M_raw, M_hub], ["raw (donor-null only)", "hub-corrected"]):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cyts, rotation=45, ha="right")
    ax.set_yticklabels(cyts)
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=axes, shrink=0.8, label="coupling")
fig.suptitle("A real version of Fig. doublecenter: IL-15's row/column flattens\n(IG$_{vsPBS}$, representative 7-cytokine subset)", y=1.03)
fig.savefig(HERE / "msg2_heatmap_before_after.png")
plt.close(fig)
print("Wrote msg2_heatmap_before_after.png")

print("\nDONE. 6 candidate figures written to", HERE)
