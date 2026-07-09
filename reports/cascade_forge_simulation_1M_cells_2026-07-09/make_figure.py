#!/usr/bin/env python
"""Generate the results figure (two panels) for the cascade_forge 1M-cell report.

Panel A: direction accuracy per config — cross_asym vs the (confound-corrected) symmetric
control, with a chance line. Panel B: the effect-size detectability floor.
Colorblind-safe Okabe-Ito palette. Outputs results_figure.pdf next to this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito (colorblind-safe)
BLUE = "#0072B2"      # cross_asym (the method)
ORANGE = "#E69F00"    # symmetric control
GRAY = "#666666"      # chance / reference

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

# ---- data (from results/cascade_forge_large, confound-corrected) ----
configs = ["all\n0.15 t6", "all\n0.20 t6", "all\n0.30 t3", "all\n0.30 t6",
           "all\n0.40 t6", "recep\n0.30 t3", "recep\n0.30 t6"]
cross = np.array([90, 100, 100, 100, 100, 100, 100])
control = np.array([50, 50, 50, 50, 50, 50, 50])   # corrected (~chance), was a fake 100%

eff = np.array([0.15, 0.20, 0.30, 0.40])
eff_cross = np.array([90, 100, 100, 100])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.7, 1]})

# Panel A: grouped bars
x = np.arange(len(configs))
w = 0.38
axA.axhline(50, color=GRAY, ls="--", lw=1.2, zorder=1)
axA.text(len(configs) - 0.5, 52, "chance", color=GRAY, fontsize=8, va="bottom", ha="right")
bA = axA.bar(x - w / 2, cross, w, label="cross_asym (method)", color=BLUE, zorder=2)
bB = axA.bar(x + w / 2, control, w, label="symmetric control", color=ORANGE, zorder=2)
for rect, v in zip(bA, cross):
    axA.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v}", ha="center", va="bottom",
             fontsize=8, color="#222222")
axA.set_xticks(x)
axA.set_xticklabels(configs, fontsize=8)
axA.set_ylim(0, 108)
axA.set_ylabel("direction accuracy (%)")
axA.set_title("(a) Direction: method vs symmetric control", fontsize=10, loc="left")
axA.legend(frameon=False, fontsize=8, loc="lower left", ncol=1)

# Panel B: effect-size floor
axB.axhline(50, color=GRAY, ls="--", lw=1.2, zorder=1)
axB.plot(eff, eff_cross, "-o", color=BLUE, lw=2, ms=7, zorder=3)
for xv, yv in zip(eff, eff_cross):
    axB.text(xv, yv + 1.8, f"{yv}", ha="center", va="bottom", fontsize=8, color="#222222")
axB.set_xticks(eff)
axB.set_ylim(0, 108)
axB.set_xlabel("effect size (log-space program bump)")
axB.set_ylabel("cross_asym accuracy (%)")
axB.set_title("(b) Effect-size floor (all, t6)", fontsize=10, loc="left")
axB.text(0.15, 84, "floor\n~0.15–0.20", fontsize=8, color=GRAY, ha="left", va="top")

fig.tight_layout()
out = Path(__file__).resolve().parent / "results_figure.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
