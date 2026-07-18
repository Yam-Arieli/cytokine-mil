#!/usr/bin/env python
"""Variant of make_sheu_cascade_graph.py, explicitly framed around cross_asym rather than a
vague "coupling score": panel C is a bar chart of |cross_asym| (|cross_median|) for all 21
pairs -- all-positive (no negative bars), height = width = |cross_median| (the same width_fn
used for the panel B graph edges), ordered ascending left-to-right by that magnitude. Since
sign is not on the y-axis, direction is carried entirely by the tick label below each bar,
which is the oriented src -> dst pair (same orientation as panel B's arrows), not the fixed
alphabetical axis_a -> axis_b order.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CSV_5HR = REPO / "results" / "sheu_cascade" / "5hr" / "pathB" / "per_axis_summary.csv"

# Okabe-Ito colorblind-safe palette (matches reports/cascade_forge_simulation_1M_cells_2026-07-09)
BLUE = "#0072B2"     # ground-truth cascade edge
ORANGE = "#E69F00"   # negative-control pair (no cascade expected)
GREEN = "#009E73"    # benchmark-correct direction call
RED = "#D55E00"       # benchmark-wrong direction call
GRAY = "#7f7f7f"      # no literature prior (uncalibrated)

# ---- Node layout (shared by both panels) ----
POS = {
    "CpG": (0.0, 3.0), "P3CSK": (0.9, 4.1), "LPS": (2.4, 3.6), "LPSlo": (2.4, 2.1),
    "PIC": (4.6, 4.1), "TNF": (1.8, 0.2), "IFNb": (3.8, 0.2),
}

# ---- Ground truth (Panel A), from reports/sheu_cascade/sheu_cascade_labels.yaml ----
# (upstream -> downstream, benchmark_class)
BENCHMARK_EDGES = [
    ("CpG", "TNF", "NFKB_SHOULD"), ("LPS", "TNF", "NFKB_SHOULD"),
    ("LPSlo", "TNF", "NFKB_SHOULD"), ("P3CSK", "TNF", "NFKB_SHOULD"),
    ("LPS", "IFNb", "IFN_MUST"), ("PIC", "IFNb", "IFN_MUST"),
    ("LPSlo", "IFNb", "IFN_SHOULD"),
]
NEGATIVE_PAIRS = [
    frozenset(("CpG", "IFNb")), frozenset(("IFNb", "P3CSK")),
    frozenset(("IFNb", "TNF")), frozenset(("LPS", "LPSlo")),
]
BENCHMARK_KEYS = {frozenset((a, b)) for a, b, _ in BENCHMARK_EDGES}


def draw_nodes(ax):
    for n, (x, y) in POS.items():
        ax.add_patch(Circle((x, y), 0.32, facecolor="#dbe9f5", edgecolor="black",
                             linewidth=1.3, zorder=3))
        ax.text(x, y, n, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=4)


def draw_edge(ax, src, dst, color, width, style="-", curve=0.12, alpha=1.0, zorder=2):
    x0, y0 = POS[src]
    x1, y1 = POS[dst]
    patch = FancyArrowPatch(
        (x0, y0), (x1, y1), connectionstyle=f"arc3,rad={curve}",
        arrowstyle="-|>", mutation_scale=14, linewidth=width, linestyle=style,
        color=color, alpha=alpha, zorder=zorder,
        shrinkA=17, shrinkB=17,
    )
    ax.add_patch(patch)


def curve_midpoint(p0, p1, rad):
    """Approximate point at t=0.5 on the same arc3-style quadratic bezier FancyArrowPatch draws."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    cx, cy = mx - rad * dy, my + rad * dx  # perpendicular offset, matplotlib arc3 convention
    return (0.25 * x0 + 0.5 * cx + 0.25 * x1, 0.25 * y0 + 0.5 * cy + 0.25 * y1)


def panel_a(ax):
    draw_nodes(ax)
    class_color = {"NFKB_SHOULD": BLUE, "IFN_MUST": BLUE, "IFN_SHOULD": BLUE}
    class_width = {"NFKB_SHOULD": 1.8, "IFN_MUST": 3.2, "IFN_SHOULD": 2.4}
    for src, dst, cls in BENCHMARK_EDGES:
        draw_edge(ax, src, dst, class_color[cls], class_width[cls])
    rad = 0.15
    for pair in NEGATIVE_PAIRS:
        a, b = tuple(pair)
        patch = FancyArrowPatch(
            POS[a], POS[b], connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
            linewidth=1.2, linestyle=(0, (2, 2)), color=ORANGE, alpha=0.85, zorder=1,
            shrinkA=17, shrinkB=17,
        )
        ax.add_patch(patch)
        mx, my = curve_midpoint(POS[a], POS[b], rad)
        ax.text(mx, my, "✕", ha="center", va="center", color=ORANGE, fontsize=10,
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.05", facecolor="white", edgecolor="none", alpha=0.8))
    ax.set_title("(a) Pre-registered ground truth\n(Sheu TLR cascade wiring, §21/§26)",
                  fontsize=10.5)


# Shared category styling (edges in panel B AND bars in panel C use the same encoding).
STYLE_ORDER = {"no_prior": 0, "negative": 1, "wrong": 2, "correct": 3}
CAT_COLOR = {"correct": GREEN, "wrong": RED, "negative": ORANGE, "no_prior": GRAY}
CAT_DASH = {"correct": "-", "wrong": "-", "negative": (0, (2, 2)), "no_prior": (0, (1, 1.5))}
CAT_ALPHA = {"correct": 1.0, "wrong": 1.0, "negative": 0.85, "no_prior": 0.55}
WMIN, WMAX = 0.6, 4.5


def categorize_rows(df: pd.DataFrame):
    """One (src, dst, cat, cross_median, axis_a, axis_b) tuple per pair. src/dst are oriented
    by cross_asym sign (for panel B's directed edges); axis_a/axis_b are the fixed alphabetical
    order the cross_median sign itself is relative to (for panel C's signed bars)."""
    rows = []
    for _, r in df.iterrows():
        a, b = str(r["axis_a"]), str(r["axis_b"])
        cm = float(r["cross_median"])
        src, dst = (a, b) if cm >= 0 else (b, a)
        key = frozenset((a, b))
        if key in BENCHMARK_KEYS:
            cat = "correct" if bool(r["cross_sign_correct"]) else "wrong"
        elif key in NEGATIVE_PAIRS:
            cat = "negative"
        else:
            cat = "no_prior"
        rows.append((src, dst, cat, cm, a, b))
    return rows


def make_width_fn(rows):
    cvals = np.abs([cm for _, _, _, cm, _, _ in rows])
    cmin, cmax = cvals.min(), cvals.max()

    def width(c):
        t = (abs(c) - cmin) / (cmax - cmin) if cmax > cmin else 0.5
        return WMIN + max(0.0, min(1.0, t)) * (WMAX - WMIN)

    return width


def panel_b(ax, rows, width_fn):
    draw_nodes(ax)
    for src, dst, cat, cm, _, _ in sorted(rows, key=lambda e: STYLE_ORDER[e[2]]):
        draw_edge(ax, src, dst, CAT_COLOR[cat], width_fn(cm), style=CAT_DASH[cat],
                  alpha=CAT_ALPHA[cat], zorder=2 + STYLE_ORDER[cat])

    n_correct = sum(1 for _, _, cat, _, _, _ in rows if cat == "correct")
    n_bench = n_correct + sum(1 for _, _, cat, _, _, _ in rows if cat == "wrong")
    acc = n_correct / n_bench if n_bench else float("nan")
    ax.set_title(f"(b) What cross_asym found (Sheu, 5hr)\n"
                 f"benchmark direction accuracy: {n_correct}/{n_bench} = {acc:.0%}",
                 fontsize=10.5)


def panel_c_bars(ax, rows, width_fn):
    """|cross_asym| bars (no negative values): height = width = |cross_median| (the same
    width_fn as the graph), ordered ascending left-to-right by that magnitude. Sign is not
    dropped, just moved off the y-axis: since all bars now point up, direction is carried
    entirely by the tick label, which is the oriented src -> dst pair (the same orientation
    used for the arrows in panel B), not the fixed alphabetical axis_a -> axis_b."""
    srt = sorted(rows, key=lambda e: abs(e[3]))  # ascending by |cross_median|
    for i, (src, dst, cat, cm, a, b) in enumerate(srt):
        score = abs(cm)
        arrow = FancyArrowPatch(
            (i, 0), (i, score), arrowstyle="-|>", mutation_scale=9,
            linewidth=width_fn(cm), linestyle=CAT_DASH[cat], color=CAT_COLOR[cat],
            alpha=CAT_ALPHA[cat], zorder=2 + STYLE_ORDER[cat],
            shrinkA=0, shrinkB=1,
        )
        ax.add_patch(arrow)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlim(-0.8, len(srt) - 0.2)
    ymax = max(abs(cm) for _, _, _, cm, _, _ in rows)
    label_y = -0.05 * ymax
    for i, (src, dst, cat, cm, a, b) in enumerate(srt):
        ax.text(i, label_y, f"{src}$\\to${dst}", rotation=45, ha="right", va="top",
                fontsize=6.2, color="black", rotation_mode="anchor")
    ax.set_ylim(-0.62 * ymax, 1.08 * ymax)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_ylabel("|cross_asym|  (|cross_median|)", fontsize=9)
    ax.set_title("(c) All 21 pairs, sorted by |cross_asym| (direction carried by the tick label)",
                 fontsize=10.5)


def legend(fig):
    handles = [
        plt.Line2D([0], [0], color=BLUE, lw=2.5, label="ground truth cascade edge"),
        plt.Line2D([0], [0], color=ORANGE, lw=1.5, ls=(0, (2, 2)), label="ground truth: no cascade (negative control)"),
        plt.Line2D([0], [0], color=GREEN, lw=2.5, label="found: correct direction (benchmark pair)"),
        plt.Line2D([0], [0], color=RED, lw=2.5, label="found: wrong direction (benchmark pair)"),
        plt.Line2D([0], [0], color=GRAY, lw=1.5, ls=(0, (1, 1.5)), label="found: no literature prior (uncalibrated)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))


def main():
    df = pd.read_csv(CSV_5HR)
    rows = categorize_rows(df)
    width_fn = make_width_fn(rows)

    fig = plt.figure(figsize=(12, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[5.6, 3.0], hspace=0.32, wspace=0.05)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])
    for ax in (axA, axB):
        ax.set_xlim(-0.8, 5.4)
        ax.set_ylim(-0.6, 4.9)
        ax.set_aspect("equal")
        ax.axis("off")
    panel_a(axA)
    panel_b(axB, rows, width_fn)
    panel_c_bars(axC, rows, width_fn)
    legend(fig)
    fig.suptitle("Sheu 2024 BMDM: real-data cascade-direction validation\n"
                 "(bars explicitly scaled by |cross_asym|)", fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    out_png = HERE / "sheu_cascade_graph_crossasym.png"
    out_pdf = HERE / "sheu_cascade_graph_crossasym.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    n_correct = sum(1 for _, _, cat, _, _, _ in rows if cat == "correct")
    n_wrong = sum(1 for _, _, cat, _, _, _ in rows if cat == "wrong")
    n_neg = sum(1 for _, _, cat, _, _, _ in rows if cat == "negative")
    n_np = sum(1 for _, _, cat, _, _, _ in rows if cat == "no_prior")
    print(f"benchmark pairs: {n_correct + n_wrong}  correct={n_correct}  wrong={n_wrong}")
    print(f"negative-control pairs (no cascade expected): {n_neg}")
    print(f"no-literature-prior pairs (uncalibrated): {n_np}")
    print(f"[wrote] {out_png}")
    print(f"[wrote] {out_pdf}")


if __name__ == "__main__":
    main()
