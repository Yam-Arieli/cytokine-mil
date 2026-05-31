"""
Generate the DATA figures (F9-F13) for the group-talk LaTeX report.

Reads only local CSVs produced this session:
  - results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv   (Oesinghaus)
  - reports/cascade_pairs/cytokine_axes_audited.csv                     (Oes labels)
  - results/sheu_cascade/{1,3,5}hr/pathB/per_celltype.csv               (Sheu)
  - reports/sheu_cascade/sheu_axes_labeled.csv                          (Sheu labels)

cross_asym = sA_PB_norm - sB_PA_norm  (antisymmetric, direction-bearing).
The Oesinghaus per_celltype is the pre-cross_asym run, so we derive the column;
the Sheu per_celltype already has it.

Outputs PDF + a console sanity print (must match: Oes 15/17=88% vs 8/17=47%;
Sheu 5hr 6/7).  matplotlib only — this is a LOCAL reporting script, not a
cluster job, so matplotlib is allowed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
FIGDIR = REPO / "reports" / "group_talk_2026-06" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

OES_PC = REPO / "results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv"
OES_LAB = REPO / "reports/cascade_pairs/cytokine_axes_audited.csv"
SHEU_PC = {t: REPO / f"results/sheu_cascade/{t}/pathB/per_celltype.csv" for t in ("1hr", "3hr", "5hr")}
SHEU_LAB = REPO / "reports/sheu_cascade/sheu_axes_labeled.csv"

GREEN, RED, BLUE, GREY = "#2a9d4a", "#c0392b", "#2c6fbb", "#888888"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.bbox": "tight"})


def _cross(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cross_asym" not in df.columns:
        df["cross_asym"] = df["sA_PB_norm"] - df["sB_PA_norm"]
    return df


def _agg(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for (a, b), g in df.groupby(["axis_a", "axis_b"]):
        s = g[col].to_numpy()
        med = float(np.median(s))
        cons = (np.sum(s > 0) if med > 0 else np.sum(s < 0)) / len(s) if len(s) else np.nan
        rows.append({"axis_a": a, "axis_b": b, f"{col}_med": med, f"{col}_cons": cons})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def fig9_oes_headline():
    """F9: cross_asym vs directional_score headline accuracy on the 17 audited axes."""
    pc = _cross(pd.read_csv(OES_PC))
    lab = pd.read_csv(OES_LAB)
    bench = lab[lab["counts_in_benchmark"].astype(str).str.lower() == "true"].copy()
    bench["exp"] = bench["expected_sign"].astype(int)
    ca = _agg(pc, "cross_asym").merge(bench, on=["axis_a", "axis_b"])
    ds = _agg(pc, "directional_score").merge(bench, on=["axis_a", "axis_b"])
    ca_acc = (np.sign(ca["cross_asym_med"]).astype(int) == ca["exp"]).mean()
    ds_acc = (np.sign(ds["directional_score_med"]).astype(int) == ds["exp"]).mean()
    n = len(ca)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(["directional_score\n(symmetric §24 scalar)", "cross_asym\n(antisymmetric)"],
                  [ds_acc * 100, ca_acc * 100], color=[GREY, BLUE], width=0.6, zorder=3)
    ax.axhline(50, ls="--", color=RED, lw=1.2, zorder=2)
    ax.text(1.45, 51.5, "chance (50%)", color=RED, fontsize=9, ha="right")
    for bar, acc, k in zip(bars, [ds_acc, ca_acc], [ds, ca]):
        ncorr = int(round(acc * n))
        ax.text(bar.get_x() + bar.get_width() / 2, acc * 100 + 1.5,
                f"{ncorr}/{n}\n{acc*100:.0f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("directional sign accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Oesinghaus 24h — same data, same signatures,\nonly the aggregation metric differs", fontsize=11)
    fig.savefig(FIGDIR / "f9_oes_headline.pdf")
    plt.close(fig)
    return ds_acc, ca_acc, n


# --------------------------------------------------------------------------- #
def fig10_oes_peraxis():
    """F10: per-axis median cross_asym on the 17 benchmark axes, correct/wrong."""
    pc = _cross(pd.read_csv(OES_PC))
    lab = pd.read_csv(OES_LAB)
    bench = lab[lab["counts_in_benchmark"].astype(str).str.lower() == "true"].copy()
    bench["exp"] = bench["expected_sign"].astype(int)
    ca = _agg(pc, "cross_asym").merge(bench, on=["axis_a", "axis_b"])
    ca["correct"] = np.sign(ca["cross_asym_med"]).astype(int) == ca["exp"]
    ca["label"] = ca["axis_a"] + " / " + ca["axis_b"]
    ca = ca.sort_values("cross_asym_med")

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    colors = [GREEN if c else RED for c in ca["correct"]]
    ax.barh(ca["label"], ca["cross_asym_med"], color=colors, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    for y, (_, r) in enumerate(ca.iterrows()):
        # expected-direction tick: a small arrow marker on the expected side
        ax.plot(0, y, marker=("4" if r["exp"] > 0 else "3"), color="black", ms=9, zorder=4)
    ax.set_xlabel("median cross_asym across cell types  (sign = inferred direction)")
    ax.set_title("Oesinghaus: per-axis direction call (17 audited cascades)\n"
                 "green = correct sign, red = wrong; ▸/◂ = expected direction", fontsize=10)
    ax.legend(handles=[Patch(color=GREEN, label="correct"), Patch(color=RED, label="wrong")],
              loc="lower right", frameon=False)
    fig.savefig(FIGDIR / "f10_oes_peraxis.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _sheu_bench():
    lab = pd.read_csv(SHEU_LAB)
    bench = lab[lab["counts_in_benchmark"].astype(str).str.lower() == "true"].copy()
    bench["exp"] = bench["expected_sign"].astype(float).astype(int)
    per_t = {}
    for t, p in SHEU_PC.items():
        df = _cross(pd.read_csv(p))
        agg = _agg(df, "cross_asym").merge(bench, on=["axis_a", "axis_b"])
        agg["correct"] = np.sign(agg["cross_asym_med"]).astype(int) == agg["exp"]
        per_t[t] = agg
    return bench, per_t


def fig11_sheu_kinetic():
    """F11: cross_asym(1/3/5h) per benchmark pair; 0 is the decision boundary."""
    bench, per_t = _sheu_bench()
    times = ["1hr", "3hr", "5hr"]
    x = [1, 3, 5]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for _, r in bench.iterrows():
        a, b, exp, cls = r["axis_a"], r["axis_b"], r["exp"], r["benchmark_class"]
        ys = []
        for t in times:
            row = per_t[t][(per_t[t]["axis_a"] == a) & (per_t[t]["axis_b"] == b)]
            ys.append(float(row["cross_asym_med"].iloc[0]) if len(row) else np.nan)
        ls = "-" if cls.startswith("IFN") else ":"
        lw = 2.4 if cls.startswith("IFN") else 1.4
        ax.plot(x, ys, ls=ls, lw=lw, marker="o", ms=5,
                label=f"{a}/{b} ({'+' if exp>0 else '−'}) [{cls.replace('_',' ')}]")
    ax.axhline(0, color="black", lw=1.0)
    ax.text(5.05, 0.005, "decision boundary", fontsize=8, va="bottom")
    ax.set_xticks(x)
    ax.set_xlabel("time point (single-frame; each point uses only that frame)")
    ax.set_ylabel("median cross_asym")
    ax.set_title("Sheu BMDM: cross_asym kinetics for the 7 benchmark cascades", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left", ncol=1, frameon=False, bbox_to_anchor=(1.01, 1.0))
    fig.savefig(FIGDIR / "f11_sheu_kinetic.pdf")
    plt.close(fig)


def fig12_sheu_heatmap():
    """F12: directional accuracy heatmap (class x time)."""
    _, per_t = _sheu_bench()
    times = ["1hr", "3hr", "5hr"]
    classes = ["IFN_MUST", "IFN_SHOULD", "NFKB_SHOULD"]
    M = np.full((len(classes), len(times)), np.nan)
    txt = [["" for _ in times] for _ in classes]
    for j, t in enumerate(times):
        g = per_t[t]
        for i, c in enumerate(classes):
            sub = g[g["benchmark_class"] == c]
            if len(sub):
                frac = sub["correct"].mean()
                M[i, j] = frac
                txt[i][j] = f"{int(sub['correct'].sum())}/{len(sub)}"
            else:
                txt[i][j] = "—"
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(times)), times)
    ax.set_yticks(range(len(classes)), [c.replace("_", " ") for c in classes])
    for i in range(len(classes)):
        for j in range(len(times)):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontweight="bold",
                    color="black", fontsize=12)
    ax.set_title("Sheu single-frame: directional accuracy by class × time", fontsize=11)
    fig.colorbar(im, ax=ax, label="fraction correct", shrink=0.8)
    fig.savefig(FIGDIR / "f12_sheu_heatmap.pdf")
    plt.close(fig)


def fig13_polyic():
    """F13: polyIC/IFNb cross_asym per cell type x time — the consistent failure."""
    times = ["1hr", "3hr", "5hr"]
    data = {}
    cts = None
    for t in times:
        df = _cross(pd.read_csv(SHEU_PC[t]))
        sub = df[(df["axis_a"] == "IFNb") & (df["axis_b"] == "PIC")].sort_values("cell_type")
        data[t] = dict(zip(sub["cell_type"], sub["cross_asym"]))
        cts = list(sub["cell_type"]) if cts is None else cts
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(cts))
    w = 0.26
    for k, t in enumerate(times):
        vals = [data[t].get(ct, np.nan) for ct in cts]
        ax.bar(x + (k - 1) * w, vals, width=w, label=t,
               color=[RED if v > 0 else GREEN for v in vals], alpha=0.55 + 0.15 * k)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, cts)
    ax.set_ylabel("cross_asym (IFNb, PIC)")
    ax.set_title("polyIC → IFNb: expected −, observed + at every cell type & time\n"
                 "(polyIC's binary-IG signature is ISG-dominated → overlaps IFNb)", fontsize=10)
    # legend by time (greyscale) since color encodes sign
    ax.legend(title="time", fontsize=8, frameon=False)
    ax.text(0.02, 0.95, "+  ⇒ wrong (IFNb scored upstream)", transform=ax.transAxes,
            color=RED, fontsize=8, va="top")
    fig.savefig(FIGDIR / "f13_polyic.pdf")
    plt.close(fig)


def main():
    ds_acc, ca_acc, n = fig9_oes_headline()
    fig10_oes_peraxis()
    fig11_sheu_kinetic()
    fig12_sheu_heatmap()
    fig13_polyic()
    # sanity prints
    print("Figures written to", FIGDIR)
    print(f"[F9] Oes directional_score {ds_acc*100:.0f}% vs cross_asym {ca_acc*100:.0f}% (n={n})")
    _, per_t = _sheu_bench()
    for t, g in per_t.items():
        print(f"[Sheu] {t}: {int(g['correct'].sum())}/{len(g)} directional correct")


if __name__ == "__main__":
    main()
