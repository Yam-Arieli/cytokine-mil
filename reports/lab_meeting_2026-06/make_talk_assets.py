"""
Generate all figures + tables for the lab-meeting talk (2026-06).
Real numbers: ID/Oes from local CSVs; headline accuracies + Sheu coupling from the
validated session results (M8 + the 2026-06 signature-coupling run).
Run:  python reports/lab_meeting_2026-06/make_talk_assets.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False,
})
GREEN, GREY, BLUE, RED = "#2ca25f", "#999999", "#3182bd", "#de2d26"


# ---------------------------------------------------------------- Fig A
def fig_direction_accuracy():
    datasets = ["Oesinghaus\n(human PBMC, 24h)", "Sheu\n(mouse BMDM, 5h)",
                "Immune Dictionary\n(mouse LN, 4h)"]
    cross = [88, 86, 83]          # cross_asym (antisymmetric) — % correct direction
    cross_n = ["15/17", "6/7", "5/6"]
    dscore = [47, np.nan, 33]     # directional_score (symmetric control)
    x = np.arange(len(datasets)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    b1 = ax.bar(x - w/2, cross, w, color=GREEN, label="cross_asym (antisymmetric → direction)")
    dvals = [d if not np.isnan(d) else 0 for d in dscore]
    b2 = ax.bar(x + w/2, dvals, w, color=GREY,
                label="directional_score (symmetric control)")
    ax.axhline(50, ls="--", lw=1.5, color=RED); ax.text(2.35, 51.5, "chance", color=RED, fontsize=12)
    for xi, v, n in zip(x - w/2, cross, cross_n):
        ax.text(xi, v + 1.5, f"{v}%\n({n})", ha="center", va="bottom", fontsize=12, fontweight="bold")
    for xi, v, raw in zip(x + w/2, dvals, dscore):
        ax.text(xi, v + 1.5, "n/a" if np.isnan(raw) else f"{int(raw)}%", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylim(0, 106)
    ax.set_ylabel("Cascade direction correct (%)")
    ax.set_title("Direction from a single snapshot: 88 / 86 / 83% on known cascades", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=11, frameon=False)
    fig.savefig(OUT / "fig_direction_accuracy.png"); plt.close(fig)
    print("wrote fig_direction_accuracy.png")


# ---------------------------------------------------------------- Fig B
def fig_cascade_examples():
    df = pd.read_csv(REPO / "reports/immune_dictionary/per_axis_summary.csv")
    df = df[df["expected_sign"].notna() & (df["expected_sign"] != 0)].copy()
    df = df[df["cross_sign_correct"] == True].copy()  # the recovered ones
    # pretty names + direction from sign
    pretty = {"IFNb": "IFN-β", "IFNg": "IFN-γ", "IL15": "IL-15", "IL18": "IL-18",
              "IL2": "IL-2", "IL6": "IL-6", "TNFa": "TNF"}
    rows = []
    for _, r in df.iterrows():
        a, b = pretty.get(r.axis_a, r.axis_a), pretty.get(r.axis_b, r.axis_b)
        up, down = (a, b) if r.cross_median > 0 else (b, a)
        rows.append((f"{up} → {down}", abs(r.cross_median)))
    rows.sort(key=lambda t: t[1])
    labels = [t[0] for t in rows]; vals = [t[1] for t in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=GREEN, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.002, yi, "✓", va="center", fontsize=16, color=GREEN, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=15)
    ax.set_xlabel("|cross_asym|  (direction signal strength)")
    ax.set_title("Recovered textbook cascades — Immune Dictionary (mouse, in vivo)")
    ax.text(0.98, 0.04, "NK→IFN-γ axis + TNF→IL-6, all correct direction",
            transform=ax.transAxes, ha="right", fontsize=11, color="#444444", style="italic")
    ax.set_xlim(0, max(vals) * 1.25)
    fig.savefig(OUT / "fig_cascade_examples.png"); plt.close(fig)
    print("wrote fig_cascade_examples.png")


# ---------------------------------------------------------------- Fig C
def fig_sheu_coupling_win():
    # Sheu 3h signature-space coupling (real, 2026-06 run). latent-geometry Path A: q=1 on all.
    pairs = ["LPS — IFN-β\n(MUST)", "polyIC — IFN-β\n(MUST)",
             "P3CSK — IFN-β\n(neg.)", "CpG — IFN-β\n(neg.)"]
    coupling = [0.63, 1.22, 0.07, 0.08]
    pval = [0.000, 0.000, 0.975, 0.953]
    colors = [GREEN, GREEN, GREY, GREY]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(pairs))
    ax.bar(x, coupling, 0.6, color=colors)
    for xi, v, p in zip(x, coupling, pval):
        tag = "p<0.001 ✓" if p < 0.05 else f"p={p:.2f} (n.s.)"
        ax.text(xi, v + 0.03, tag, ha="center", fontsize=11.5,
                color=(GREEN if p < 0.05 else GREY), fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=12)
    ax.set_ylabel("signature coupling  (M[a,b]+M[b,a])")
    ax.set_ylim(0, 1.55)
    ax.set_title("Signature coupling recovers what latent geometry missed (Sheu 3h)",
                 fontsize=15, pad=16)
    ax.text(0.5, 0.93, "latent-geometry Path A on Sheu: q = 1 on every pair  →  0 / 2 MUST recovered",
            transform=ax.transAxes, ha="center", fontsize=12, color=RED,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec=RED, lw=1))
    fig.savefig(OUT / "fig_sheu_coupling_win.png"); plt.close(fig)
    print("wrote fig_sheu_coupling_win.png")


# ---------------------------------------------------------------- Fig D (backup)
def fig_coupling_vs_pathA():
    pc = pd.read_csv(REPO / "results/gene_dynamics_phase0/pipeline_a_b_full19/per_celltype.csv")
    pc["sym"] = pc["sA_PB_norm"] + pc["sB_PA_norm"]
    g = pc.groupby(["axis_a", "axis_b"])["sym"].median().reset_index()
    ax_csv = pd.read_csv(REPO / "reports/cascade_pairs/cytokine_axes.csv")[["axis_a", "axis_b", "axis_strength"]]
    m = g.merge(ax_csv, on=["axis_a", "axis_b"], how="inner")
    rx = m["sym"].rank(); ry = m["axis_strength"].rank()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.scatter(m["axis_strength"], m["sym"], s=45, color=BLUE, alpha=0.7, edgecolor="white")
    ax.set_xlabel("Path A coupling — latent space (axis_strength)")
    ax.set_ylabel("Coupling — gene signatures (M+Mᵀ)")
    ax.set_title(f"Two coupling notions disagree (Spearman ρ = {rho:.2f}, n={len(m)})")
    ax.text(0.04, 0.93, "full 48-cytokine run: ρ = 0.11", transform=ax.transAxes,
            fontsize=11, color="#444444", style="italic")
    fig.savefig(OUT / "fig_coupling_vs_pathA.png"); plt.close(fig)
    print(f"wrote fig_coupling_vs_pathA.png (rho={rho:.3f}, n={len(m)})")


# ---------------------------------------------------------------- Tables
def tables():
    datasets = pd.DataFrame([
        ["Oesinghaus", "human PBMC (mixed blood)", "human", "91 cytokines + PBS", "12",
         "~4000 HVGs", "24 h", "ex vivo", "scale + the standing coupling result"],
        ["Sheu", "mouse BMDM (macrophages)", "mouse", "7 stimuli + PBS", "~4 pseudo-donors",
         "500-gene immune panel", "time-course 1/3/5 h", "ex vivo", "clean textbook TLR cascades"],
        ["Immune Dictionary", "mouse lymph node (whole tissue)", "mouse", "86 cytokines + PBS", "3 mice",
         "~4000 HVGs (of 31k)", "4 h", "in vivo", "in-vivo cross-check"],
    ], columns=["Dataset", "System", "Species", "Stimuli", "Donors", "Genes",
                "Time point", "Setting", "Role"])
    grid = pd.DataFrame([
        ["Direction (cross_asym)", "✓ 88% (15/17)", "✓ 86% (6/7)", "✓ 83% (5/6)"],
        ["Coupling — latent space (Path A)", "✓ 121 axes (~50% lit)", "✗ no power (q=1)", "— not run"],
        ["Coupling — gene signatures (new)", "⚠ gate too loose", "✓ 2/2 recovered", "— not run"],
    ], columns=["Method", "Oesinghaus", "Sheu", "Immune Dictionary"])
    with pd.ExcelWriter(OUT / "table_datasets.xlsx") as w:
        datasets.to_excel(w, index=False, sheet_name="datasets")
    with pd.ExcelWriter(OUT / "table_results_grid.xlsx") as w:
        grid.to_excel(w, index=False, sheet_name="results")
    datasets.to_markdown(OUT / "table_datasets.md", index=False)
    grid.to_markdown(OUT / "table_results_grid.md", index=False)
    print("wrote table_datasets.{xlsx,md}, table_results_grid.{xlsx,md}")


if __name__ == "__main__":
    fig_direction_accuracy()
    fig_cascade_examples()
    fig_sheu_coupling_win()
    fig_coupling_vs_pathA()
    tables()
    print("DONE ->", OUT)
