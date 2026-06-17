#!/usr/bin/env python
"""Analysis + figures for the COVID-progression experiment (§30 analysis stage).

Consumes the fit-stage artifacts (no model / no atlas reload), scores cross_asym
direction against the clinical severity order, runs the nested-donor bootstrap,
recovers the progression order, emits 7 figures, and writes the honest-caveat
results doc with the pre-registered GREEN/AMBER/RED verdict.

Usage (cluster, CPU):
  python scripts/analyze_covid_progression.py \
      --fit_dir results/covid_progression/fit \
      --output_dir results/covid_progression --n_boot 1000 --seed 42
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cascadir.analysis import score_directions
from cascadir.progression import (
    SIG_PREFIX, accuracy_vs_oracle, bootstrap_cross_asym, kendall_tau, recover_order,
)

CONTROL = "Healthy"
GRADES = ["Asymptomatic", "Mild", "Moderate", "Severe", "Critical"]
GREEN, RED, BLUE, GREY = "#2a9d4a", "#c0392b", "#2c6fbb", "#888888"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def _present_grades(direction_table) -> list:
    present = set(direction_table["condition_a"]) | set(direction_table["condition_b"])
    return [g for g in GRADES if g in present]


def _oracle(grades) -> list:
    return [(grades[i], grades[j]) for i in range(len(grades)) for j in range(i + 1, len(grades))]


# --------------------------- figures ---------------------------

def _fig_accuracy_bar(bench, boot, save):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    vals = [bench.cross_accuracy, bench.dirscore_accuracy]
    ax.bar([0, 1], vals, color=[BLUE, GREY], width=0.6)
    ax.axhline(0.5, ls="--", c=RED, lw=1, label="chance")
    if boot.get("accuracy"):
        a = boot["accuracy"]
        ax.errorbar(0, a["point"], yerr=[[a["point"] - a["ci_lo"]], [a["ci_hi"] - a["point"]]],
                    fmt="o", color="black", capsize=4, label="donor-bootstrap 95% CI")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cross_asym\n(direction)", "directional_score\n(symmetric control)"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("sign-accuracy vs severity order")
    ax.set_title("Direction recovery from a single snapshot")
    for x, v in zip([0, 1], vals):
        ax.text(x, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(save, dpi=150); plt.close(fig)


def _fig_per_pair(bench_table, save):
    t = bench_table[bench_table["found"]].copy().sort_values("cross_asym_median")
    colors = [GREEN if c else RED for c in t["cross_correct"]]
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(t) + 1.5))
    y = np.arange(len(t))
    ax.barh(y, t["cross_asym_median"], color=colors)
    ax.axvline(0, c="black", lw=0.8)
    labels = [f"{a} vs {b}  (↑{u})" for a, b, u in
              zip(t["condition_a"], t["condition_b"], t["expected_upstream"])]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("cross_asym_median  (>0 ⇒ alphabetically-first is upstream)")
    ax.set_title("Per-pair direction call (green=correct, red=wrong)")
    fig.savefig(save, dpi=150); plt.close(fig)


def _fig_cross_engagement_heatmap(per_ct, grades, save):
    # M[a,b] = median over cell types of sA_PB_norm for the (a,b) ordered orientation.
    n = len(grades); gi = {g: i for i, g in enumerate(grades)}
    M = np.full((n, n), np.nan)
    for (a, b), df in per_ct.groupby(["condition_a", "condition_b"]):
        if a in gi and b in gi:
            M[gi[a], gi[b]] = df["sA_PB_norm"].median()
            M[gi[b], gi[a]] = df["sB_PA_norm"].median()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    vmax = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(grades, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(grades, fontsize=8)
    ax.set_xlabel("signature engaged (S_col)"); ax.set_ylabel("cells from (row)")
    ax.set_title("Cross-engagement M[row, col] = s(row, S_col) − Healthy")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(save, dpi=150); plt.close(fig)


def _fig_celltype_consensus(per_ct, save):
    piv = per_ct.assign(pair=per_ct["condition_a"] + "→" + per_ct["condition_b"],
                        sign=np.sign(per_ct["cross_asym"])).pivot_table(
        index="pair", columns="cell_type", values="sign", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(0.5 * piv.shape[1] + 3, 0.4 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title("Per-cell-type cross_asym sign (consensus = a real seed)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(save, dpi=150); plt.close(fig)


def _fig_signature_scatter(cells, a, b, save, max_per=1200):
    sa, sb = f"{SIG_PREFIX}{a}", f"{SIG_PREFIX}{b}"
    cts = cells["cell_type"].value_counts().head(4).index.tolist()
    fig, axes = plt.subplots(1, len(cts), figsize=(3.4 * len(cts), 3.4), squeeze=False)
    for ax, ct in zip(axes[0], cts):
        for cond, col in [(CONTROL, GREY), (a, BLUE), (b, RED)]:
            d = cells[(cells["cell_type"] == ct) & (cells["condition"] == cond)]
            if len(d) > max_per:
                d = d.sample(max_per, random_state=0)
            ax.scatter(d[sa], d[sb], s=4, alpha=0.4, color=col, label=cond)
        ax.set_title(ct, fontsize=9); ax.set_xlabel(f"score on S_{a}"); ax.set_ylabel(f"score on S_{b}")
    axes[0][0].legend(fontsize=7, markerscale=2)
    fig.suptitle(f"Seed view: {a} (upstream) cells extend along S_{b} axis", fontsize=10)
    fig.savefig(save, dpi=150); plt.close(fig)


def _fig_order_recovery(recovered, true_order, tau, save):
    fig, ax = plt.subplots(figsize=(4.4, 0.6 * len(true_order) + 1.5))
    n = len(true_order)
    rank_rec = {g: i for i, g in enumerate(recovered)}
    for i, g in enumerate(true_order):
        j = rank_rec.get(g, i)
        ok = (i == j)
        ax.plot([0, 1], [n - 1 - i, n - 1 - j], "-", color=(GREEN if ok else RED), lw=1.5)
        ax.text(-0.02, n - 1 - i, g, ha="right", va="center", fontsize=9)
        ax.text(1.02, n - 1 - j, g, ha="left", va="center", fontsize=9)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.5, n - 0.5); ax.axis("off")
    ax.text(0, n - 0.2, "true order", ha="center", fontsize=9, weight="bold")
    ax.text(1, n - 0.2, "recovered", ha="center", fontsize=9, weight="bold")
    ax.set_title(f"Severity-order recovery from one snapshot (Kendall τ = {tau:+.2f})")
    fig.savefig(save, dpi=150); plt.close(fig)


def _fig_bootstrap_ci(boot_per_pair, save):
    t = boot_per_pair.copy()
    t["pair"] = t["condition_a"] + " vs " + t["condition_b"]
    t = t.sort_values("cross_asym")
    y = np.arange(len(t))
    excl0 = (t["ci_lo"] > 0) | (t["ci_hi"] < 0)
    colors = [BLUE if e else GREY for e in excl0]
    fig, ax = plt.subplots(figsize=(6.0, 0.5 * len(t) + 1.5))
    ax.errorbar(t["cross_asym"], y,
                xerr=[t["cross_asym"] - t["ci_lo"], t["ci_hi"] - t["cross_asym"]],
                fmt="o", ecolor="gray", capsize=3, mfc="black", mec="black", ls="none")
    ax.axvline(0, c=RED, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(t["pair"], fontsize=8)
    ax.set_xlabel("cross_asym (donor-bootstrap 95% CI)")
    ax.set_title("Donor-bootstrap CI per grade pair (blue = CI excludes 0)")
    fig.savefig(save, dpi=150); plt.close(fig)


# --------------------------- driver ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_dir", default="results/covid_progression/fit")
    ap.add_argument("--output_dir", default="results/covid_progression")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    fit = Path(args.fit_dir); out = Path(args.output_dir); plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    dt = pd.read_csv(fit / "direction_table.csv")
    cache = pd.read_parquet(fit / "donor_signature_scores.parquet")
    per_ct = pd.read_csv(fit / "per_celltype_cross_asym.csv")
    cells = pd.read_parquet(fit / "cell_scores_subsample.parquet")

    grades = _present_grades(dt)
    oracle = _oracle(grades)
    bench = score_directions(dt, oracle, null_alpha=0.05)
    bench.table.to_csv(out / "benchmark.csv", index=False)

    boot = bootstrap_cross_asym(cache, grades, CONTROL, oracle=oracle,
                                n_boot=args.n_boot, seed=args.seed)
    boot["per_pair"].to_csv(out / "bootstrap_per_pair.csv", index=False)

    # point estimates from direction_table for order recovery
    cross_by_pair = {(r.condition_a, r.condition_b): r.cross_asym_median
                     for r in dt.itertuples()}
    recovered = recover_order(cross_by_pair, grades)
    tau = kendall_tau(recovered, grades)

    # figures
    _fig_accuracy_bar(bench, boot, plots / "direction_accuracy_bar.pdf")
    _fig_per_pair(bench.table, plots / "per_pair_cross_asym.pdf")
    if not per_ct.empty:
        _fig_cross_engagement_heatmap(per_ct, grades, plots / "cross_engagement_heatmap.pdf")
        _fig_celltype_consensus(per_ct, plots / "per_celltype_sign_consensus.pdf")
    _fig_order_recovery(recovered, grades, tau, plots / "severity_order_recovery.pdf")
    _fig_bootstrap_ci(boot["per_pair"], plots / "cross_asym_bootstrap_ci.pdf")
    # cleanest adjacent pair (largest |cross_asym| among consecutive grades)
    adj = [(grades[i], grades[i + 1]) for i in range(len(grades) - 1)]
    if adj and not cells.empty:
        best = max(adj, key=lambda p: abs(cross_by_pair.get(tuple(sorted(p)), 0.0)))
        a, b = sorted(best)
        _fig_signature_scatter(cells, a, b, plots / f"signature_scatter_{a}_vs_{b}.pdf")

    # verdict (pre-registered thresholds)
    acc_ci_lo = boot["accuracy"]["ci_lo"] if boot.get("accuracy") else float("nan")
    tau_pt = boot["kendall_tau"]["point"] if boot.get("kendall_tau") else tau
    green = (bench.cross_accuracy >= 0.8 and bench.dirscore_accuracy <= 0.6
             and acc_ci_lo > 0.5 and tau_pt >= 0.6)
    amber = (not green) and (bench.cross_accuracy >= 0.6)
    verdict = "GREEN" if green else ("AMBER" if amber else "RED")

    summary = {
        "grades": grades, "n_pairs": len(oracle),
        "cross_accuracy": bench.cross_accuracy,
        "dirscore_accuracy": bench.dirscore_accuracy,
        "n_scored": bench.n_scored, "classification_counts": bench.classification_counts,
        "bootstrap_accuracy": boot.get("accuracy"),
        "kendall_tau_point": tau_pt, "kendall_tau_boot": boot.get("kendall_tau"),
        "recovered_order": recovered, "true_order": grades, "verdict": verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    _write_report(out, summary, bench, boot, recovered, grades, tau_pt)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nVERDICT: {verdict}.  Report: reports/covid_progression/COVID_PROGRESSION_RESULTS.md")
    return 0


def _write_report(out, s, bench, boot, recovered, grades, tau_pt):
    acc = boot.get("accuracy") or {}
    lines = [
        "# COVID severity-progression — cascade-direction results (§30)",
        "",
        "## ⚠ VALIDITY — READ FIRST",
        "- **Cross-sectional**: this recovers a *direction from a single snapshot*, NOT a "
        "per-subject forecast (no future timepoints).",
        "- **Nested donors**: each patient has ONE grade, so donor-level rigour is a "
        "**donor-bootstrap** (resampling donors within each grade + Healthy), not "
        "within-donor pairing. Path A / `signature_coupling(donor_level=True)` are N/A here.",
        "- **Magnitude confound**: severity is a monotone-intensity axis. The headline check "
        "is `cross_asym` accuracy ≫ the SYMMETRIC `directional_score` control. The apparatus "
        "(`APPARATUS_GATE_RESULTS.md`) shows pure monotone nesting FLIPS direction, while a "
        "genuine seed recovers it — so the cross-vs-control gap + per-cell-type consensus "
        "decide which regime COVID is in.",
        "- Direction ≠ causation; PBMC blood only; single dataset.",
        "",
        f"## Verdict: **{s['verdict']}**",
        "",
        "| metric | value | pre-registered GREEN |",
        "|---|---|---|",
        f"| cross_asym accuracy (vs severity order, {bench.n_scored} pairs) | "
        f"**{bench.cross_accuracy:.0%}** | ≥ 80% |",
        f"| directional_score control accuracy | {bench.dirscore_accuracy:.0%} | ≤ 60% (≪ cross) |",
        f"| donor-bootstrap accuracy 95% CI | "
        f"[{acc.get('ci_lo', float('nan')):.2f}, {acc.get('ci_hi', float('nan')):.2f}] | lower > 0.50 |",
        f"| Kendall τ (recovered vs true order) | {tau_pt:+.2f} | ≥ 0.60 |",
        "",
        f"**Recovered order:** {' → '.join(recovered)}  ",
        f"**True clinical order:** {' → '.join(grades)}",
        "",
        "## Figures (`plots/`)",
        "- `direction_accuracy_bar.pdf` — cross_asym vs symmetric control (headline).",
        "- `severity_order_recovery.pdf` — recovered ladder vs truth (Kendall τ).",
        "- `per_pair_cross_asym.pdf` — per grade-pair call (green=correct).",
        "- `cross_engagement_heatmap.pdf` — grade×grade M[row, S_col].",
        "- `per_celltype_sign_consensus.pdf` — is the seed consistent across cell types?",
        "- `cross_asym_bootstrap_ci.pdf` — donor-bootstrap CI per pair.",
        "- `signature_scatter_*.pdf` — upstream cells extending along the downstream-signature axis.",
        "",
        "Generated by `scripts/analyze_covid_progression.py`. Scored vs the pre-registered "
        "severity oracle (`reports/covid_progression/PRE_REGISTRATION.md`).",
    ]
    rp = Path("reports/covid_progression/COVID_PROGRESSION_RESULTS.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
