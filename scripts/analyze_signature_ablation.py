"""
Analysis + figures for the 2x2 signature-definition ablation.

Consumes the experiment outputs written by `scripts/run_signature_ablation.py`
(ablation_summary.csv, coupling_<variant>.csv, direction_detail_<variant>.csv)
and ALSO re-derives the four signature gene sets (importing the ablation module)
so it can report gene-level overlap -- the strongest evidence for "DE ~= IG" and
for "what panel-residualisation actually removes".

Outputs (under <output_dir>):
  analysis_stats.md / analysis_stats.json   -- extracted statistics
  plots/fig1_summary_bars.png               -- direction_acc | coupled_frac | hub  (the decision)
  plots/fig2_benchmark_heatmap.png          -- 17 benchmark pairs x 4 variants: correct/wrong + sign
  plots/fig3_coupling_null.png              -- per-variant coupling_null_p distribution (over-permissiveness)
  plots/fig4_signature_overlap.png          -- 4x4 mean-Jaccard between variants + per-cytokine DE-vs-IG
  plots/fig5_panel_effect.png               -- mean_expression of genes REMOVED vs KEPT by vs-panel

Robust: each figure/stat is in its own try-block; a missing input degrades that
piece to a warning, the rest still produces output. CPU-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_signature_ablation as rsa  # noqa: E402

VARIANTS = ["IG_vsPBS", "IG_vsPanel", "DE_vsPBS", "DE_vsPanel"]
COLORS = {"IG_vsPBS": "#4C72B0", "IG_vsPanel": "#55A868",
          "DE_vsPBS": "#C44E52", "DE_vsPanel": "#8172B3"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": False,
})


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_summary_bars(summ: pd.DataFrame, out: Path) -> None:
    summ = summ.set_index("variant").reindex(VARIANTS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    cols = [c for c in VARIANTS if c in summ.index]
    bar_colors = [COLORS[c] for c in cols]

    # 1) direction accuracy with the 88% reference
    ax = axes[0]
    acc = summ.loc[cols, "direction_acc"].to_numpy()
    ax.bar(cols, acc, color=bar_colors)
    ax.axhline(15 / 17, ls="--", c="k", lw=1, label="published 15/17 = 88%")
    for i, v in enumerate(acc):
        n = summ.loc[cols[i]]
        ax.text(i, v + 0.01, f"{int(n['direction_correct'])}/{int(n['direction_total'])}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("direction accuracy")
    ax.set_title("Direction (HOLD this)"); ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(axis="x", rotation=25)

    # 2) coupled fraction (over-permissiveness; LOWER = better)
    ax = axes[1]
    cf = summ.loc[cols, "coupled_frac"].to_numpy()
    ax.bar(cols, cf, color=bar_colors)
    for i, v in enumerate(cf):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("fraction of pairs 'coupled' (null p<0.05)")
    ax.set_title("Coupling gate (LOWER = less over-call)")
    ax.tick_params(axis="x", rotation=25)

    # 3) hub domination (LOWER = better)
    ax = axes[2]
    hub = summ.loc[cols, "top20_max_cyt_count"].to_numpy()
    ax.bar(cols, hub, color=bar_colors)
    for i, v in enumerate(hub):
        lbl = f"{int(v)}  ({summ.loc[cols[i], 'top20_max_cyt']})"
        ax.text(i, v + 0.1, lbl, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("max single-cytokine count in top-20 coupled")
    ax.set_title("Hub domination (LOWER = better)")
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Signature ablation — the decision at a glance", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_benchmark_heatmap(detail: Dict[str, pd.DataFrame], out: Path) -> None:
    # union of benchmark pairs, ordered by pair_status then name
    base = None
    for v in VARIANTS:
        if v in detail and not detail[v].empty:
            base = detail[v]; break
    if base is None:
        raise RuntimeError("no direction_detail available")
    base = base.copy()
    base["pair"] = base["axis_a"] + " / " + base["axis_b"]
    order = base.sort_values(["pair_status", "pair"])["pair"].tolist()

    grid = np.full((len(order), len(VARIANTS)), np.nan)      # 1 correct, 0 wrong
    signs = np.full((len(order), len(VARIANTS)), 0.0)
    pidx = {p: i for i, p in enumerate(order)}
    for j, v in enumerate(VARIANTS):
        d = detail.get(v)
        if d is None or d.empty:
            continue
        d = d.copy(); d["pair"] = d["axis_a"] + " / " + d["axis_b"]
        for _, r in d.iterrows():
            if r["pair"] in pidx:
                grid[pidx[r["pair"]], j] = 1.0 if bool(r["correct"]) else 0.0
                signs[pidx[r["pair"]], j] = r["cross_asym_median"]

    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.42 * len(order) + 1)))
    cmap = matplotlib.colors.ListedColormap(["#C44E52", "#55A868"])  # wrong, correct
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels(VARIANTS, rotation=25, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    for i in range(len(order)):
        for j in range(len(VARIANTS)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "·", ha="center", va="center", color="grey")
            else:
                mark = "✓" if grid[i, j] == 1 else "✗"
                ax.text(j, i, f"{mark}\n{signs[i, j]:+.3f}", ha="center", va="center",
                        fontsize=6.5, color="white")
    ax.set_title("Per-benchmark-pair direction: ✓correct / ✗wrong\n(value = median cross_asym)")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_coupling_null(coup: Dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, v in zip(axes.ravel(), VARIANTS):
        d = coup.get(v)
        ax.set_title(v, color=COLORS[v])
        if d is None or d.empty or "coupling_null_p" not in d.columns:
            ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes); continue
        p = d["coupling_null_p"].dropna().to_numpy()
        ax.hist(p, bins=20, range=(0, 1), color=COLORS[v], alpha=0.85)
        frac = float(np.mean(p < 0.05))
        ax.axvline(0.05, ls="--", c="k", lw=1)
        ax.text(0.97, 0.92, f"p<0.05: {frac:.0%}\nn={len(p)}", ha="right",
                va="top", transform=ax.transAxes, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("coupling_null_p")
    for ax in axes[:, 0]:
        ax.set_ylabel("# pairs")
    fig.suptitle("Coupling gene-set-null p-values — left spike = over-permissive gate",
                 y=1.0)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_signature_overlap(sigs: Dict[str, Dict[str, List[str]]], out: Path) -> Dict:
    cyts = sorted(set.intersection(*[set(sigs[v]) for v in sigs]))
    # 4x4 mean Jaccard across cytokines
    M = np.full((len(VARIANTS), len(VARIANTS)), np.nan)
    for i, vi in enumerate(VARIANTS):
        for j, vj in enumerate(VARIANTS):
            if vi in sigs and vj in sigs:
                js = [_jaccard(sigs[vi][c], sigs[vj][c]) for c in cyts]
                M[i, j] = float(np.nanmean(js))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels(VARIANTS, rotation=30, ha="right")
    ax.set_yticks(range(len(VARIANTS))); ax.set_yticklabels(VARIANTS)
    for i in range(len(VARIANTS)):
        for j in range(len(VARIANTS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="mean Jaccard")
    ax.set_title("Signature overlap between variants\n(mean across cytokines)")

    # per-cytokine DE_vsPBS vs IG_vsPBS Jaccard (the 'do we need the encoder' number)
    ax = axes[1]
    if "DE_vsPBS" in sigs and "IG_vsPBS" in sigs:
        js = {c: _jaccard(sigs["DE_vsPBS"][c], sigs["IG_vsPBS"][c]) for c in cyts}
        order = sorted(js, key=js.get)
        ax.barh(order, [js[c] for c in order], color="#C44E52")
        ax.axvline(float(np.nanmean(list(js.values()))), ls="--", c="k",
                   label=f"mean {np.nanmean(list(js.values())):.2f}")
        ax.set_xlabel("Jaccard(DE_vsPBS, IG_vsPBS)"); ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=7); ax.legend(fontsize=8)
        ax.set_title("Per-cytokine DE-vs-IG gene overlap\n(high ⇒ encoder/IG unnecessary)")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return {"jaccard_matrix": M.tolist(), "variant_order": VARIANTS,
            "n_cytokines": len(cyts)}


def fig_panel_effect(sigs, ig_long: pd.DataFrame, out: Path) -> Dict:
    """For vs-panel variants: are the genes REMOVED (relative to vs-PBS) the
    high-mean-expression (shared-activation proxy) ones?"""
    mean_expr = (ig_long.groupby("gene")["mean_expression"].mean().to_dict()
                 if "mean_expression" in ig_long.columns else {})
    if not mean_expr:
        raise RuntimeError("no mean_expression in binary_ig to proxy shared activation")
    pairs = [("IG_vsPBS", "IG_vsPanel"), ("DE_vsPBS", "DE_vsPanel")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    stats = {}
    for ax, (base, panel) in zip(axes, pairs):
        if base not in sigs or panel not in sigs:
            ax.set_visible(False); continue
        removed_expr, kept_expr = [], []
        for c in set(sigs[base]) & set(sigs[panel]):
            b, p = set(sigs[base][c]), set(sigs[panel][c])
            removed_expr += [mean_expr[g] for g in (b - p) if g in mean_expr]
            kept_expr += [mean_expr[g] for g in (b & p) if g in mean_expr]
        data = [kept_expr or [0], removed_expr or [0]]
        ax.boxplot(data, labels=["kept by panel", "removed by panel"], showfliers=False)
        ax.set_ylabel("gene mean_expression (shared-activation proxy)")
        ax.set_title(f"{base} → {panel}")
        stats[f"{base}->{panel}"] = {
            "n_removed": len(removed_expr), "n_kept": len(kept_expr),
            "median_removed_expr": float(np.median(removed_expr)) if removed_expr else None,
            "median_kept_expr": float(np.median(kept_expr)) if kept_expr else None,
        }
    fig.suptitle("Does vs-panel remove the high-expression (shared-activation) genes?",
                 y=1.0)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_dir", required=True,
                   help="Dir written by run_signature_ablation.py (has ablation_summary.csv)")
    p.add_argument("--output_dir", default=None, help="Default: <ablation_dir>/analysis")
    # inputs needed to re-derive the 4 signature sets (for gene-overlap figs)
    p.add_argument("--dataset", default="oesinghaus", choices=["oesinghaus", "sheu"])
    p.add_argument("--binary_ig_parquet", required=True)
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--hvg_path", required=True)
    p.add_argument("--top_n", type=int, default=50)
    p.add_argument("--pbs_label", default="PBS")
    p.add_argument("--time_filter", default=None)
    p.add_argument("--exclude_donors", nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    abl = Path(args.ablation_dir)
    out = Path(args.output_dir) if args.output_dir else abl / "analysis"
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    log = lambda m="": print(m, flush=True)
    stats: Dict[str, object] = {}

    # ---- load experiment outputs ----
    summ = pd.read_csv(abl / "ablation_summary.csv")
    stats["summary"] = summ.to_dict(orient="records")
    coup = {v: (pd.read_csv(abl / f"coupling_{v}.csv")
                if (abl / f"coupling_{v}.csv").exists() else pd.DataFrame())
            for v in VARIANTS}
    detail = {v: (pd.read_csv(abl / f"direction_detail_{v}.csv")
                  if (abl / f"direction_detail_{v}.csv").exists() else pd.DataFrame())
              for v in VARIANTS}

    # ---- which benchmark pairs each variant gets wrong + flips vs the IG baseline ----
    wrong = {}
    for v in VARIANTS:
        d = detail.get(v)
        if d is not None and not d.empty:
            wrong[v] = sorted((d.loc[~d["correct"].astype(bool), "axis_a"] + " / "
                               + d.loc[~d["correct"].astype(bool), "axis_b"]).tolist())
    stats["wrong_pairs_per_variant"] = wrong
    if "IG_vsPBS" in wrong:
        base = set(wrong["IG_vsPBS"])
        stats["flips_vs_IG_vsPBS"] = {
            v: {"newly_wrong": sorted(set(wrong.get(v, [])) - base),
                "newly_fixed": sorted(base - set(wrong.get(v, [])))}
            for v in VARIANTS if v != "IG_vsPBS"}

    # ---- re-derive the 4 signature sets for gene-overlap (needs cells) ----
    sigs = None
    try:
        t0 = time.time()
        ig_df, _ = rsa._ig_matrix(args.binary_ig_parquet)
        ig_long = pd.read_parquet(args.binary_ig_parquet)
        sig_cyts = sorted(str(c) for c in ig_df.index)
        ld_args = SimpleNamespace(
            dataset=args.dataset, manifest_path=args.manifest_path,
            hvg_path=args.hvg_path, pbs_label=args.pbs_label,
            exclude_donors=args.exclude_donors, time_filter=args.time_filter)
        cells, gene_names = rsa._load_cells(ld_args, sig_cyts)
        de_df = rsa.de_matrix_from_cells(cells, sig_cyts, gene_names, pbs_label=args.pbs_label)
        ig_df = ig_df.reindex(columns=gene_names).dropna(axis=1, how="all").fillna(0.0)
        sigs = rsa.build_signature_variants(ig_df, de_df, args.top_n)
        log(f"re-derived 4 signature sets ({time.time()-t0:.0f}s)")
    except Exception as e:  # noqa: BLE001
        log(f"WARN: signature re-derivation failed ({e}); skipping overlap/panel figs")

    # ---- figures (each independent) ----
    for name, fn in [
        ("fig1_summary_bars", lambda: fig_summary_bars(summ, plots / "fig1_summary_bars.png")),
        ("fig2_benchmark_heatmap", lambda: fig_benchmark_heatmap(detail, plots / "fig2_benchmark_heatmap.png")),
        ("fig3_coupling_null", lambda: fig_coupling_null(coup, plots / "fig3_coupling_null.png")),
    ]:
        try:
            fn(); log(f"wrote {name}.png")
        except Exception as e:  # noqa: BLE001
            log(f"WARN: {name} failed ({e})")

    if sigs is not None:
        try:
            stats["signature_overlap"] = fig_signature_overlap(sigs, plots / "fig4_signature_overlap.png")
            log("wrote fig4_signature_overlap.png")
        except Exception as e:  # noqa: BLE001
            log(f"WARN: fig4 failed ({e})")
        try:
            stats["panel_effect"] = fig_panel_effect(sigs, ig_long, plots / "fig5_panel_effect.png")
            log("wrote fig5_panel_effect.png")
        except Exception as e:  # noqa: BLE001
            log(f"WARN: fig5 failed ({e})")

    # ---- write stats ----
    (out / "analysis_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    L = ["# Signature ablation — analysis", ""]
    L.append("## 2x2 summary")
    cols = ["variant", "direction_correct", "direction_total", "direction_acc",
            "coupled", "pairs", "coupled_frac", "top20_max_cyt", "top20_max_cyt_count"]
    cols = [c for c in cols if c in summ.columns]
    L.append(rsa._md(summ[cols], cols))
    L.append("")
    if "flips_vs_IG_vsPBS" in stats:
        L.append("## Direction flips vs the IG_vsPBS baseline")
        for v, fl in stats["flips_vs_IG_vsPBS"].items():
            L.append(f"- **{v}**: newly_fixed={fl['newly_fixed'] or '—'}; "
                     f"newly_wrong={fl['newly_wrong'] or '—'}")
        L.append("")
    if "signature_overlap" in stats:
        jm = np.array(stats["signature_overlap"]["jaccard_matrix"])
        i_de, i_ig = VARIANTS.index("DE_vsPBS"), VARIANTS.index("IG_vsPBS")
        L.append("## Gene-level signature overlap")
        L.append(f"- mean Jaccard(DE_vsPBS, IG_vsPBS) = **{jm[i_de, i_ig]:.2f}** "
                 f"(high ⇒ DE recovers the IG genes ⇒ encoder/IG may be unnecessary).")
        L.append("")
    if "panel_effect" in stats:
        L.append("## What vs-panel removes (shared-activation proxy = mean_expression)")
        for k, s in stats["panel_effect"].items():
            L.append(f"- **{k}**: removed median expr {s['median_removed_expr']}, "
                     f"kept median expr {s['median_kept_expr']} "
                     f"(removed > kept ⇒ panel strips high-expression shared genes).")
        L.append("")
    L.append("## Decision tree")
    L.append("- IG_vsPBS reproduces ~88% (sanity) → wiring correct.")
    L.append("- vs-panel ↓coupled_frac while holding direction_acc → specificity is the lever.")
    L.append("- DE ≈ IG (acc AND gene-Jaccard) → drop the encoder/IG (simplify).")
    L.append("- neither → pivot to donor-level statistics.")
    L.append("")
    L.append("Figures in `plots/`: fig1 (decision bars), fig2 (per-pair flips), "
             "fig3 (gate over-permissiveness), fig4 (signature overlap), fig5 (panel effect).")
    (out / "analysis_stats.md").write_text("\n".join(L) + "\n")
    log(f"\nwrote {out/'analysis_stats.md'} and {out/'analysis_stats.json'}")
    log("DONE.")


if __name__ == "__main__":
    main()
