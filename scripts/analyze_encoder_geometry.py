"""Verdict + figures for the encoder gene-space geometry probe.

Reads what `probe_encoder_gene_geometry.py` wrote and answers the four pre-registered
questions in `reports/encoder_geometry/PRE_REGISTRATION.md`:

  B1  do the collapsed fits have a lower-rank gene -> embedding map than the healthy ones?
  B2  across the eight sweep arms (one shared panel, everything else pinned), does that
      rank predict signature diversity?
  B3  within a fit, are the chronically-shared signature genes the high-norm /
      top-direction-aligned ones?
  B4  or is the geometry flat across all fits -- in which case the encoder is NOT the
      bottleneck and the collapse lives downstream.

Signature diversity is recomputed here with `analyze_encsweep.diversity`, the same function
behind every number in CLAUDE.md §38.3/§38.5, so the two sides of the correlation are on
one footing. Descriptive statistics only -- no engagement scores, no coupling, no direction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _encoder_geometry_fits as R  # noqa: E402
from analyze_encsweep import diversity  # noqa: E402
from probe_encoder_gene_geometry import spearman  # noqa: E402

PRIMARY = "jacobian"   # the probe with the mechanistic chain to IG


def measured_diversity(fit: dict, top_n: int) -> dict | None:
    """meanJ for one fit, from its own signature table, on the §38 footing."""
    if not fit.get("signatures"):
        return None
    p = REPO_ROOT / fit["signatures"]
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "epoch" in df.columns:
        df = df[df.epoch == df.epoch.max()]
    if "rank_ig" not in df.columns:
        df = df.sort_values(["cytokine", "ig"], ascending=[True, False])
        df["rank_ig"] = df.groupby("cytokine").cumcount()
    if fit.get("conditions"):
        df = df[df.cytokine.isin(fit["conditions"])]
    if df.empty:
        return None
    return diversity(df[["cytokine", "gene", "rank_ig"]], top_n)


def make_figures(geo: pd.DataFrame, spec: pd.DataFrame, out: Path, top_n: int) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    made = []
    colour = {"cytokine_mil": "#1f77b4", "cascadir": "#d62728", "control": "#7f7f7f"}
    real = geo[(geo.probe == PRIMARY) & (~geo.fit.str.endswith("__untrained"))]

    # 1. singular-value spectra, healthy vs collapsed
    if spec is not None and len(spec):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, probe in zip(axes, ("onehot", PRIMARY)):
            sub = spec[spec.probe == probe]
            for key, g in sub.groupby("fit"):
                cp = geo.loc[geo.fit == key, "code_path"]
                c = colour.get(cp.iloc[0] if len(cp) else "control", "#999999")
                sv = g.sort_values("k").singular_value.to_numpy()
                if sv.sum() > 0:
                    ax.semilogy(np.arange(len(sv)), sv / sv[0], color=c, alpha=0.75, lw=1.2)
            ax.set_title(f"{probe}: normalised singular spectrum")
            ax.set_xlabel("component"); ax.set_ylabel(r"$\sigma_k/\sigma_0$")
        axes[0].plot([], [], color=colour["cytokine_mil"], label="cytokine_mil path")
        axes[0].plot([], [], color=colour["cascadir"], label="cascadir path")
        axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(plots / "spectra.png", dpi=150); plt.close(fig)
        made.append("spectra.png")

    # 2. PR vs meanJ -- the B2 scatter
    if "mean_jaccard" in real.columns and real.mean_jaccard.notna().any():
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for cp, g in real.groupby("code_path"):
            ax.scatter(g.raw_pr_frac, g.mean_jaccard, s=54, label=cp,
                       color=colour.get(cp, "#999999"),
                       edgecolor="k", linewidth=0.4)
        for _, r in real.iterrows():
            if np.isfinite(r.get("mean_jaccard", np.nan)):
                ax.annotate(r.fit, (r.raw_pr_frac, r.mean_jaccard), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"participation ratio / embed_dim  ({PRIMARY} probe)")
        ax.set_ylabel(f"mean between-cytokine Jaccard @ top-{top_n}")
        ax.axhline(0.006, ls=":", c="grey", lw=1)
        ax.annotate("chance", (ax.get_xlim()[0], 0.006), fontsize=7, color="grey")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(plots / "pr_vs_meanj.png", dpi=150); plt.close(fig)
        made.append("pr_vs_meanj.png")

    # 3. per-gene: does encoder response predict signature membership?
    pg_dir = out / "per_gene"
    files = sorted(pg_dir.glob("*.parquet")) if pg_dir.exists() else []
    if files:
        n = min(len(files), 6)
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), squeeze=False)
        for ax, f in zip(axes[0], files[:n]):
            d = pd.read_parquet(f)
            col = f"{PRIMARY}_norm"
            if "sig_freq" not in d.columns or col not in d.columns:
                ax.set_visible(False); continue
            ax.scatter(d[col], d.sig_freq, s=3, alpha=0.25, color="#333333")
            ax.set_title(f.stem, fontsize=8)
            ax.set_xlabel("gene response norm", fontsize=8)
            ax.set_ylabel("signatures containing gene", fontsize=8)
        fig.tight_layout(); fig.savefig(plots / "gene_norm_vs_freq.png", dpi=150)
        plt.close(fig)
        made.append("gene_norm_vs_freq.png")

    # 4. trained vs its matched untrained control
    unt = geo[geo.fit.str.endswith("__untrained") & (geo.probe == "onehot")].copy()
    if len(unt):
        unt["base"] = unt.fit.str.replace("__untrained", "", regex=False)
        oh = geo[(geo.probe == "onehot") & (~geo.fit.str.endswith("__untrained"))]
        m = oh.merge(unt[["base", "raw_pr_frac"]].rename(
            columns={"base": "fit", "raw_pr_frac": "untrained_pr_frac"}), on="fit")
        if len(m):
            fig, ax = plt.subplots(figsize=(7.5, 4.0))
            x = np.arange(len(m))
            ax.bar(x - 0.2, m.untrained_pr_frac, 0.4, label="untrained (matched dims)",
                   color="#bbbbbb")
            ax.bar(x + 0.2, m.raw_pr_frac, 0.4, label="trained",
                   color=[colour.get(c, "#999") for c in m.code_path])
            ax.set_xticks(x); ax.set_xticklabels(m.fit, rotation=60, ha="right", fontsize=7)
            ax.set_ylabel("PR / embed_dim (one-hot)")
            ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(plots / "trained_vs_untrained.png", dpi=150)
            plt.close(fig)
            made.append("trained_vs_untrained.png")
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "results" / "encoder_geometry"))
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    geo = pd.read_csv(out / "gene_geometry.csv")
    spec_p = out / "spectra.parquet"
    spec = pd.read_parquet(spec_p) if spec_p.exists() else None

    # attach each fit's own measured signature diversity
    div_rows = []
    for fit in R.FITS:
        d = measured_diversity(fit, args.top_n)
        if d:
            div_rows.append({"fit": fit["key"], "mean_jaccard": d["mean_jaccard"],
                             "top5_pool": d["top5_pool"], "distinct_genes": d["distinct_genes"],
                             "collapse_x": d["collapse_x"], "n_cytokines": d["n_cytokines"]})
    div = pd.DataFrame(div_rows)
    if len(div):
        geo = geo.merge(div, on="fit", how="left", suffixes=("", "_sig"))
        div.to_csv(out / "measured_diversity.csv", index=False)
    geo.to_csv(out / "gene_geometry.csv", index=False)

    real = geo[(geo.probe == PRIMARY) & (~geo.fit.str.endswith("__untrained"))].copy()
    lines = []
    A = lines.append
    A("# Encoder gene-space geometry — does Stage 1 bottleneck the gene space?\n")
    A(f"Probe: `{PRIMARY}` (primary; one-hot and w1 in `gene_geometry.csv`). "
      f"Signature diversity at top-{args.top_n}, recomputed with "
      "`analyze_encsweep.diversity`.\n")

    cols = [c for c in ["fit", "code_path", "panel", "embed_dim", "raw_pr", "raw_pr_frac",
                        "raw_var_top1", "mean_cos_u1", "norm_gini", "rho_freq_norm",
                        "mean_jaccard"] if c in real.columns]
    A("## Per-fit geometry\n")
    A(real[cols].sort_values("raw_pr_frac").to_markdown(index=False, floatfmt=".3f") + "\n")

    # ---- B1 -----------------------------------------------------------------
    A("## B1 — do collapsed fits have a lower-rank gene map?\n")
    grp = real.groupby("code_path").raw_pr_frac.agg(["mean", "min", "max", "count"])
    A(grp.to_markdown(floatfmt=".3f") + "\n")
    cm = real[real.code_path == "cytokine_mil"].raw_pr_frac
    cd = real[real.code_path == "cascadir"].raw_pr_frac
    b1 = bool(len(cm) and len(cd) and cm.mean() > cd.mean() and cm.min() > cd.max())
    A(f"\n**B1 {'SUPPORTED' if b1 else 'NOT SUPPORTED'}** — cytokine_mil mean PR/d "
      f"{cm.mean():.3f} vs cascadir {cd.mean():.3f}"
      f"{'; the two groups do not overlap.' if b1 else '; the groups overlap.'}\n")

    # ---- B2 -----------------------------------------------------------------
    A("\n## B2 — does the rank predict signature diversity?\n")
    sw = real[(real.panel == "sweep24") & real.mean_jaccard.notna()]
    rho = spearman(sw.raw_pr_frac.to_numpy(), sw.mean_jaccard.to_numpy()) if len(sw) >= 4 else np.nan
    A(f"Within the eight sweep arms (one shared panel, all other settings pinned), "
      f"n={len(sw)}: **Spearman(PR/d, meanJ) = {rho:.3f}**.\n")
    b2 = bool(np.isfinite(rho) and rho <= -0.7)
    A(f"\n**B2 {'SUPPORTED' if b2 else 'NOT SUPPORTED'}** "
      f"(pre-registered threshold rho <= -0.7).\n")

    # ---- B3 -----------------------------------------------------------------
    A("\n## B3 — are the chronically-shared genes the high-response ones?\n")
    if "rho_freq_norm" in real.columns:
        b3t = real[["fit", "code_path", "rho_freq_norm", "rho_freq_cos_u1",
                    "sig_gene_norm_ratio"]].copy()
        A(b3t.to_markdown(index=False, floatfmt=".3f") + "\n")
        hm = real[real.code_path == "cytokine_mil"].rho_freq_norm.mean()
        hc = real[real.code_path == "cascadir"].rho_freq_norm.mean()
        b3 = bool(np.isfinite(hc) and np.isfinite(hm) and hc > 0.2 and hc > hm + 0.15)
        A(f"\nMean rho(signature frequency, gene response norm): cascadir {hc:.3f}, "
          f"cytokine_mil {hm:.3f}. **B3 {'SUPPORTED' if b3 else 'NOT SUPPORTED'}**.\n")
    else:
        b3 = False
        A("No signature tables resolved — the gene-level link could not be computed.\n")

    # ---- B4 -----------------------------------------------------------------
    A("\n## B4 — or is the geometry simply flat?\n")
    spread = float(real.raw_pr_frac.max() / max(real.raw_pr_frac.min(), 1e-12))
    A(f"PR/d across fits: min {real.raw_pr_frac.min():.3f}, max "
      f"{real.raw_pr_frac.max():.3f} (ratio {spread:.2f}x).\n")
    b4 = bool(spread < 1.5 and not b1 and not b2)
    A(f"\n**B4 {'FIRES' if b4 else 'does not fire'}** — a flat geometry would mean the "
      "encoder is not the bottleneck and the collapse lives downstream (attention/"
      "classifier head, or the IG path).\n")

    # ---- untrained reference ------------------------------------------------
    unt = geo[geo.fit.str.endswith("__untrained") & (geo.probe == "onehot")]
    if len(unt):
        A("\n## Untrained reference\n")
        A(f"Matched-dimension untrained encoders: PR/d mean {unt.raw_pr_frac.mean():.3f} "
          f"(range {unt.raw_pr_frac.min():.3f}-{unt.raw_pr_frac.max():.3f}). An untrained "
          "encoder is a product of random matrices, whose spectrum already concentrates, "
          "so this — not the embedding dimension — is the ceiling that trained values "
          "should be read against.\n")

    A("\n## Verdict\n")
    if b1 and b2:
        A("The encoder's gene-space geometry both differs between healthy and collapsed "
          "fits AND tracks signature diversity within a controlled panel. The bottleneck "
          "hypothesis is supported.\n")
    elif b1 or b2:
        A("Partial support: one of the two structural predictions holds. Report as "
          "suggestive, not established.\n")
    else:
        A("Neither B1 nor B2 holds. The encoder's gene-space geometry does not explain "
          "the signature collapse; the search moves downstream of the encoder.\n")
    A("\nCorrelation across fits, not causation. `s1sweep_pub_replica` is diagnostic-only "
      "(it violates CLAUDE.md §16 by design) and must never seed a production fit. "
      "Direction != existence != causation (§26.4) carries over.\n")

    figs = make_figures(geo, spec, out, args.top_n)
    if figs:
        A("\n## Figures\n")
        for f in figs:
            A(f"- `plots/{f}`")
        A("")

    report = Path(args.report) if args.report else out / "ENCODER_GEOMETRY.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[write] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
