"""
Recurrent-IG analysis (CLAUDE.md §31, Part C) — depends on the Part B training jobs.

Reads the per-seed ``ig_traj.parquet`` from scripts/run_recurrent_ig_oesinghaus.py,
the Oesinghaus cells (train donors), and the audited directional labels, then:

  * builds the recruitment table (tau_in / tau_out / stability / volatility / category),
  * tests the five pre-registered predictions P-A..P-E,
  * builds the per-epoch degree-corrected coupling panel via cascadir.coupling_trajectory
    (cross_asym over training) + the §26 final-epoch direction regression check,
  * renders the figure set,
  * writes reports/recurrent_ig/RECURRENT_IG_RESULTS.md (verdict + objective read + caveats).

Operationalizations are locked in reports/recurrent_ig/PRE_REGISTRATION.md.

Usage:
    python scripts/analyze_recurrent_ig.py \
        --ig_dir results/recurrent_ig \
        --output_dir results/recurrent_ig \
        --report reports/recurrent_ig/RECURRENT_IG_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))

from cytokine_mil.analysis.oesinghaus_cell_loader import load_oesinghaus_cells_by_pair  # noqa: E402
from cascadir.dynamics import coupling_trajectory  # noqa: E402
from cascadir.types import Signature  # noqa: E402

# ---------------------------------------------------------------------------
# Locked operationalizations (mirror PRE_REGISTRATION.md)
# ---------------------------------------------------------------------------
TOP_K = 50                 # "in the signature" band
PERSIST = 0.8              # fraction of remaining checkpoints a gene must stay in band
EARLY_FRAC, LATE_FRAC = 1 / 3, 2 / 3   # thirds of the epoch range -> Anchor / Climber
SHARED_FRAC = 0.25         # gene in top-K of >= this fraction of cytokines == shared-activation
SEED_STABLE_MIN = 2        # final-epoch top-K membership in >= this many seeds
N_PERM = 1000
PERM_SEED = 123
VAL_DONORS = ["Donor2", "Donor3"]
MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
AUDITED_CSV = REPO_ROOT / "reports" / "cascade_pairs" / "cytokine_axes_audited.csv"

# Marker panel for P-B (from run_binary_ig_probe.py).
MARKER_PANEL = {
    "type_I_IFN_ISGs": (["ISG15", "IFIT2", "IFIT3", "RSAD2"], ["IFN-beta"]),
    "IFN_gamma_STAT1": (["CXCL9", "CXCL10", "CXCL11", "GBP1", "GBP5", "STAT1"], ["IFN-gamma"]),
    "NFkB_direct": (["TNF", "IL1B", "CXCL8", "BIRC3", "CCL3", "CCL4"], ["IL-1-beta", "TNF-alpha"]),
    "IL2_STAT5": (["IL2RA"], ["IL-2"]),
    "STAT3_direct": (["SOCS2"], ["IL-6", "IL-10"]),
}
EXEMPLARS = ["IFN-beta", "TNF-alpha", "IL-12", "IL-6"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_traj(ig_dir: Path) -> pd.DataFrame:
    paths = sorted(ig_dir.glob("seed_*/ig_traj.parquet")) or sorted(ig_dir.glob("**/ig_traj.parquet"))
    if not paths:
        raise FileNotFoundError(f"No ig_traj.parquet under {ig_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return df, [str(p) for p in paths]


# ---------------------------------------------------------------------------
# Recruitment per (cytokine, gene, seed)
# ---------------------------------------------------------------------------


def _rank_series(sub: pd.DataFrame, epochs: list, big: int) -> np.ndarray:
    """rank over the full epoch grid for one (cytokine, gene, seed); censored -> big."""
    m = dict(zip(sub["epoch"], sub["rank_ig"]))
    return np.array([m.get(e, big) for e in epochs], dtype=float)


def _tau_in(member: np.ndarray, epochs: list):
    """First epoch where the gene is in-band and stays in-band >= PERSIST of the rest."""
    T = len(epochs)
    for i in range(T):
        if member[i] and member[i:].mean() >= PERSIST:
            return epochs[i]
    return np.nan


def build_recruitment(df: pd.DataFrame, epochs: list, big: int) -> pd.DataFrame:
    rows = []
    e_early = epochs[0] + EARLY_FRAC * (epochs[-1] - epochs[0])
    e_late = epochs[0] + LATE_FRAC * (epochs[-1] - epochs[0])
    final_e = epochs[-1]
    for (cyt, seed), g in df.groupby(["cytokine", "seed"]):
        for gene, sub in g.groupby("gene"):
            ranks = _rank_series(sub, epochs, big)
            member = ranks < TOP_K
            if not member.any():
                continue
            tin = _tau_in(member, epochs)
            in_band_epochs = [e for e, mm in zip(epochs, member) if mm]
            tout = in_band_epochs[-1] if in_band_epochs else np.nan
            final_member = bool(member[epochs.index(final_e)])
            stab = float(member.mean())
            band_ranks = ranks[ranks < big]
            vol = float(np.median(np.abs(band_ranks - np.median(band_ranks)))) if band_ranks.size else np.nan
            if np.isfinite(tin) and tin <= e_early and final_member:
                cat = "Anchor"
            elif np.isfinite(tin) and tin >= e_late and final_member:
                cat = "Climber"
            elif (min(in_band_epochs) <= e_early) and not final_member:
                cat = "Flicker"
            else:
                cat = "Mid"
            rows.append({
                "cytokine": cyt, "seed": seed, "gene": gene, "tau_in": tin,
                "tau_out": tout, "stab": stab, "vol": vol,
                "final_member": final_member, "category": cat,
            })
    return pd.DataFrame(rows)


def seed_aggregate(rec: pd.DataFrame) -> pd.DataFrame:
    """Per (cytokine, gene): median tau_in, seed counts, seed-stable flag."""
    rows = []
    for (cyt, gene), g in rec.groupby(["cytokine", "gene"]):
        tins = g["tau_in"].dropna().to_numpy()
        n_final = int(g["final_member"].sum())
        rows.append({
            "cytokine": cyt, "gene": gene,
            "tau_in_median": float(np.median(tins)) if tins.size else np.nan,
            "n_seeds_recruited": int(g["tau_in"].notna().sum()),
            "n_seeds_final": n_final,
            "seed_stable": n_final >= SEED_STABLE_MIN,
            "category_mode": g["category"].mode().iloc[0] if len(g) else "Mid",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Effect size (P-A) and per-epoch top-K sets
# ---------------------------------------------------------------------------


def effect_sizes(cells_by_pair, gene_names, cytokines, pbs_label="PBS"):
    """mean(log-expr in cytokine) - mean(log-expr in PBS) per (cytokine, gene)."""
    def pooled_mean(cyt):
        arrs = [v for (c, _ct), v in cells_by_pair.items() if c == cyt]
        return np.concatenate(arrs, 0).mean(0) if arrs else None
    pbs = pooled_mean(pbs_label)
    out = {}
    for c in cytokines:
        mc = pooled_mean(c)
        if mc is None or pbs is None:
            continue
        out[c] = dict(zip(gene_names, mc - pbs))
    return out


def topk_sets(df: pd.DataFrame, epoch: int):
    """{(cytokine, seed): set(top-K genes)} at a given epoch."""
    sub = df[(df["epoch"] == epoch) & (df["rank_ig"] < TOP_K)]
    out = {}
    for (cyt, seed), g in sub.groupby(["cytokine", "seed"]):
        out[(cyt, seed)] = set(g["gene"])
    return out


# ---------------------------------------------------------------------------
# Coupling / cross_asym trajectory via cascadir (dogfood)
# ---------------------------------------------------------------------------


def coupling_over_epochs(df, cells_by_pair, gene_names, epochs, seeds):
    """Per-seed cascadir coupling_trajectory, averaged across seeds -> per (epoch,pair)."""
    frames = []
    for seed in seeds:
        ds = df[df["seed"] == seed]
        sigs_by_epoch = {}
        for e in epochs:
            sub = ds[(ds["epoch"] == e) & (ds["rank_ig"] < TOP_K)]
            per = {}
            for cyt, g in sub.groupby("cytokine"):
                gg = g.sort_values("rank_ig")
                per[cyt] = Signature(cyt, tuple(gg["gene"]), tuple(gg["ig"]), TOP_K)
            sigs_by_epoch[e] = per
        panel = coupling_trajectory(sigs_by_epoch, cells_by_pair, tuple(gene_names),
                                    control_label="PBS", degree_correct=True)
        for e, d in panel.items():
            if len(d):
                d = d.copy()
                d["seed"] = seed
                frames.append(d)
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    agg = (allp.groupby(["epoch", "condition_a", "condition_b"])
           [["coupling", "cross_asym", "coupling_raw"]].mean().reset_index())
    return agg


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def predict_A(seed_agg, eff):
    """tau_in vs |effect size| (seed-stable members). Expect NEGATIVE Spearman."""
    xs, ys, per_cyt = [], [], []
    for cyt, g in seed_agg[seed_agg["seed_stable"]].groupby("cytokine"):
        emap = eff.get(cyt, {})
        cx, cy = [], []
        for _, r in g.iterrows():
            if not np.isfinite(r["tau_in_median"]) or r["gene"] not in emap:
                continue
            cx.append(r["tau_in_median"]); cy.append(abs(emap[r["gene"]]))
        if len(cx) >= 5:
            rho, _ = spearmanr(cx, cy)
            per_cyt.append((cyt, rho, len(cx)))
        xs += cx; ys += cy
    pooled_rho, pooled_p = (spearmanr(xs, ys) if len(xs) >= 10 else (np.nan, np.nan))
    med_cyt_rho = float(np.median([r for _, r, _ in per_cyt])) if per_cyt else np.nan
    return {"pooled_rho": float(pooled_rho), "pooled_p": float(pooled_p),
            "median_cyt_rho": med_cyt_rho, "n_points": len(xs),
            "per_cyt": per_cyt, "xs": xs, "ys": ys}


def predict_B(seed_agg, available, epochs):
    e_early = epochs[0] + EARLY_FRAC * (epochs[-1] - epochs[0])
    rows = []
    for _panel, (genes, winners) in MARKER_PANEL.items():
        for w in winners:
            if w not in available:
                continue
            sub = seed_agg[seed_agg["cytokine"] == w].set_index("gene")
            for gene in genes:
                if gene not in sub.index:
                    rows.append({"cytokine": w, "gene": gene, "tau_in": np.nan,
                                 "is_anchor": False, "present": False})
                    continue
                tin = sub.loc[gene, "tau_in_median"]
                rows.append({"cytokine": w, "gene": gene, "tau_in": tin,
                             "is_anchor": bool(np.isfinite(tin) and tin <= e_early
                                               and sub.loc[gene, "category_mode"] == "Anchor"),
                             "present": True})
    df = pd.DataFrame(rows)
    present = df[df["present"]]
    frac_anchor = float(present["is_anchor"].mean()) if len(present) else np.nan
    return {"table": df, "frac_anchor": frac_anchor, "n_markers_present": int(len(present))}


def predict_C(df, seed_agg, epochs, seeds, available):
    """Promiscuity decay + specificity-entropy sharpening (shared vs specific genes)."""
    final_e = epochs[-1]
    n_cyt = len(available)
    shared_n = max(2, int(np.ceil(SHARED_FRAC * n_cyt)))
    # promiscuity per gene at final epoch (mean over seeds)
    final_sets = topk_sets(df, final_e)
    prom_final = {}
    for (cyt, seed), s in final_sets.items():
        for g in s:
            prom_final[g] = prom_final.get(g, 0) + 1
    for g in prom_final:
        prom_final[g] /= len(seeds)
    shared_genes = {g for g, p in prom_final.items() if p >= shared_n}
    specific_genes = {g for g, p in prom_final.items() if 0 < p <= 1.0}

    def mean_prom(gene_set, epoch):
        sets = topk_sets(df, epoch)
        counts = {}
        for (_cyt, _seed), s in sets.items():
            for g in s & gene_set:
                counts[g] = counts.get(g, 0) + 1
        vals = [c / len(seeds) for c in counts.values()]
        return float(np.mean(vals)) if vals else 0.0

    prom_curve = {e: (mean_prom(shared_genes, e), mean_prom(specific_genes, e)) for e in epochs}

    # specificity entropy H_g(t): entropy over cytokines of positive IG (seed-mean) where in band
    ent_curve = {}
    for e in epochs:
        sub = df[(df["epoch"] == e) & (df["rank_ig"] < TOP_K)]
        piv = (sub.groupby(["gene", "cytokine"])["ig"].mean()
               .clip(lower=0).reset_index())
        Hs = []
        for _g, gg in piv.groupby("gene"):
            w = gg["ig"].to_numpy()
            tot = w.sum()
            if tot <= 0 or len(w) < 2:
                continue
            p = w / tot
            Hs.append(float(-(p * np.log(p + 1e-12)).sum()))
        ent_curve[e] = float(np.mean(Hs)) if Hs else np.nan

    shared_slope = np.polyfit(epochs, [prom_curve[e][0] for e in epochs], 1)[0]
    ent_vals = [ent_curve[e] for e in epochs]
    ent_slope = np.polyfit(epochs, ent_vals, 1)[0] if all(np.isfinite(ent_vals)) else np.nan
    return {"prom_curve": prom_curve, "ent_curve": ent_curve,
            "n_shared": len(shared_genes), "n_specific": len(specific_genes),
            "shared_prom_slope": float(shared_slope), "entropy_slope": float(ent_slope)}


def _tau_in_lookup(rec):
    """{(cytokine, gene, seed): tau_in}."""
    return {(r.cytokine, r.gene, r.seed): r.tau_in for r in rec.itertuples()}


def predict_D(rec, df, labeled, coup_final, epochs, seeds):
    """Delta-tau direction vs cross_asym/expected_sign + timing-permutation null."""
    tau = _tau_in_lookup(rec)
    final_e = epochs[-1]
    final_sets = topk_sets(df, final_e)

    def shared_genes(a, b):
        sa = set().union(*[final_sets.get((a, s), set()) for s in seeds]) if seeds else set()
        sb = set().union(*[final_sets.get((b, s), set()) for s in seeds]) if seeds else set()
        return sa & sb

    def mean_dtau(a, b, tau_map):
        vals = []
        for g in shared_genes(a, b):
            for s in seeds:
                ta = tau_map.get((a, g, s), np.nan)
                tb = tau_map.get((b, g, s), np.nan)
                if np.isfinite(ta) and np.isfinite(tb):
                    vals.append(tb - ta)   # >0 => a earlier => a upstream
        return float(np.mean(vals)) if vals else np.nan

    rows = []
    cmap = {(r.condition_a, r.condition_b): r.cross_asym for r in coup_final.itertuples()} if len(coup_final) else {}
    for _, L in labeled.iterrows():
        a, b, exp = L["axis_a"], L["axis_b"], int(L["expected_sign"])
        dt = mean_dtau(a, b, tau)
        xa = cmap.get((a, b), np.nan)
        rows.append({"axis_a": a, "axis_b": b, "expected_sign": exp,
                     "mean_dtau": dt, "cross_asym": xa,
                     "dtau_sign": int(np.sign(dt)) if np.isfinite(dt) else 0,
                     "cross_sign": int(np.sign(xa)) if np.isfinite(xa) else 0})
    res = pd.DataFrame(rows)
    usable = res[res["dtau_sign"] != 0]
    agree_cross = float((usable["dtau_sign"] == usable["cross_sign"]).mean()) if len(usable) else np.nan
    agree_exp = float((usable["dtau_sign"] == usable["expected_sign"]).mean()) if len(usable) else np.nan

    # permutation null: shuffle epoch labels within each (cytokine, gene, seed) trajectory
    rng = np.random.default_rng(PERM_SEED)
    big = max(df["rank_ig"]) + 1
    # precompute per (cyt,gene,seed) rank arrays for genes in labeled shared sets
    needed = set()
    pairs = list(zip(usable["axis_a"], usable["axis_b"]))
    for a, b in pairs:
        for g in shared_genes(a, b):
            for s in seeds:
                needed.add((a, g, s)); needed.add((b, g, s))
    rank_arrays = {}
    for (cyt, gene, seed) in needed:
        sub = df[(df["cytokine"] == cyt) & (df["gene"] == gene) & (df["seed"] == seed)]
        rank_arrays[(cyt, gene, seed)] = _rank_series(sub, epochs, big)

    def perm_agree():
        tau_perm = {}
        for key, arr in rank_arrays.items():
            a2 = rng.permutation(arr)
            tau_perm[key] = _tau_in(a2 < TOP_K, epochs)
        signs = []
        for _, L in usable.iterrows():
            dt = mean_dtau(L["axis_a"], L["axis_b"], tau_perm)
            if np.isfinite(dt) and np.sign(dt) != 0:
                signs.append(int(np.sign(dt)) == int(L["expected_sign"]))
        return float(np.mean(signs)) if signs else np.nan

    null = np.array([perm_agree() for _ in range(N_PERM)])
    null = null[np.isfinite(null)]
    p_perm = float(np.mean(null >= agree_exp)) if (null.size and np.isfinite(agree_exp)) else np.nan
    return {"table": res, "agree_cross": agree_cross, "agree_exp": agree_exp,
            "n_usable": int(len(usable)), "null_mean": float(np.mean(null)) if null.size else np.nan,
            "p_perm": p_perm}


def predict_E(df, labeled, prom_final_epoch, epochs, seeds):
    """Early shared-program score per labeled pair vs direction correctness."""
    early_e = epochs[0]
    early_sets = topk_sets(df, early_e)

    def early_shared(a, b):
        sa = set().union(*[early_sets.get((a, s), set()) for s in seeds]) if seeds else set()
        sb = set().union(*[early_sets.get((b, s), set()) for s in seeds]) if seeds else set()
        inter = sa & sb
        if not (sa or sb):
            return np.nan
        return len(inter) / max(1, len(sa | sb))

    rows = []
    for _, L in labeled.iterrows():
        rows.append({"axis_a": L["axis_a"], "axis_b": L["axis_b"],
                     "early_shared": early_shared(L["axis_a"], L["axis_b"]),
                     "cross_correct": L.get("cross_correct", np.nan)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_rank_ribbons(df, rec, available, epochs, plots, seeds):
    ex = [c for c in EXEMPLARS if c in available][:4] or list(available)[:4]
    fig, axes = plt.subplots(1, len(ex), figsize=(5 * len(ex), 4.2), sharey=True)
    if len(ex) == 1:
        axes = [axes]
    cat_color = {"Anchor": "tab:blue", "Climber": "tab:red", "Flicker": "tab:orange", "Mid": "0.7"}
    big = max(df["rank_ig"]) + 1
    for ax, cyt in zip(axes, ex):
        recc = rec[(rec["cytokine"] == cyt) & rec["final_member"]]
        top_genes = (recc.groupby("gene")["stab"].mean().sort_values(ascending=False).head(25).index)
        for gene in top_genes:
            sub = df[(df["cytokine"] == cyt) & (df["gene"] == gene)]
            r = sub.groupby("epoch")["rank_ig"].mean().reindex(epochs).fillna(big)
            cat = recc[recc["gene"] == gene]["category"].mode()
            color = cat_color.get(cat.iloc[0] if len(cat) else "Mid", "0.7")
            ax.plot(epochs, r.values, color=color, alpha=0.6, lw=1.0)
        ax.axhline(TOP_K, color="k", ls=":", lw=0.8)
        ax.set_ylim(0, min(big, 120)); ax.invert_yaxis()
        ax.set_title(cyt); ax.set_xlabel("epoch")
    axes[0].set_ylabel("IG rank (1=top; lower=better)")
    handles = [plt.Line2D([], [], color=c, label=k) for k, c in cat_color.items()]
    axes[-1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle("Gene IG-rank trajectories (final-signature members), seed-mean")
    fig.tight_layout()
    fig.savefig(plots / "fig1_rank_ribbons.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_A(resA, plots):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    if resA["xs"]:
        ax.scatter(resA["xs"], resA["ys"], s=8, alpha=0.3)
    ax.set_xlabel("tau_in (recruitment epoch)"); ax.set_ylabel("|effect size| vs PBS")
    ax.set_title(f"P-A: tau_in vs effect size\npooled Spearman rho={resA['pooled_rho']:.3f} "
                 f"(p={resA['pooled_p']:.1e}), n={resA['n_points']}")
    fig.tight_layout(); fig.savefig(plots / "fig3_tauin_vs_effect.png", dpi=150); plt.close(fig)


def fig_B(resB, plots):
    df = resB["table"]; df = df[df["present"]]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(df) + 1))
    labels = [f"{r.gene} @ {r.cytokine}" for r in df.itertuples()]
    colors = ["tab:blue" if a else "0.6" for a in df["is_anchor"]]
    ax.barh(range(len(df)), df["tau_in"].fillna(0), color=colors)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("tau_in (epoch)")
    ax.set_title(f"P-B: marker recruitment (blue=Anchor); anchor frac={resB['frac_anchor']:.2f}")
    fig.tight_layout(); fig.savefig(plots / "fig2_marker_recruitment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_C(resC, epochs, plots):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(epochs, [resC["prom_curve"][e][0] for e in epochs], "-o", label=f"shared (n={resC['n_shared']})")
    axs[0].plot(epochs, [resC["prom_curve"][e][1] for e in epochs], "-o", label=f"specific (n={resC['n_specific']})")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("mean promiscuity P(g,t)")
    axs[0].set_title(f"P-C: promiscuity (shared slope={resC['shared_prom_slope']:.2e})"); axs[0].legend(fontsize=8)
    axs[1].plot(epochs, [resC["ent_curve"][e] for e in epochs], "-o", color="tab:purple")
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("mean specificity entropy H_g(t)")
    axs[1].set_title(f"P-C: sharpening (slope={resC['entropy_slope']:.2e})")
    fig.tight_layout(); fig.savefig(plots / "fig4_promiscuity_sharpening.png", dpi=150); plt.close(fig)


def fig_D(resD, plots):
    df = resD["table"].dropna(subset=["mean_dtau", "cross_asym"])
    fig, ax = plt.subplots(figsize=(5.4, 5))
    if len(df):
        agree = df["dtau_sign"] == df["cross_sign"]
        ax.scatter(df["cross_asym"], df["mean_dtau"],
                   c=["tab:green" if a else "tab:red" for a in agree], s=40)
        for r in df.itertuples():
            ax.annotate(f"{r.axis_a[:6]}/{r.axis_b[:6]}", (r.cross_asym, r.mean_dtau), fontsize=6)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("cross_asym (final)"); ax.set_ylabel("mean delta-tau (shared genes)")
    ax.set_title(f"P-D: timing vs magnitude direction\nagree(cross)={resD['agree_cross']:.2f} "
                 f"agree(expected)={resD['agree_exp']:.2f} p_perm={resD['p_perm']:.3f}")
    fig.tight_layout(); fig.savefig(plots / "fig6_dtau_vs_crossasym.png", dpi=150); plt.close(fig)


def fig_crossasym_epoch(coup, labeled, plots):
    if not len(coup):
        return
    lab = {(r.axis_a, r.axis_b) for r in labeled.itertuples()}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for (a, b), g in coup.groupby(["condition_a", "condition_b"]):
        if (a, b) in lab:
            ax.plot(g["epoch"], g["cross_asym"], "-o", lw=1, ms=3, label=f"{a[:5]}/{b[:5]}")
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("epoch"); ax.set_ylabel("cross_asym")
    ax.set_title("P-D aux: cross_asym vs epoch (labeled pairs) — does direction stabilize?")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(plots / "fig8_crossasym_over_epoch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_E(resE, plots):
    df = resE.dropna(subset=["early_shared"])
    if df.empty or df["cross_correct"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(5, 4.4))
    for correct, sub in df.groupby("cross_correct"):
        ax.scatter([correct] * len(sub), sub["early_shared"], alpha=0.6,
                   label="correct" if correct else "wrong")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["wrong", "correct"])
    ax.set_ylabel("early shared-program score (epoch=first)")
    ax.set_title("P-E: early signature overlap vs direction correctness")
    fig.tight_layout(); fig.savefig(plots / "fig7_collapse_warning.png", dpi=150); plt.close(fig)


def fig_categories(rec, available, plots):
    cyts = [c for c in available][:24]
    cats = ["Anchor", "Climber", "Flicker", "Mid"]
    colors = {"Anchor": "tab:blue", "Climber": "tab:red", "Flicker": "tab:orange", "Mid": "0.7"}
    data = {cat: [] for cat in cats}
    for c in cyts:
        sub = rec[rec["cytokine"] == c]
        # per-gene mode category, dedup genes
        gcat = sub.groupby("gene")["category"].agg(lambda s: s.mode().iloc[0] if len(s) else "Mid")
        vc = gcat.value_counts()
        for cat in cats:
            data[cat].append(int(vc.get(cat, 0)))
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(cyts)), 4.6))
    bottom = np.zeros(len(cyts))
    for cat in cats:
        ax.bar(cyts, data[cat], bottom=bottom, color=colors[cat], label=cat)
        bottom += np.array(data[cat])
    ax.set_ylabel("# genes ever in top-50"); ax.set_title("Gene category composition per cytokine")
    ax.legend(fontsize=8); plt.xticks(rotation=90, fontsize=7)
    fig.tight_layout(); fig.savefig(plots / "fig9_category_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def verdict(resA, resB, resC, resD, regress, labeled_n):
    def gate(cond_green, cond_amber):
        return "GREEN" if cond_green else ("AMBER" if cond_amber else "RED")
    vA = gate(resA["pooled_rho"] < -0.1 and resA["pooled_p"] < 0.05, resA["pooled_rho"] < 0)
    vB = gate(resB["frac_anchor"] >= 0.6, resB["frac_anchor"] >= 0.4)
    vC = gate(resC["entropy_slope"] < 0 and resC["shared_prom_slope"] <= 0,
              resC["entropy_slope"] < 0)
    vD = gate(np.isfinite(resD["p_perm"]) and resD["p_perm"] < 0.05 and resD["agree_exp"] >= 0.6,
              np.isfinite(resD["agree_exp"]) and resD["agree_exp"] > 0.5)
    overall = ("GREEN" if (vA == "GREEN" and vD == "GREEN")
               else ("AMBER" if "GREEN" in (vA, vB, vC) else "RED"))
    return {"P-A": vA, "P-B": vB, "P-C": vC, "P-D": vD, "overall": overall}


def write_report(path, info):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    resA, resB, resC, resD, resE, regress, v, meta = (
        info["A"], info["B"], info["C"], info["D"], info["E"], info["regress"], info["verdict"], info["meta"])
    L = []
    L.append("# Recurrent IG over training dynamics — results (Oesinghaus)\n")
    L.append(f"_Auto-generated by scripts/analyze_recurrent_ig.py. Seeds: {meta['seeds']}. "
             f"Cytokines: {meta['n_cyt']}. Checkpoints: {meta['epochs']}._\n")
    L.append(f"**Overall verdict: {v['overall']}** "
             "(GREEN = recurrent IG adds biology worth folding in; AMBER = descriptive only; "
             "RED = redundant with the static probe).\n")
    L.append("## §26 regression check (faithfulness)\n")
    L.append(f"Final-epoch cross_asym signed accuracy on the {regress['n']} labeled non-AMBIGUOUS "
             f"directional pairs = **{regress['acc']:.2f}** (published §26: 15/17≈0.88). "
             f"{regress['note']}\n")
    L.append("## Pre-registered predictions\n")
    L.append("| ID | prediction | result | verdict |")
    L.append("|---|---|---|---|")
    L.append(f"| P-A | tau_in vs |effect|: negative Spearman | rho={resA['pooled_rho']:.3f} "
             f"(p={resA['pooled_p']:.1e}), median per-cytokine rho={resA['median_cyt_rho']:.3f}, "
             f"n={resA['n_points']} | {v['P-A']} |")
    L.append(f"| P-B | markers are Anchors in their cytokine | anchor frac={resB['frac_anchor']:.2f} "
             f"({resB['n_markers_present']} markers present) | {v['P-B']} |")
    L.append(f"| P-C | sharpening: H_g(t) falls; shared promiscuity decays | "
             f"entropy slope={resC['entropy_slope']:.2e}, shared-prom slope={resC['shared_prom_slope']:.2e} "
             f"(n_shared={resC['n_shared']}, n_specific={resC['n_specific']}) | {v['P-C']} |")
    L.append(f"| P-D | delta-tau sign agrees with direction, beats null | "
             f"agree(cross)={resD['agree_cross']:.2f}, agree(expected)={resD['agree_exp']:.2f}, "
             f"perm-null mean={resD['null_mean']:.2f}, p={resD['p_perm']:.3f}, n={resD['n_usable']} | {v['P-D']} |")
    L.append("")
    L.append("## Objective — what visually emerges (independent of the hypotheses)\n")
    L.append(info["objective"])
    L.append("\n## Honest caveats\n")
    L.append("- **Read-out learning, frozen encoder.** Recruitment order reflects the "
             "attention/classifier learning to weight a FIXED gene→feature map, not the "
             "representation drifting (CLAUDE.md §31).")
    L.append("- **Not donor-level.** IG is computed on pooled train tubes per checkpoint; the "
             "robustness axis here is seed-stability (3 seeds), NOT a donor bootstrap of "
             "recruitment timing. Recruitment-timing donor-robustness is future work.")
    L.append("- **Small n on labeled pairs** (P-D/§26): the directional benchmark is a handful "
             "of pairs; treat accuracies as point estimates.")
    L.append("- **Correlational, not causal.** Cross-model 'sweeps' are coincidences of "
             "independent fits unless they survive as seed-stable, FDR-controlled structure; "
             "direction is consistent-with, not proof-of, causation.")
    L.append(f"\n_Figures in `{meta['plots']}` ; stats in `{meta['stats']}`._\n")
    p.write_text("\n".join(L))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ig_dir", type=str, default="results/recurrent_ig")
    ap.add_argument("--output_dir", type=str, default="results/recurrent_ig")
    ap.add_argument("--report", type=str, default="reports/recurrent_ig/RECURRENT_IG_RESULTS.md")
    ap.add_argument("--manifest_path", type=str, default=MANIFEST_PATH)
    ap.add_argument("--hvg_path", type=str, default=HVG_PATH)
    args = ap.parse_args()

    out = Path(args.output_dir); stats = out / "stats"; plots = out / "plots"
    stats.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)

    df, traj_paths = load_traj(Path(args.ig_dir))
    print(f"Loaded ig_traj: {len(df)} rows from {traj_paths}", flush=True)
    epochs = sorted(df["epoch"].unique().tolist())
    seeds = sorted(df["seed"].unique().tolist())
    available = sorted(df["cytokine"].unique().tolist())
    big = int(df["rank_ig"].max()) + 1

    # recruitment
    rec = build_recruitment(df, epochs, big)
    rec.to_csv(stats / "recruitment_table.csv", index=False)
    seed_agg = seed_aggregate(rec)
    seed_agg.to_csv(stats / "recruitment_seed_aggregate.csv", index=False)
    (rec.groupby(["cytokine", "category"]).size().unstack(fill_value=0)
     .to_csv(stats / "gene_categories.csv"))

    # cells (train donors) — heavy
    print("Loading Oesinghaus cells (train donors)...", flush=True)
    cells_by_pair, gene_names = load_oesinghaus_cells_by_pair(
        manifest_path=args.manifest_path, cytokines=available, hvg_path=args.hvg_path,
        pbs_label="PBS", exclude_donors=VAL_DONORS)

    # P-A effect sizes
    eff = effect_sizes(cells_by_pair, gene_names, available)
    resA = predict_A(seed_agg, eff)
    resB = predict_B(seed_agg, available, epochs)
    resC = predict_C(df, seed_agg, epochs, seeds, available)

    # coupling / cross_asym trajectory (cascadir dogfood)
    print("Building coupling trajectory via cascadir...", flush=True)
    coup = coupling_over_epochs(df, cells_by_pair, gene_names, epochs, seeds)
    coup.to_csv(stats / "coupling_traj.csv", index=False)
    coup_final = coup[coup["epoch"] == epochs[-1]] if len(coup) else pd.DataFrame()

    # labeled directional pairs
    aud = pd.read_csv(AUDITED_CSV)
    labeled = aud[(aud.get("counts_in_benchmark", False) == True)].copy()  # noqa: E712
    labeled = labeled[labeled["expected_sign"].isin([1, -1])]
    labeled = labeled[labeled["axis_a"].isin(available) & labeled["axis_b"].isin(available)]
    print(f"Labeled directional pairs usable: {len(labeled)}", flush=True)

    # §26 regression check on final cross_asym
    cmap = {(r.condition_a, r.condition_b): r.cross_asym for r in coup_final.itertuples()} if len(coup_final) else {}
    reg_rows = []
    for _, Lr in labeled.iterrows():
        xa = cmap.get((Lr["axis_a"], Lr["axis_b"]), np.nan)
        correct = (np.sign(xa) == Lr["expected_sign"]) if np.isfinite(xa) else np.nan
        reg_rows.append({"axis_a": Lr["axis_a"], "axis_b": Lr["axis_b"],
                         "expected_sign": int(Lr["expected_sign"]), "cross_asym": xa,
                         "cross_correct": correct})
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(stats / "regression_check.csv", index=False)
    reg_usable = reg_df.dropna(subset=["cross_asym"])
    reg_acc = float((np.sign(reg_usable["cross_asym"]) == reg_usable["expected_sign"]).mean()) if len(reg_usable) else np.nan
    regress = {"n": int(len(reg_usable)), "acc": reg_acc,
               "note": ("Reproduces the §26 direction signal." if reg_acc >= 0.75
                        else "REGRESSION vs §26 — final-epoch signatures under-reproduce; "
                             "interpret trajectory claims with care (P4-style).")}
    labeled = labeled.merge(reg_df[["axis_a", "axis_b", "cross_correct"]], on=["axis_a", "axis_b"], how="left")

    resD = predict_D(rec, df, labeled, coup_final, epochs, seeds)
    resD["table"].to_csv(stats / "delta_tau_direction.csv", index=False)
    resE = predict_E(df, labeled, None, epochs, seeds)
    resE.to_csv(stats / "collapse_warning.csv", index=False)

    v = verdict(resA, resB, resC, resD, regress, len(labeled))
    pd.DataFrame([{"prediction": k, "verdict": val} for k, val in v.items()]).to_csv(
        stats / "predictions_verdict.csv", index=False)

    # figures
    print("Rendering figures...", flush=True)
    fig_rank_ribbons(df, rec, available, epochs, plots, seeds)
    fig_A(resA, plots); fig_B(resB, plots); fig_C(resC, epochs, plots)
    fig_D(resD, plots); fig_crossasym_epoch(coup, labeled, plots); fig_E(resE, plots)
    fig_categories(rec, available, plots)

    # objective read (data-driven sentences)
    obj = []
    anchor_frac = (rec["category"] == "Anchor").mean()
    climber_frac = (rec["category"] == "Climber").mean()
    flicker_frac = (rec["category"] == "Flicker").mean()
    obj.append(f"- Of all (gene×cytokine×seed) members ever in top-{TOP_K}, "
               f"{anchor_frac:.0%} are Anchors (early & stable), {climber_frac:.0%} Climbers "
               f"(late arrivals), {flicker_frac:.0%} Flickers (early then dropped).")
    obj.append(f"- Mean specificity entropy moves from {resC['ent_curve'][epochs[0]]:.2f} at "
               f"epoch {epochs[0]} to {resC['ent_curve'][epochs[-1]]:.2f} at epoch {epochs[-1]} "
               f"(slope {resC['entropy_slope']:.2e}/epoch): "
               f"{'sharpening' if resC['entropy_slope'] < 0 else 'no sharpening / diffusing'}.")
    obj.append(f"- Shared-activation genes (in top-{TOP_K} of ≥{max(2, int(np.ceil(SHARED_FRAC*len(available))))} "
               f"cytokines) number {resC['n_shared']}; their mean promiscuity slope over training is "
               f"{resC['shared_prom_slope']:.2e} ({'decaying' if resC['shared_prom_slope']<0 else 'flat/rising'}).")
    if np.isfinite(resD["agree_exp"]):
        obj.append(f"- On {resD['n_usable']} labeled pairs, recruitment-timing (Δτ) recovers the known "
                   f"direction {resD['agree_exp']:.0%} of the time and agrees with the magnitude "
                   f"cross_asym {resD['agree_cross']:.0%} of the time "
                   f"(permutation null {resD['null_mean']:.0%}, p={resD['p_perm']:.3f}).")
    obj.append(f"- Final-epoch cross_asym reproduces the known direction on {regress['n']} pairs at "
               f"{regress['acc']:.0%} (§26 anchor ≈88%).")
    objective = "\n".join(obj)

    write_report(args.report, {
        "A": resA, "B": resB, "C": resC, "D": resD, "E": resE, "regress": regress,
        "verdict": v, "objective": objective,
        "meta": {"seeds": seeds, "n_cyt": len(available), "epochs": epochs,
                 "plots": str(plots), "stats": str(stats)},
    })
    print(f"Wrote report -> {args.report}", flush=True)
    print(f"Verdict: {v}", flush=True)


if __name__ == "__main__":
    main()
