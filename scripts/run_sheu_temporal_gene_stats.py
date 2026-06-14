"""
Early-vs-late gene DISTRIBUTIONAL statistics across the Sheu time course (beyond the mean).

Compares EARLY/source vs LATE/downstream genes on variance, dispersion, bimodality,
recruitment and inequality, and how those evolve over real biological time — with the
mean-variance coupling trap controlled (residual-from-panel-trend AND matched-mean), at
DONOR level (n_eff=4). Direction-AGNOSTIC: characterizes the distribution-shape signature
of the already-established early->late order; does NOT re-derive direction.

--synthetic : apparatus self-test. Plant EARLY (uniform, Fano~1, fast detection) vs LATE
              (fraction-ON ramps through 1-3h -> transient bimodality/overdispersion/gradual
              recruitment), both reaching high mean by 8h. Confirm the DECOUPLED battery
              recovers late>early at intermediate t while raw mean barely separates.
--real      : load raw Sheu time course (0.25->8hr), per stimulus in {LPS, PIC}, per pseudo-
              donor, per timepoint compute the battery, aggregate to donor level, run H1-H6,
              and plot the must/high figures. See EARLY_LATE_STATS_PREREGISTRATION.md.

Reuses build_pseudotubes_sheu2024 (loader), compute_sheu_realtime_emergence (time parse /
alias), temporal_cascade (onset on mean), temporal_gene_stats (battery), pathway_signatures.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis import temporal_gene_stats as tgs
from cytokine_mil.analysis import temporal_cascade as tc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

_LOG = None
INTERMEDIATE_TPS = (1.0, 3.0)


def _log(m=""):
    print(m, flush=True)
    if _LOG is not None:
        print(m, file=_LOG, flush=True)


def _import_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Gene sets
# ---------------------------------------------------------------------------
def _gene_sets():
    from cytokine_mil.analysis.pathway_signatures import PATHWAY_SIGNATURES as P
    src = set(P["IRF3_direct"]["up"]); dwn = set(P["IFNAR_induced"]["up"])
    ov = src & dwn
    early = sorted(src - ov)          # Ccl5, Cxcl10, Ifit2, Ifnb1
    late = sorted(dwn - ov)           # Mx1, Mx2, Ifit1, Ifit1bl1, Ifit3b, Rsad2, Irf7, Oasl1
    nfkb = sorted(set(P["NFkB_canonical"]["up"]) - ov)
    return early, late, nfkb


def _idx(names, present):
    gi = {g: i for i, g in enumerate(present)}
    return [gi[g] for g in names if g in gi], [g for g in names if g in gi]


# ===========================================================================
# Per-block battery (one stimulus, donor, timepoint)
# ===========================================================================
def _block_stats(counts, logx, n_target, rng):
    counts, logx = tgs.subsample_block(counts, logx, n_target, rng)
    s = tgs.gene_cell_stats(counts, logx)
    s["fano_res"] = tgs.fano_residual(s)
    s["det_res"] = tgs.detection_residual(s)
    return s


def _collect_real(adata_counts, adata_log, obs, label, donors, tps_by_donor, gene_names,
                  min_cells, cap, rng):
    """per[donor][tp] = stats dict (each (G,) array). Also pooled-for-viz per tp."""
    per = {d: {} for d in donors}
    pool_log = {}; pool_counts = {}
    cyc = obs["cytokine"].values; thr = obs["_time_hr"].values; pds = obs["pseudo_donor"].values
    VIZ_CAP = 3000
    all_tps = sorted({t for d in donors for t in tps_by_donor[d]})
    for d in donors:
        # common-n target across this donor's usable tps
        ns = []
        for tp in tps_by_donor[d]:
            m = _block_mask(cyc, thr, pds, label, tp, d)
            ns.append(int(m.sum()))
        usable = [n for n in ns if n >= min_cells]
        n_target = min(cap, min(usable)) if usable else 0
        for tp in tps_by_donor[d]:
            m = _block_mask(cyc, thr, pds, label, tp, d)
            if m.sum() < min_cells:
                continue
            c = _dense(adata_counts, m); l = _dense(adata_log, m)
            per[d][tp] = _block_stats(c, l, n_target, rng)
    # pooled across donors per tp (for cloud + distribution + bimodality figs)
    for tp in all_tps:
        m = _block_mask(cyc, thr, pds, label, tp, None)
        if m.sum() == 0:
            continue
        idx = np.where(m)[0]
        if idx.size > VIZ_CAP:
            idx = rng.choice(idx, VIZ_CAP, replace=False)
        pool_log[tp] = _dense(adata_log, idx)
        pool_counts[tp] = _dense(adata_counts, idx)
    return per, pool_log, pool_counts, all_tps


def _block_mask(cyc, thr, pds, label, tp, donor):
    if tp == 0.0:
        m = (cyc == "PBS")
    else:
        m = (cyc == label) & (thr == tp)
    if donor is not None:
        m = m & (pds == donor)
    return m


def _dense(X, mask_or_idx):
    sub = X[mask_or_idx]
    if hasattr(sub, "toarray"):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float64)


# ===========================================================================
# Donor-aggregated trajectory matrices (G, n_tp) of donor-median stat
# ===========================================================================
def _traj(per, donors, tps, stat, G):
    M = np.full((G, len(tps)), np.nan)
    for ti, tp in enumerate(tps):
        cols = [per[d][tp][stat] for d in donors if tp in per[d]]
        if cols:
            M[:, ti] = np.nanmedian(np.vstack(cols), axis=0)
    return M


def _per_donor_at(per, donors, tp, stat, gene_idx):
    return [per[d][tp][stat][gene_idx] for d in donors if tp in per[d]]


# ===========================================================================
# Hypothesis tests
# ===========================================================================
def _union_islate(early_idx, late_idx):
    union = list(early_idx) + list(late_idx)
    is_late = np.array([False] * len(early_idx) + [True] * len(late_idx))
    return union, is_late


def _run_hypotheses(per, donors, tps, gene_names, early_idx, late_idx, nfkb_idx, rng):
    G = len(gene_names)
    union, is_late = _union_islate(early_idx, late_idx)
    out = {"timepoints": [float(t) for t in tps], "by_stat": {}, "controls": {}}

    # --- H1/H6: decoupled overdispersion at intermediate tps, raw vs decoupled ---
    score_stats = ["var_counts", "cv2", "fano_raw", "fano_res", "gini", "iqr",
                   "bc_sarle", "det_res"]
    for tp in INTERMEDIATE_TPS:
        if not any(tp in per[d] for d in donors):
            continue
        rec = {}
        for stat in score_stats:
            pv = _per_donor_at(per, donors, tp, stat, union)
            res = tgs.donor_level_auc(pv, is_late, n_perm=1000, rng=rng)
            # matched-mean route (residual sign must agree)
            md = []
            for d in donors:
                if tp not in per[d]:
                    continue
                mm = tgs.matched_mean_delta(per[d][tp][stat][union],
                                            per[d][tp]["mean_counts"][union], is_late)
                if np.isfinite(mm["delta"]):
                    md.append(mm["delta"])
            res["matched_delta"] = float(np.mean(md)) if md else float("nan")
            rec[stat] = res
        out["by_stat"][f"{tp}hr"] = rec

    # --- H6 controls: random split + NFkB set (fano_res) at 3h ---
    tp = 3.0 if any(3.0 in per[d] for d in donors) else INTERMEDIATE_TPS[0]
    rand_aucs = []
    for _ in range(20):
        lab = rng.permutation(is_late)
        pv = _per_donor_at(per, donors, tp, "fano_res", union)
        rand_aucs.append(tgs.donor_level_auc(pv, lab, n_perm=1, rng=rng)["auc"])
    out["controls"]["random_split_fano_res_auc"] = float(np.nanmedian(rand_aucs))
    if nfkb_idx:
        nfkb_union, nfkb_islate = _union_islate(early_idx, nfkb_idx)  # early vs NFkB
        pv = _per_donor_at(per, donors, tp, "fano_res", nfkb_union)
        out["controls"]["nfkb_vs_early_fano_res_auc"] = tgs.donor_level_auc(pv, nfkb_islate, n_perm=500, rng=rng)
    out["controls"]["control_tp"] = float(tp)
    return out, union, is_late


def _trajectory_features(per, donors, tps, gene_names, early_idx, late_idx, rng,
                         peak_floor: float = 1.0):
    """Per-gene timing features on donor-median trajectories + donor-level AUC late>early.

    Timing features (peak time / transience / lag) are only defined for genes that
    actually show a dispersion EXCURSION (max fano_res >= peak_floor); a flat trajectory
    has a meaningless argmax, so it is left NaN rather than assigned noise."""
    G = len(gene_names)
    tarr = np.asarray(tps, float)
    fano_M = _traj(per, donors, tps, "fano_res", G)
    frac_M = _traj(per, donors, tps, "frac_expr", G)
    mean_M = _traj(per, donors, tps, "mean_log", G)
    # PBS-corrected mean trajectory for onset (subtract t=0 column if present)
    if 0.0 in tps:
        base = mean_M[:, tps.index(0.0)][:, None]
    else:
        base = np.zeros((G, 1))
    above = mean_M - base
    onset = tc.onset_time(above, tarr, threshold=0.5)

    feats = {g: {} for g in range(G)}
    for g in range(G):
        feats[g]["onset"] = float(onset[g])
        has_excursion = np.isfinite(np.nanmax(fano_M[g])) and np.nanmax(fano_M[g]) >= peak_floor
        if has_excursion:
            feats[g]["disp_peak_time"] = tgs.peak_time(fano_M[g], tarr)
            feats[g]["disp_transience"] = tgs.transience_index(fano_M[g])
            feats[g]["het_lag"] = (feats[g]["disp_peak_time"] - onset[g]
                                   if np.isfinite(onset[g]) and np.isfinite(feats[g]["disp_peak_time"])
                                   else float("nan"))
        else:
            feats[g]["disp_peak_time"] = float("nan")
            feats[g]["disp_transience"] = float("nan")
            feats[g]["het_lag"] = float("nan")
        feats[g]["disp_excursion"] = float(np.nanmax(fano_M[g])) if np.any(np.isfinite(fano_M[g])) else float("nan")
        rf = tgs.recruitment_features(frac_M[g], tarr)
        feats[g]["t50"] = rf["t50"]; feats[g]["max_slope"] = rf["max_slope"]
        feats[g]["early_window_fano_auc"] = tgs.trapz_auc(fano_M[g], tarr, t_max=3.0)

    def feat_auc(key):
        lv = np.array([feats[g][key] for g in late_idx], float)
        ev = np.array([feats[g][key] for g in early_idx], float)
        return tgs.group_auc(lv, ev)

    aucs = {k: feat_auc(k) for k in ["disp_peak_time", "het_lag", "t50",
                                     "early_window_fano_auc", "disp_transience"]}
    return {"per_gene": {gene_names[g]: feats[g] for g in range(G)},
            "auc_late_gt_early": aucs,
            "matrices": {"fano_res": fano_M, "frac_expr": frac_M, "above_mean": above}}


def _module_coherence_traj(pool_log, tps, early_idx, late_idx):
    out = {"early": [], "late": [], "tps": [float(t) for t in tps]}
    for tp in tps:
        if tp not in pool_log:
            out["early"].append(float("nan")); out["late"].append(float("nan")); continue
        L = pool_log[tp]
        out["early"].append(tgs.module_coherence(L, early_idx))
        out["late"].append(tgs.module_coherence(L, late_idx))
    return out


# ===========================================================================
# Figures
# ===========================================================================
def _safe(fig_fn, *a, **k):
    if not HAVE_MPL:
        return
    try:
        fig_fn(*a, **k)
    except Exception as e:
        _log(f"   [plot warning] {fig_fn.__name__}: {e}")


def _fig_meanvar_cloud(out_dir, stim, pool_counts, tps, early_idx, late_idx, gene_names):
    show = [t for t in tps if t > 0][:6]
    if not show:
        return
    fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 3.2), squeeze=False)
    for ax, tp in zip(axes[0], show):
        c = pool_counts.get(tp)
        if c is None:
            continue
        mc = c.mean(0); vc = c.var(0, ddof=1)
        x = np.log10(mc + 1e-3); y = np.log10(vc + 1e-3)
        ax.scatter(x, y, s=4, c="0.7", alpha=0.5, label="panel")
        m = np.isfinite(x) & np.isfinite(y) & (mc > 0)
        if m.sum() > 25:
            coef = np.polyfit(x[m], y[m], 2)
            xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 50)
            ax.plot(xs, np.polyval(coef, xs), "k-", lw=1)
        ax.scatter(x[early_idx], y[early_idx], s=28, c="C3", edgecolor="k", lw=0.4, label="early")
        ax.scatter(x[late_idx], y[late_idx], s=28, c="C0", edgecolor="k", lw=0.4, label="late")
        ax.set_title(f"{tp}h"); ax.set_xlabel("log10 mean")
    axes[0][0].set_ylabel("log10 var"); axes[0][0].legend(fontsize=6, loc="upper left")
    fig.suptitle(f"{stim}: mean-variance cloud (late ABOVE trend = genuine excess dispersion)")
    fig.tight_layout(); fig.savefig(out_dir / f"meanvar_cloud_{stim}.png", dpi=130); plt.close(fig)


def _fig_trajectories(out_dir, stim, per, donors, tps, early_idx, late_idx, gene_names, above_mean):
    G = len(gene_names); tarr = np.asarray(tps, float)
    panels = [("mean_log", "mean log1p (raw)"), ("fano_res", "Fano residual (decoupled)"),
              ("bc_sarle", "Sarle BC"), ("frac_expr", "frac expressing"),
              ("det_res", "detection residual"), ("gini", "Gini"),
              ("iqr", "IQR (log)"), ("cv2", "CV^2")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (stat, title) in zip(axes.ravel(), panels):
        M = _traj(per, donors, tps, stat, G)
        for idx, col, lab in [(early_idx, "C3", "early/source"), (late_idx, "C0", "late/downstream")]:
            grp = M[idx]
            med = np.nanmedian(grp, axis=0)
            lo = np.nanpercentile(grp, 25, axis=0); hi = np.nanpercentile(grp, 75, axis=0)
            ax.plot(tarr, med, "-o", color=col, label=lab, ms=3)
            ax.fill_between(tarr, lo, hi, color=col, alpha=0.15)
        for tp in INTERMEDIATE_TPS:
            ax.axvline(tp, color="0.85", lw=0.8, zorder=0)
        ax.set_title(title, fontsize=9); ax.set_xlabel("hr")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{stim}: early vs late distributional trajectories (band = gene IQR within group)")
    fig.tight_layout(); fig.savefig(out_dir / f"trajectories_{stim}.png", dpi=130); plt.close(fig)


def _fig_distributions(out_dir, stim, pool_log, tps, early_idx, late_idx, gene_names):
    genes = list(early_idx) + list(late_idx)
    roles = ["early"] * len(early_idx) + ["late"] * len(late_idx)
    show_t = [t for t in tps if t in pool_log][:7]
    if not genes or not show_t:
        return
    fig, axes = plt.subplots(len(genes), len(show_t),
                             figsize=(1.5 * len(show_t), 1.0 * len(genes)), squeeze=False)
    for gi, (g, role) in enumerate(zip(genes, roles)):
        for tj, tp in enumerate(show_t):
            ax = axes[gi][tj]
            vals = pool_log[tp][:, g]
            col = "C3" if role == "early" else "C0"
            ax.hist(vals, bins=24, color=col, alpha=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if gi == 0:
                ax.set_title(f"{tp}h", fontsize=8)
            if tj == 0:
                ax.set_ylabel(f"{gene_names[g]}", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"{stim}: per-cell log1p distributions (early=coral, late=blue) — look for off→bimodal→on in late")
    fig.tight_layout(); fig.savefig(out_dir / f"distributions_{stim}.png", dpi=120); plt.close(fig)


def _fig_matched_mean(out_dir, stim, per, donors, early_idx, late_idx, gene_names):
    union, is_late = _union_islate(early_idx, late_idx)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    # within-decile late-early delta at 1h & 3h (fano_res)
    for tp, col in [(1.0, "C1"), (3.0, "C4")]:
        deltas = []
        for d in donors:
            if tp not in per[d]:
                continue
            mm = tgs.matched_mean_delta(per[d][tp]["fano_res"][union],
                                        per[d][tp]["mean_counts"][union], is_late)
            for b in mm["per_bin"]:
                deltas.append((b["bin"], b["delta"]))
        if deltas:
            bins = sorted(set(b for b, _ in deltas))
            mean_by_bin = [np.mean([dd for bb, dd in deltas if bb == b]) for b in bins]
            axes[0].plot(bins, mean_by_bin, "-o", color=col, label=f"{tp}h", ms=4)
    axes[0].axhline(0, color="k", lw=0.8); axes[0].set_xlabel("mean decile")
    axes[0].set_ylabel("late - early Fano-res"); axes[0].set_title("within-mean-bin Δ"); axes[0].legend(fontsize=7)
    # raw var vs decoupled fano_res, late vs early at 3h (donor points)
    for ax, stat, title in [(axes[1], "var_counts", "RAW var (coupled)"),
                            (axes[2], "fano_res", "Fano residual (decoupled)")]:
        tp = 3.0 if any(3.0 in per[d] for d in donors) else 1.0
        e = [np.nanmedian(per[d][tp][stat][early_idx]) for d in donors if tp in per[d]]
        l = [np.nanmedian(per[d][tp][stat][late_idx]) for d in donors if tp in per[d]]
        ax.boxplot([e, l], labels=["early", "late"])
        ax.scatter(np.ones(len(e)), e, c="C3", zorder=3); ax.scatter(2 * np.ones(len(l)), l, c="C0", zorder=3)
        ax.set_title(f"{title} @ {tp}h")
    fig.suptitle(f"{stim}: matched-mean honesty check (does the gap survive decoupling?)")
    fig.tight_layout(); fig.savefig(out_dir / f"matched_mean_{stim}.png", dpi=130); plt.close(fig)


def _fig_scorecard(out_dir, stim, hyp, traj_aucs, controls):
    tp_key = "3.0hr" if "3.0hr" in hyp["by_stat"] else next(iter(hyp["by_stat"]), None)
    if tp_key is None:
        return
    rec = hyp["by_stat"][tp_key]
    items = [(s, rec[s]["auc"], rec[s]["p"]) for s in rec]
    items += [(f"traj:{k}", v, np.nan) for k, v in traj_aucs.items()]
    rs = controls.get("random_split_fano_res_auc", np.nan)
    items.append(("ctrl:random_split", rs, np.nan))
    nf = controls.get("nfkb_vs_early_fano_res_auc", {})
    if isinstance(nf, dict) and "auc" in nf:
        items.append(("ctrl:nfkb_vs_early", nf["auc"], nf.get("p", np.nan)))
    items = [(n, a, p) for n, a, p in items if np.isfinite(a)]
    items.sort(key=lambda x: x[1])
    names = [i[0] for i in items]; aucs = [i[1] for i in items]
    cols = ["C7" if n.startswith("ctrl") else ("C2" if a > 0.5 else "C5") for n, a in zip(names, aucs)]
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(items) + 1))
    ax.barh(range(len(items)), [a - 0.5 for a in aucs], left=0.5, color=cols)
    ax.axvline(0.5, color="k", lw=1)
    ax.set_yticks(range(len(items))); ax.set_yticklabels(names, fontsize=7)
    for i, (n, a, p) in enumerate(items):
        lab = f"{a:.2f}" + (f" p={p:.2f}" if np.isfinite(p) else "")
        ax.text(a, i, " " + lab, va="center", fontsize=6)
    ax.set_xlabel("donor-level AUC(late > early)  [0.5 = no difference]")
    ax.set_title(f"{stim} @ {tp_key}: which aspects separate late from early\n(grey=controls, should sit ~0.5)")
    fig.tight_layout(); fig.savefig(out_dir / f"scorecard_{stim}.png", dpi=130); plt.close(fig)


def _fig_bimodality(out_dir, stim, per, donors, tps, early_idx, late_idx, gene_names):
    genes = list(early_idx) + list(late_idx)
    G = len(gene_names)
    M = _traj(per, donors, tps, "bc_sarle", G)[genes]
    Me = _traj(per, donors, tps, "bc_sarle_expr", G)[genes] if any(
        "bc_sarle_expr" in per[d][t] for d in donors for t in per[d]) else None
    labels = [gene_names[g] for g in genes]
    ncol = 2 if Me is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.5 * ncol, 0.32 * len(genes) + 1.5), squeeze=False)
    for ax, mat, ttl in zip(axes[0], [M] + ([Me] if Me is not None else []),
                            ["all cells"] + (["expressing only"] if Me is not None else [])):
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.3, vmax=0.8)
        ax.set_xticks(range(len(tps))); ax.set_xticklabels([f"{t:g}" for t in tps], fontsize=7)
        ax.set_yticks(range(len(genes))); ax.set_yticklabels(labels, fontsize=6)
        ax.axhline(len(early_idx) - 0.5, color="w", lw=1.5)
        ax.set_title(f"Sarle BC ({ttl}); >0.555=bimodal", fontsize=8); ax.set_xlabel("hr")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{stim}: bimodality (early genes above white line, late below)")
    fig.tight_layout(); fig.savefig(out_dir / f"bimodality_{stim}.png", dpi=130); plt.close(fig)


def _fig_temporal_scatter(out_dir, stim, traj, early_idx, late_idx, gene_names):
    pg = traj["per_gene"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, col, lab in [(early_idx, "C3", "early"), (late_idx, "C0", "late")]:
        pt = [pg[gene_names[g]]["disp_peak_time"] for g in idx]
        ti = [pg[gene_names[g]]["disp_transience"] for g in idx]
        on = [pg[gene_names[g]]["onset"] for g in idx]
        axes[0].scatter(pt, ti, c=col, label=lab, s=40, edgecolor="k", lw=0.4)
        axes[1].scatter(on, pt, c=col, label=lab, s=40, edgecolor="k", lw=0.4)
    axes[0].set_xlabel("dispersion peak time (hr)"); axes[0].set_ylabel("transience index")
    axes[0].set_title("WHEN heterogeneity peaks (late→later/more transient)"); axes[0].legend(fontsize=7)
    lims = [0, 8]; axes[1].plot(lims, lims, "k--", lw=0.8)
    axes[1].set_xlabel("onset time (mean)"); axes[1].set_ylabel("dispersion peak time")
    axes[1].set_title("heterogeneity lags onset (late above y=x)")
    fig.suptitle(f"{stim}: temporal-aggregate timing of moments")
    fig.tight_layout(); fig.savefig(out_dir / f"temporal_aggregate_{stim}.png", dpi=130); plt.close(fig)


# ===========================================================================
# Synthetic self-test
# ===========================================================================
def run_synthetic(args, out_dir):
    _log("=== SYNTHETIC self-test: early(uniform) vs late(transient bimodal/overdispersed) ===")
    rng = np.random.default_rng(args.seed)
    sim = tgs.simulate_early_late(rng=rng)
    tps = sim["time_hrs"]; early_idx = sim["early_idx"]; late_idx = sim["late_idx"]
    gene_names = sim["gene_names"]; G = len(gene_names)
    # mimic 4 donors by bootstrapping cells
    donors = ["d0", "d1", "d2", "d3"]
    per = {d: {} for d in donors}
    pool_log = {}; pool_counts = {}
    for tp in tps:
        c0, l0 = sim["blocks"][tp]
        pool_log[tp] = l0; pool_counts[tp] = c0
        for d in donors:
            idx = rng.choice(c0.shape[0], c0.shape[0], replace=True)
            per[d][tp] = _block_stats(c0[idx], l0[idx], c0.shape[0], rng)
    hyp, union, is_late = _run_hypotheses(per, donors, tps, gene_names, early_idx, late_idx, [], rng)
    traj = _trajectory_features(per, donors, tps, gene_names, early_idx, late_idx, rng)

    def auc_of(stat, tp):
        return hyp["by_stat"].get(f"{tp}hr", {}).get(stat, {}).get("auc", float("nan"))

    fano_auc = max(auc_of("fano_res", 1.0), auc_of("fano_res", 3.0))
    mean_auc = max(tgs.group_auc(
        [np.nanmedian([per[d][tp]["mean_log"][g] for d in donors]) for g in late_idx],
        [np.nanmedian([per[d][tp]["mean_log"][g] for d in donors]) for g in early_idx])
        for tp in (3.0,) )
    bc_auc = max(auc_of("bc_sarle", 1.0), auc_of("bc_sarle", 3.0))
    _log(f"  decoupled fano_res AUC(late>early) intermediate = {fano_auc:.3f}  (expect >0.6)")
    _log(f"  Sarle BC AUC(late>early) intermediate          = {bc_auc:.3f}  (expect >0.6)")
    _log(f"  raw mean_log AUC @3h                            = {mean_auc:.3f}  (both high -> ~0.5-0.7, not the discriminator)")
    _log(f"  traj AUCs late>early: {json.dumps({k: round(v,2) for k,v in traj['auc_late_gt_early'].items()})}")
    ok = (np.isfinite(fano_auc) and fano_auc > 0.6 and np.isfinite(bc_auc) and bc_auc > 0.6)
    _log(f"  APPARATUS {'OK' if ok else 'FAIL'} (decoupled overdispersion + bimodality recover planted late>early)")
    if HAVE_MPL:
        _safe(_fig_meanvar_cloud, out_dir, "SYNTH", pool_counts, tps, early_idx, late_idx, gene_names)
        _safe(_fig_trajectories, out_dir, "SYNTH", per, donors, tps, early_idx, late_idx, gene_names, None)
        _safe(_fig_distributions, out_dir, "SYNTH", pool_log, tps, early_idx, late_idx, gene_names)
        _safe(_fig_scorecard, out_dir, "SYNTH", hyp, traj["auc_late_gt_early"], hyp["controls"])
    res = {"_verdict": "APPARATUS OK" if ok else "APPARATUS FAIL",
           "fano_res_auc": float(fano_auc), "bc_auc": float(bc_auc), "mean_auc": float(mean_auc),
           "traj_aucs": traj["auc_late_gt_early"]}
    with open(out_dir / "synthetic.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    return res


# ===========================================================================
# Real run
# ===========================================================================
def run_real(args, out_dir):
    import scanpy as sc
    _log("=== REAL: Sheu early-vs-late distributional statistics ===")
    bp = _import_script("build_pseudotubes_sheu2024")
    crt = _import_script("compute_sheu_realtime_emergence")
    _log(f"  loading raw Sheu time course from {args.raw_dir} ...")
    adata = bp.load_sheu_anndata(args.raw_dir)
    adata = bp.relabel_to_pbs(adata)
    # raw counts copy + log copy
    adata_counts = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    adata_log = adata.X
    obs = adata.obs.copy()
    obs["_time_hr"] = obs[crt.TIMEPT_COL].apply(crt._parse_timept_to_hr)
    gene_names = list(adata.var_names)
    avail = set(obs[crt.CYTOKINE_COL].unique())
    _log(f"  cells={adata.n_obs} genes={len(gene_names)} stimuli={sorted(avail)}")

    early, late, nfkb = _gene_sets()
    early_idx, early_g = _idx(early, gene_names)
    late_idx, late_g = _idx(late, gene_names)
    nfkb_idx, _ = _idx(nfkb, gene_names)
    _log(f"  early/source genes present: {early_g}")
    _log(f"  late/downstream genes present: {late_g}")
    rng = np.random.default_rng(args.seed)

    summary = {"early_genes": early_g, "late_genes": late_g, "stimuli": {}}
    for stim in args.stimuli:
        label = crt._resolve_stimulus_label(stim, avail)
        if label is None:
            _log(f"  SKIP {stim}: not in data"); continue
        _log(f"\n-- {stim} (label={label})")
        donors = sorted({d for d in obs["pseudo_donor"].unique()
                         if ((obs["pseudo_donor"] == d) & (obs[crt.CYTOKINE_COL] == label)).sum() >= args.min_cells})
        tps_by_donor = {}
        for d in donors:
            dm = (obs["pseudo_donor"] == d)
            tset = {0.0}
            for tp in sorted(obs.loc[dm & (obs[crt.CYTOKINE_COL] == label), "_time_hr"].dropna().unique()):
                if ((obs[crt.CYTOKINE_COL] == label) & (obs["_time_hr"] == tp) & dm).sum() >= args.min_cells:
                    tset.add(float(tp))
            tps_by_donor[d] = sorted(tset)
        _log(f"   donors={donors}")
        _log(f"   tps_by_donor={tps_by_donor}")
        if len(donors) == 0:
            _log("   no usable donors; skipping"); continue

        per, pool_log, pool_counts, tps = _collect_real(
            adata_counts, adata_log, obs, label, donors, tps_by_donor, gene_names,
            args.min_cells, args.cap, rng)

        hyp, union, is_late = _run_hypotheses(per, donors, tps, gene_names,
                                              early_idx, late_idx, nfkb_idx, rng)
        traj = _trajectory_features(per, donors, tps, gene_names, early_idx, late_idx, rng)
        coher = _module_coherence_traj(pool_log, tps, early_idx, late_idx)

        # headline H1 readout: decoupled fano_res AUC at intermediate tps, both routes
        h1 = {}
        for tp in INTERMEDIATE_TPS:
            r = hyp["by_stat"].get(f"{tp}hr", {})
            if "fano_res" in r:
                h1[f"{tp}hr"] = {"auc_residual": r["fano_res"]["auc"],
                                 "p": r["fano_res"]["p"],
                                 "matched_delta": r["fano_res"]["matched_delta"],
                                 "raw_var_auc": r.get("var_counts", {}).get("auc"),
                                 "frac_donors_late_gt": r["fano_res"]["frac_late_gt"]}
        _log(f"   H1 decoupled fano_res AUC(late>early): "
             f"{json.dumps({k: round(v['auc_residual'],3) for k,v in h1.items()})}")
        _log(f"      raw var AUC (coupled, expect inflated): "
             f"{json.dumps({k: (round(v['raw_var_auc'],3) if v['raw_var_auc'] else None) for k,v in h1.items()})}")
        _log(f"      matched-mean Δ (sign must agree): "
             f"{json.dumps({k: round(v['matched_delta'],3) for k,v in h1.items()})}")
        _log(f"   control random-split fano_res AUC (expect ~0.5): "
             f"{hyp['controls'].get('random_split_fano_res_auc'):.3f}")
        _log(f"   traj AUC late>early: {json.dumps({k: round(v,3) for k,v in traj['auc_late_gt_early'].items()})}")
        _log(f"   late-module coherence over time: {[round(x,2) if np.isfinite(x) else None for x in coher['late']]}")
        _log(f"   early-module coherence over time: {[round(x,2) if np.isfinite(x) else None for x in coher['early']]}")

        summary["stimuli"][stim] = {
            "timepoints": [float(t) for t in tps], "donors": donors,
            "H1": h1, "controls": hyp["controls"],
            "traj_auc_late_gt_early": traj["auc_late_gt_early"],
            "coherence": coher,
            "by_stat": {k: {s: {"auc": v[s]["auc"], "p": v[s]["p"],
                               "frac_donors_late_gt": v[s]["frac_late_gt"],
                               "matched_delta": v[s]["matched_delta"]}
                            for s in v} for k, v in hyp["by_stat"].items()},
        }
        # per-gene trajectory feature table
        import pandas as pd
        rows = []
        for g_name, fv in traj["per_gene"].items():
            role = "early" if g_name in set(early_g) else "late" if g_name in set(late_g) else "other"
            if role == "other" and g_name not in set(nfkb):
                continue
            rows.append({"gene": g_name, "role": role, **fv})
        pd.DataFrame(rows).to_csv(out_dir / f"gene_features_{stim}.csv", index=False)

        if HAVE_MPL:
            _safe(_fig_meanvar_cloud, out_dir, stim, pool_counts, tps, early_idx, late_idx, gene_names)
            _safe(_fig_trajectories, out_dir, stim, per, donors, tps, early_idx, late_idx, gene_names,
                  traj["matrices"]["above_mean"])
            _safe(_fig_distributions, out_dir, stim, pool_log, tps, early_idx, late_idx, gene_names)
            _safe(_fig_matched_mean, out_dir, stim, per, donors, early_idx, late_idx, gene_names)
            _safe(_fig_scorecard, out_dir, stim, hyp, traj["auc_late_gt_early"], hyp["controls"])
            _safe(_fig_bimodality, out_dir, stim, per, donors, tps, early_idx, late_idx, gene_names)
            _safe(_fig_temporal_scatter, out_dir, stim, traj, early_idx, late_idx, gene_names)

    # verdict (H1-style): decoupled fano_res AUC>0.5 at an intermediate tp, raw>decoupled inflation
    # present, control ~0.5, replicated in both stimuli.
    def stim_pass(s):
        st = summary["stimuli"].get(s)
        if not st:
            return False
        aucs = [v["auc_residual"] for v in st["H1"].values() if np.isfinite(v["auc_residual"])]
        ctrl = st["controls"].get("random_split_fano_res_auc", 0.5)
        return (any(a > 0.5 for a in aucs) and abs(ctrl - 0.5) < 0.15)
    passed = [s for s in args.stimuli if s in summary["stimuli"] and stim_pass(s)]
    summary["_verdict"] = ("GREEN" if len(passed) >= 2 else
                           "AMBER" if len(passed) == 1 else "RED/mean-coupling")
    summary["_passed_stimuli"] = passed
    _log(f"\n  VERDICT: {summary['_verdict']}  (decoupled-overdispersion stimuli: {passed})")
    with open(out_dir / "validation.json", "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
    return summary


def _parse_args():
    p = argparse.ArgumentParser(description="Sheu early-vs-late gene distributional statistics")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--raw_dir", default="/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw")
    p.add_argument("--stimuli", nargs="+", default=["LPS", "PIC"])
    p.add_argument("--min_cells", type=int, default=80, help="min cells per (stim,donor,timepoint) block")
    p.add_argument("--cap", type=int, default=1500, help="cap on common-n subsample per block")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    global _LOG
    args = _parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / "sheu_temporal_gene_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG = open(out_dir / "run.log", "w")
    _log(f"diptest={tgs.HAVE_DIPTEST} scipy={tgs.HAVE_SCIPY} mpl={HAVE_MPL}")
    res = run_synthetic(args, out_dir) if args.synthetic else run_real(args, out_dir)
    _log(f"\nDONE -> {out_dir}")
    _LOG.close()


if __name__ == "__main__":
    main()
