"""
Diagnose the source-potency RED verdict (reports/source_potency/SOURCE_POTENCY_RESULTS.md):
is `source_potency` (learning-curve lateness, computed on TRAIN records) actually just
measuring per-cytokine overfitting, rather than a genuine primary/secondary-signal cascade
signature?

Motivation: run_summary.json for all 3 seeds already shows a severe aggregate train/val gap
(train_final ~0.36-0.48, val_final ~0.05-0.06 on 91-class chance ~0.011) -- this script checks
whether that gap is uneven across cytokines, and whether the unevenness explains the ranking
`compute_source_potency.py` produced (which reads only `records`, i.e. TRAIN donors).

Reuses cytokine_mil.analysis.source_potency (does not reimplement the score) and
cytokine_mil.analysis.dynamics.aggregate_to_donor_level. Adds:
  - per-cytokine VAL-side metrics (P_max, final, peak epoch) alongside the existing TRAIN-side
    source_potency table
  - generalization_gap = train_P_max - val_P_max per cytokine
  - Spearman(source_potency, generalization_gap) -- the key diagnostic
  - count of "underfit-on-train" cytokines (train_P_max below a floor)
  - the aggregate (mean-over-cytokines) train vs val curve across the 25 logged epochs

Usage:
    python scripts/diagnose_source_potency_overfit.py \
        --base_dir results/attention_dynamics --seeds 42 123 7 \
        --out reports/source_potency/OVERFIT_DIAGNOSIS.md
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from cytokine_mil.analysis.dynamics import aggregate_to_donor_level
from cytokine_mil.analysis.source_potency import (
    ceiling, normalized_trajectory_auc, per_cytokine_metrics, source_potency_table,
    DEEP_POOL as _DEEP, SHALLOW_POOL as _SHALLOW,
)
from cytokine_mil.analysis.attention_dynamics import spearman  # numpy-only
from cytokine_mil.analysis.confusion_dynamics import compute_confusion_trajectory

AXES_CSV = REPO / "reports/cascade_pairs/cytokine_axes.csv"


def _dyn_paths(args):
    return [Path(args.base_dir) / f"seed_{s}" / "dynamics.pkl" for s in args.seeds]


class _LabelShim:
    """Minimal .cytokines/.encode() shim from the saved index-ordered cytokine list --
    avoids needing the original CytokineLabel/manifest to rebuild the encoder."""
    def __init__(self, cytokines_list):
        self.cytokines = cytokines_list
        self._to_idx = {c: i for i, c in enumerate(cytokines_list)}

    def encode(self, name):
        return self._to_idx[name]


def _stem(name: str) -> str:
    """Naive family-naming heuristic: strip a trailing greek-word-number or digit-letter
    variant suffix so e.g. IL-17A/IL-17F -> 'IL-17', IFN-alpha1/IFN-alpha2 -> 'IFN-alpha',
    IFN-lambda1/IFN-lambda3 -> 'IFN-lambda'. Deliberately does NOT strip a bare trailing
    number (IL-4, IL-13 must stay distinct -- the number IS the interleukin's identity,
    not a variant suffix). NOT a curated immunology family list -- just a transparent
    string heuristic to flag candidates for the reader to judge."""
    import re
    s = name
    s = re.sub(r"(alpha|beta|gamma|delta|epsilon|omega|lambda)\d+$", r"\1", s)
    s = re.sub(r"(?<=[0-9])[A-F]$", "", s)  # IL-17A -> IL-17
    return s


def _confusion_diagnostics(records, cytokines_list, exclude):
    """Per-cytokine top confused-with class at the FINAL logged epoch (train records --
    reflects what the optimizer actually spent gradient effort resolving)."""
    enc = _LabelShim(cytokines_list)
    conf, names = compute_confusion_trajectory(records, enc)  # (K, K, T)
    final = conf[:, :, -1]  # (K, K) at last logged epoch
    out = {}
    for a_idx, name in enumerate(names):
        if name in exclude:
            continue
        row = final[a_idx].copy()
        row[a_idx] = -1.0  # exclude self
        order = np.argsort(row)[::-1]
        top1_idx, top2_idx = int(order[0]), int(order[1])
        out[name] = {
            "top1_confusor": names[top1_idx], "top1_mass": float(row[top1_idx]),
            "top2_confusor": names[top2_idx], "top2_mass": float(row[top2_idx]),
            "top1_is_pbs": names[top1_idx] == "PBS",
            "top1_same_stem": _stem(name) == _stem(names[top1_idx]),
        }
    return out


def _donor_mean_traj(records, exclude):
    """{cytokine -> mean-across-donors trajectory}, matching source_potency's convention
    (median across a donor's tubes, then mean across donors)."""
    donor_trajs = aggregate_to_donor_level(records, "p_correct_trajectory")
    out = {}
    for cyt, by_donor in donor_trajs.items():
        if cyt in exclude:
            continue
        arrs = [np.asarray(v, dtype=np.float64) for v in by_donor.values()]
        n = min(a.size for a in arrs)
        out[cyt] = np.mean(np.stack([a[:n] for a in arrs]), axis=0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="results/attention_dynamics")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--exclude", nargs="+", default=["PBS"])
    p.add_argument("--ceiling_floor", type=float, default=0.1)
    p.add_argument("--underfit_floor", type=float, default=0.2,
                    help="train P_max below this = 'never learned even on train'")
    p.add_argument("--out", default="reports/source_potency/OVERFIT_DIAGNOSIS.md")
    args = p.parse_args()
    exclude = set(args.exclude)

    seed_train_metrics = []      # per-seed source_potency-style train metrics
    seed_val_traj = []           # per-seed {cyt -> val trajectory}
    seed_train_traj = []         # per-seed {cyt -> train trajectory}
    seed_confusion = []          # per-seed {cyt -> confusion diagnostic dict}
    epochs_ref = None

    for dp in _dyn_paths(args):
        if not dp.exists():
            print(f"skip (missing): {dp}"); continue
        with open(dp, "rb") as f:
            d = pickle.load(f)
        recs, val_recs, epochs = d.get("records") or [], d.get("val_records") or [], d.get("logged_epochs") or []
        if not recs or not epochs:
            print(f"skip (no records/epochs): {dp}"); continue
        epochs_ref = epochs
        seed_train_metrics.append(per_cytokine_metrics(recs, epochs, exclude=args.exclude))
        seed_train_traj.append(_donor_mean_traj(recs, exclude))
        seed_val_traj.append(_donor_mean_traj(val_recs, exclude) if val_recs else {})
        cyt_list = d.get("label_encoder_cytokines")
        if cyt_list and recs[0].get("softmax_trajectory") is not None:
            seed_confusion.append(_confusion_diagnostics(recs, cyt_list, exclude))
        else:
            seed_confusion.append({})
        print(f"loaded {dp}: {len(recs)} train / {len(val_recs)} val records, {len(epochs)} epochs")

    if not seed_train_metrics:
        sys.exit("No usable dynamics.pkl found.")

    # ---- source_potency table (exact reuse of the published computation) ----
    keys = ("P_max", "normalized_auc", "plateau_epoch", "late_gain")
    all_cyts = sorted({c for m in seed_train_metrics for c in m})
    avg_train = {}
    for c in all_cyts:
        vals = {k: [m[c][k] for m in seed_train_metrics if c in m] for k in keys}
        avg_train[c] = {k: float(np.nanmean(vals[k])) if vals[k] else float("nan") for k in keys}
    table = source_potency_table(avg_train, ceiling_floor=args.ceiling_floor)

    # ---- NEW: val-side metrics, seed-averaged ----
    val_metrics = {}
    for c in all_cyts:
        vps = [ceiling(sv[c]) for sv in seed_val_traj if c in sv]
        vfs = [float(sv[c][-1]) for sv in seed_val_traj if c in sv]
        vpeak_ep = [int(np.argmax(sv[c])) for sv in seed_val_traj if c in sv]
        val_metrics[c] = {
            "val_P_max": float(np.mean(vps)) if vps else float("nan"),
            "val_final": float(np.mean(vfs)) if vfs else float("nan"),
            "val_peak_epoch_idx": float(np.mean(vpeak_ep)) if vpeak_ep else float("nan"),
            "n_seeds_val": len(vps),
        }

    # ---- NEW: confusion diagnostics, seed-averaged (Q4) ----
    conf_metrics = {}
    for c in all_cyts:
        entries = [sc[c] for sc in seed_confusion if c in sc]
        if not entries:
            continue
        conf_metrics[c] = {
            "top1_is_pbs_frac": float(np.mean([e["top1_is_pbs"] for e in entries])),
            "top1_same_stem_frac": float(np.mean([e["top1_same_stem"] for e in entries])),
            "top1_confusor_seed0": entries[0]["top1_confusor"],
            "top1_mass_seed0": entries[0]["top1_mass"],
        }

    # ---- generalization gap + join ----
    rows = []
    for c in all_cyts:
        r = dict(table[c]); r.update(val_metrics.get(c, {})); r.update(conf_metrics.get(c, {}))
        r["cytokine"] = c
        r["gen_gap"] = r["P_max"] - r.get("val_P_max", float("nan"))
        rows.append(r)

    # ---- key diagnostic: does source_potency just track overfitting? ----
    included = [r for r in rows if r["included"] and np.isfinite(r["gen_gap"])]
    rho_gap, n_gap = spearman(
        [r["source_potency"] for r in included], [r["gen_gap"] for r in included])
    rho_valpmax, n_valpmax = spearman(
        [r["source_potency"] for r in included], [r["val_P_max"] for r in included])

    # ---- Q4: does source_potency track family-confusion vs PBS-confusion? ----
    with_conf = [r for r in included if "top1_is_pbs_frac" in r]
    rho_pbs, n_pbs = spearman(
        [r["source_potency"] for r in with_conf], [r["top1_is_pbs_frac"] for r in with_conf])
    rho_stem, n_stem = spearman(
        [r["source_potency"] for r in with_conf], [r["top1_same_stem_frac"] for r in with_conf])
    n_top1_pbs = sum(1 for r in with_conf if r["top1_is_pbs_frac"] >= 0.5)
    n_top1_other = len(with_conf) - n_top1_pbs
    n_top1_same_stem = sum(1 for r in with_conf if r["top1_same_stem_frac"] >= 0.5)

    # ---- underfit-on-train candidates ----
    underfit = sorted([r["cytokine"] for r in rows if r["P_max"] < args.underfit_floor])

    # ---- aggregate train vs val curve across logged epochs ----
    n_ep = len(epochs_ref)
    agg_train = np.zeros(n_ep); agg_val = np.zeros(n_ep); n_c = 0
    for c in all_cyts:
        tt = [st[c] for st in seed_train_traj if c in st]
        vv = [sv[c] for sv in seed_val_traj if c in sv]
        if not tt or not vv:
            continue
        m = min(min(a.size for a in tt), n_ep)
        agg_train[:m] += np.mean(np.stack([a[:m] for a in tt]), axis=0)
        agg_val[:m] += np.mean(np.stack([a[:m] for a in vv]), axis=0)
        n_c += 1
    agg_train /= n_c; agg_val /= n_c
    val_peak_idx = int(np.argmax(agg_val))

    # ---- outputs ----
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_ref, agg_train, "-o", color="tab:blue", label="train (mean over cytokines)", ms=3)
    ax.plot(epochs_ref, agg_val, "-o", color="tab:red", label="val D2/D3 (mean over cytokines)", ms=3)
    ax.axvline(epochs_ref[val_peak_idx], color="gray", ls="--", lw=1,
               label=f"val peak @ epoch {epochs_ref[val_peak_idx]}")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean p_correct across cytokines")
    ax.set_title("Aggregate train vs val accuracy over training (3-seed mean)")
    ax.legend(); fig.tight_layout()
    fig.savefig(outp.parent / "overfit_aggregate_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    xs = [r["gen_gap"] for r in included]; ys = [r["source_potency"] for r in included]
    ax.scatter(xs, ys, s=30, alpha=0.7)
    for r in included:
        if abs(r["source_potency"]) > 2 or r["gen_gap"] > 0.5:
            ax.annotate(r["cytokine"], (r["gen_gap"], r["source_potency"]), fontsize=6)
    ax.set_xlabel("generalization gap  (train P_max - val P_max)")
    ax.set_ylabel("source_potency")
    ax.set_title(f"source_potency vs overfitting gap  (Spearman rho={rho_gap:.3f}, n={n_gap})")
    fig.tight_layout()
    fig.savefig(outp.parent / "overfit_potency_vs_gap.png", dpi=150)
    plt.close(fig)

    L = []
    L.append("# Overfit diagnosis for the source-potency RED verdict\n\n")
    L.append(f"Seeds: {len(seed_train_metrics)} · cytokines: {len(all_cyts)} "
             f"(included above ceiling_floor={args.ceiling_floor}: {len(included)})\n\n")
    L.append("## Q1 -- Dataset\n")
    L.append("Oesinghaus 24h PBMC, 91-class multiclass (90 cytokines + PBS), Stage-2 AB-MIL "
             "with a FROZEN Stage-1 encoder. Confirmed from `results/attention_dynamics/"
             "seed_42/train.log`: 10920 tubes total (9100 train / 1820 val), val donors = "
             "Donor2/Donor3. **Stage 2 here is the short 'Stage 2a warmup' from the "
             "confusion-dynamics (§33) plan -- 250 epochs at lr=0.001, explicitly a small "
             "LR chosen 'so the attention/FCN settle without disturbing the encoder' / "
             "'smooth centroid trajectories', NOT a from-scratch convergence run.** "
             "`source_potency` reuses this dynamics.pkl (`records`, i.e. TRAIN donors only) "
             "without re-training -- it was never trained/tuned for this specific question.\n\n")
    L.append("## Q2 -- Overfitting\n")
    L.append(f"**Severe, seed-consistent.** run_summary.json, all 3 seeds:\n\n")
    L.append("| seed | train_final | val_final |\n|---|---:|---:|\n")
    L.append("| 42 | 0.4765 | 0.0491 |\n| 123 | 0.4545 | 0.0517 |\n| 7 | 0.3585 | 0.0565 |\n\n")
    L.append(f"Val final accuracy (5-6%) is barely above 91-class chance (1.1%) and far below "
             f"train (36-48%). Aggregate (mean-over-cytokines) curve: val peaks at epoch "
             f"**{epochs_ref[val_peak_idx]}** of 250 then "
             f"{'declines' if agg_val[val_peak_idx] > agg_val[-1] else 'plateaus'} "
             f"(val_peak={agg_val[val_peak_idx]:.4f}, val_final={agg_val[-1]:.4f}); train keeps "
             f"climbing throughout (train_final={agg_train[-1]:.4f}). "
             "See `overfit_aggregate_curve.png`.\n\n")
    L.append("## Q3 -- Underfitting (per class, on TRAIN)\n")
    L.append(f"{len(underfit)} of {len(all_cyts)} cytokines never exceed train P_max="
             f"{args.underfit_floor} (never learned even on train donors): "
             f"{', '.join(underfit) if underfit else '(none)'}\n\n")
    L.append("## Q2/Q3 tie-in -- does source_potency just measure overfitting?\n")
    L.append(f"**Spearman(source_potency, generalization_gap) = {rho_gap:.3f}** (n={n_gap}). "
             f"Spearman(source_potency, val_P_max) = {rho_valpmax:.3f} (n={n_valpmax}).\n\n")
    L.append("(interpretation added by hand after inspecting the numbers above)\n\n")
    L.append("## Q4 -- family/similar-class confusion vs PBS confusion\n")
    L.append("Per cytokine, at the FINAL logged epoch of TRAIN records, which OTHER class gets "
             "the most softmax mass (`compute_confusion_trajectory`, reused unchanged from "
             "`analysis/confusion_dynamics.py`). `top1_same_stem` is a naive string heuristic "
             "(strip trailing greek-letter/number/single-letter suffixes, e.g. IL-17A/IL-17F -> "
             "'IL-17') -- NOT a curated immunology family list; a flag to inspect, not a claim.\n\n")
    L.append(f"Of {len(with_conf)} scored cytokines: **{n_top1_pbs}** have PBS as their #1 "
             f"confusor (majority vote across seeds), **{n_top1_other}** have another cytokine. "
             f"Of those {n_top1_other}, **{n_top1_same_stem}** share a name-stem with their #1 "
             f"confusor.\n\n")
    L.append(f"**Spearman(source_potency, top1_is_pbs_frac) = {rho_pbs:.3f}** (n={n_pbs}). "
             f"**Spearman(source_potency, top1_same_stem_frac) = {rho_stem:.3f}** (n={n_stem}).\n\n")
    L.append("(interpretation added by hand after inspecting the numbers above)\n\n")
    L.append("## Full per-cytokine table\n")
    L.append("| cytokine | source_potency | train_Pmax | val_Pmax | gen_gap | val_final | "
             "top1_confusor | top1_mass | same_stem | pool |\n"
             "|---|---:|---:|---:|---:|---:|---|---:|---:|---|\n")
    for r in sorted(rows, key=lambda r: -r["gen_gap"] if np.isfinite(r["gen_gap"]) else 999):
        pool = "DEEP" if r["cytokine"] in _DEEP else ("SHALLOW" if r["cytokine"] in _SHALLOW else "")
        L.append(f"| {r['cytokine']} | {r['source_potency']:+.2f} | {r['P_max']:.3f} | "
                 f"{r.get('val_P_max', float('nan')):.3f} | {r['gen_gap']:.3f} | "
                 f"{r.get('val_final', float('nan')):.3f} | "
                 f"{r.get('top1_confusor_seed0', '')} | {r.get('top1_mass_seed0', float('nan')):.3f} | "
                 f"{r.get('top1_same_stem_frac', float('nan')):.2f} | {pool} |\n")
    outp.write_text("".join(L))
    print(f"\nSaved: {outp}")
    print(f"rho(potency, gen_gap)={rho_gap:.3f}  rho(potency, val_Pmax)={rho_valpmax:.3f}")
    print(f"rho(potency, top1_is_pbs_frac)={rho_pbs:.3f}  rho(potency, top1_same_stem_frac)={rho_stem:.3f}")
    print(f"val peaks at epoch {epochs_ref[val_peak_idx]}/250, val_final={agg_val[-1]:.4f}")
    print(f"underfit-on-train (P_max<{args.underfit_floor}): {len(underfit)} cytokines")
    print(f"top1-confusor: {n_top1_pbs} PBS / {n_top1_other} other-cytokine "
          f"({n_top1_same_stem} of those same-stem)")


if __name__ == "__main__":
    main()
