"""
Recompute source_potency on the PRE-OVERFITTING window only, and re-run the exact same
validation as compute_source_potency.py (P1: Spearman vs audited directed out-degree; P3:
DEEP vs SHALLOW permutation test).

Motivation (reports/source_potency/OVERFIT_DIAGNOSIS.md): the published source_potency
(SOURCE_POTENCY_RESULTS.md, RED: P1 rho=-0.067) is computed on the FULL 250-epoch TRAIN
trajectory, but val (D2/D3) peaks at epoch ~60 then flatlines/declines while train keeps
climbing to epoch 250 -- so late-training "shape" on the full window is dominated by
memorization of train-donor-specific noise, not a generalizable cascade signature. This
script truncates every trajectory to the epoch where the AGGREGATE val curve peaks (computed
from the same dynamics.pkl, not hardcoded) and reruns the identical P1/P3 tests on that
truncated window -- the direct test of whether removing the overfitting-dominated tail
recovers the predicted P1 relationship.

Does not modify cytokine_mil/analysis/source_potency.py; only truncates each record's
p_correct_trajectory before calling the existing per_cytokine_metrics/source_potency_table/
validate_against_degree/validate_deep_vs_shallow.

Usage:
    python scripts/recompute_source_potency_early_window.py \
        --base_dir results/attention_dynamics --seeds 42 123 7
"""

import argparse
import csv
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
    DEEP_POOL, MASTER_REGULATORS, SHALLOW_POOL, graph_coupling_degree, graph_out_degree,
    per_cytokine_metrics, source_potency_table, validate_against_degree,
    validate_deep_vs_shallow,
)

AXES_CSV = REPO / "reports/cascade_pairs/cytokine_axes.csv"
AUDITED_CSV = REPO / "reports/cascade_pairs/cytokine_axes_audited.csv"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="results/attention_dynamics")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--axes_csv", default=str(AXES_CSV))
    p.add_argument("--audited_csv", default=str(AUDITED_CSV))
    p.add_argument("--ceiling_floor", type=float, default=0.1)
    p.add_argument("--plateau_frac", type=float, default=0.9)
    p.add_argument("--exclude", nargs="+", default=["PBS"])
    p.add_argument("--truncate_epoch", type=int, default=None,
                    help="override: truncate window at this epoch instead of the "
                         "data-driven aggregate-val-peak epoch")
    p.add_argument("--out", default=str(REPO / "reports/source_potency/EARLY_WINDOW_RESULTS.md"))
    return p.parse_args()


def _dyn_paths(args):
    return [Path(args.base_dir) / f"seed_{s}" / "dynamics.pkl" for s in args.seeds]


def _read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _donor_mean_traj(records, exclude):
    donor_trajs = aggregate_to_donor_level(records, "p_correct_trajectory")
    out = {}
    for cyt, by_donor in donor_trajs.items():
        if cyt in exclude:
            continue
        arrs = [np.asarray(v, dtype=np.float64) for v in by_donor.values()]
        n = min(a.size for a in arrs)
        out[cyt] = np.mean(np.stack([a[:n] for a in arrs]), axis=0)
    return out


def _find_val_peak_epoch(dyn_dicts, exclude):
    """Data-driven truncation point: epoch where the aggregate (mean-over-cytokines,
    mean-over-seeds) val trajectory peaks. Matches OVERFIT_DIAGNOSIS.md's methodology."""
    epochs_ref = None
    per_seed_val = []
    for d in dyn_dicts:
        val_recs, epochs = d.get("val_records") or [], d.get("logged_epochs") or []
        if not val_recs:
            continue
        epochs_ref = epochs
        per_seed_val.append(_donor_mean_traj(val_recs, exclude))
    if not per_seed_val or epochs_ref is None:
        return None, None
    n_ep = len(epochs_ref)
    all_cyts = sorted({c for sv in per_seed_val for c in sv})
    agg = np.zeros(n_ep)
    n_c = 0
    for c in all_cyts:
        vv = [sv[c] for sv in per_seed_val if c in sv]
        if not vv:
            continue
        m = min(min(a.size for a in vv), n_ep)
        agg[:m] += np.mean(np.stack([a[:m] for a in vv]), axis=0)
        n_c += 1
    agg /= max(n_c, 1)
    peak_idx = int(np.argmax(agg))
    return int(epochs_ref[peak_idx]), epochs_ref


def _truncate_records(records, epochs, max_epoch):
    keep_n = sum(1 for e in epochs if e <= max_epoch)
    keep_n = max(keep_n, 2)
    trunc_epochs = list(epochs)[:keep_n]
    out = []
    for r in records:
        r2 = dict(r)
        traj = r.get("p_correct_trajectory")
        if traj is not None:
            r2["p_correct_trajectory"] = np.asarray(traj, dtype=np.float64)[:keep_n]
        out.append(r2)
    return out, trunc_epochs


def _run_pipeline(seed_dicts, exclude, epochs_key_records, max_epoch, ceiling_floor,
                   plateau_frac, axes_rows, audited_rows, label):
    """Full source_potency compute + P1/P2/P3 validation for one truncation setting."""
    seed_metrics = []
    for d in seed_dicts:
        recs, epochs = d.get(epochs_key_records) or [], d.get("logged_epochs") or []
        if not recs or not epochs:
            continue
        if max_epoch is not None:
            recs, epochs = _truncate_records(recs, epochs, max_epoch)
        seed_metrics.append(per_cytokine_metrics(recs, epochs, exclude=exclude,
                                                   plateau_frac=plateau_frac))
    if not seed_metrics:
        return None

    keys = ("P_max", "normalized_auc", "plateau_epoch", "late_gain")
    all_cyts = sorted({c for m in seed_metrics for c in m})
    avg = {}
    for c in all_cyts:
        vals = {k: [m[c][k] for m in seed_metrics if c in m] for k in keys}
        avg[c] = {k: float(np.nanmean(vals[k])) if vals[k] else float("nan") for k in keys}
    table = source_potency_table(avg, ceiling_floor=ceiling_floor)

    coupling_deg = graph_coupling_degree(axes_rows)
    out_deg = graph_out_degree(audited_rows)
    potency = {c: table[c]["source_potency"] for c in table}

    v_out = validate_against_degree(potency, out_deg)
    v_coup = validate_against_degree(potency, coupling_deg)
    v_ds = validate_deep_vs_shallow(potency)

    ranked = sorted([c for c in table if table[c]["included"]],
                     key=lambda c: table[c]["source_potency"], reverse=True)
    mr_ranks = {mr: (ranked.index(mr) + 1 if mr in ranked else None) for mr in MASTER_REGULATORS}

    return {
        "label": label, "table": table, "ranked": ranked, "n_scored": len(ranked),
        "n_total": len(all_cyts), "v_out": v_out, "v_coup": v_coup, "v_ds": v_ds,
        "mr_ranks": mr_ranks, "out_deg": out_deg, "coupling_deg": coupling_deg,
    }


def _scatter(table, path, title):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for pool, color, marker in [("other", "0.7", "o"), ("SHALLOW", "tab:blue", "o"),
                                 ("DEEP", "tab:red", "^")]:
        members = (SHALLOW_POOL if pool == "SHALLOW" else DEEP_POOL if pool == "DEEP"
                   else [c for c in table if c not in SHALLOW_POOL and c not in DEEP_POOL])
        xs = [table[c]["normalized_auc"] for c in members if c in table]
        ys = [table[c]["P_max"] for c in members if c in table]
        ax.scatter(xs, ys, c=color, marker=marker, s=60 if pool != "other" else 25,
                   label=pool, edgecolors="k" if pool != "other" else "none", linewidths=0.4,
                   alpha=0.9 if pool != "other" else 0.5, zorder=3 if pool != "other" else 1)
        if pool != "other":
            for c in members:
                if c in table:
                    ax.annotate(c, (table[c]["normalized_auc"], table[c]["P_max"]),
                                fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Normalized trajectory AUC  (HIGH = plateaus early = shallow)")
    ax.set_ylabel("P_max ceiling")
    ax.set_title(title)
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    args = _parse_args()
    exclude = set(args.exclude)

    seed_dicts = []
    for dp in _dyn_paths(args):
        if not dp.exists():
            print(f"skip (missing): {dp}"); continue
        with open(dp, "rb") as f:
            seed_dicts.append(pickle.load(f))
        print(f"loaded {dp}")
    if not seed_dicts:
        sys.exit("No usable dynamics.pkl found.")

    if args.truncate_epoch is not None:
        trunc_epoch, epochs_ref = args.truncate_epoch, None
    else:
        trunc_epoch, epochs_ref = _find_val_peak_epoch(seed_dicts, exclude)
        if trunc_epoch is None:
            sys.exit("Could not determine val-peak epoch (no val_records found).")

    axes_rows = _read_csv_rows(args.axes_csv)
    audited_rows = _read_csv_rows(args.audited_csv)

    full = _run_pipeline(seed_dicts, exclude, "records", None, args.ceiling_floor,
                          args.plateau_frac, axes_rows, audited_rows, "FULL window (0-250, published)")
    early = _run_pipeline(seed_dicts, exclude, "records", trunc_epoch, args.ceiling_floor,
                           args.plateau_frac, axes_rows, audited_rows,
                           f"EARLY window (0-{trunc_epoch}, pre-overfitting)")

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    _scatter(early["table"], outp.parent / "early_window_shape_vs_ceiling.png",
             f"Source-potency (EARLY window, epoch<={trunc_epoch}): shape x ceiling")

    L = []
    L.append("# Source-potency, recomputed on the pre-overfitting window\n\n")
    L.append(f"Truncation epoch (data-driven: where the aggregate val D2/D3 curve peaks, "
             f"3-seed mean): **{trunc_epoch}** of 250. Same dynamics.pkl as the published "
             f"run (`results/attention_dynamics/seed_*`), same TRAIN records "
             f"-- only the trajectory length changes; no re-training.\n\n")
    L.append("## Headline comparison: does removing the overfit tail rescue P1?\n\n")
    L.append("| window | n scored | P1: rho(potency, out-degree) | P2: rho(potency, coupling-degree) "
             "| P3: DEEP>SHALLOW p |\n|---|---:|---:|---:|---:|\n")
    for res in (full, early):
        L.append(f"| {res['label']} | {res['n_scored']}/{res['n_total']} | "
                  f"{res['v_out']['rho']:.3f} (n={res['v_out']['n']}) | "
                  f"{res['v_coup']['rho']:.3f} (n={res['v_coup']['n']}) | "
                  f"{res['v_ds']['p']:.4f} (Delta={res['v_ds']['obs_diff']:.3f}) |\n")
    L.append("\n## Master-regulator ranks, both windows\n\n")
    L.append("| master regulator | FULL rank | EARLY rank |\n|---|---:|---:|\n")
    for mr in MASTER_REGULATORS:
        fr = full["mr_ranks"].get(mr); er = early["mr_ranks"].get(mr)
        L.append(f"| {mr} | {fr if fr else 'n/a'}/{full['n_scored']} | "
                 f"{er if er else 'n/a'}/{early['n_scored']} |\n")
    L.append("\n## Full ranked table (EARLY window)\n\n")
    L.append("| rank | cytokine | source_potency | P_max | norm_auc | plateau_ep | late_gain | "
             "out_deg | coup_deg | pool |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for i, c in enumerate(early["ranked"]):
        r = early["table"][c]
        pool = "DEEP" if c in DEEP_POOL else ("SHALLOW" if c in SHALLOW_POOL else "")
        L.append(f"| {i+1} | {c} | {r['source_potency']:+.2f} | {r['P_max']:.3f} | "
                 f"{r['normalized_auc']:.3f} | {r['plateau_epoch']:.0f} | {r['late_gain']:.3f} | "
                 f"{early['out_deg'].get(c,0)} | {early['coupling_deg'].get(c,0)} | {pool} |\n")
    L.append("\n(interpretation added by hand after inspecting the numbers above)\n")
    outp.write_text("".join(L))

    print(f"\nSaved: {outp}")
    print(f"trunc_epoch={trunc_epoch}")
    print(f"FULL:  P1 rho={full['v_out']['rho']:.3f} (n={full['v_out']['n']})  "
          f"P3 p={full['v_ds']['p']:.4f}")
    print(f"EARLY: P1 rho={early['v_out']['rho']:.3f} (n={early['v_out']['n']})  "
          f"P3 p={early['v_ds']['p']:.4f}")


if __name__ == "__main__":
    main()
