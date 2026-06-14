"""
Time-resolved gene cascades on the Sheu time course (PIC, LPS).

--synthetic : local apparatus self-test. Plant an early->late two-wave cascade; confirm
              activation-time ordering + directed precedence edges + V1/V2 recover it.
--real      : load raw Sheu time course (0.25->8hr), build per-gene PBS-corrected trajectories
              per stimulus (pooled + per pseudo-donor), order genes by activation time, infer
              directed gene->gene edges by temporal precedence, validate against the known IFN
              cascade (IRF3-direct precedes IFNAR-induced ISGs), and plot.

Reuses scripts/compute_sheu_realtime_emergence.py (verified raw loader: _compute_time_series,
_resolve_stimulus_label) and cytokine_mil/analysis/temporal_cascade.py. See
reports/sheu2024_temporal/TEMPORAL_CASCADE_PREREGISTRATION.md.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis import temporal_cascade as tc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

_LOG = None


def _log(m=""):
    print(m, flush=True)
    if _LOG is not None:
        print(m, file=_LOG, flush=True)


def _import_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _source_downstream():
    from cytokine_mil.analysis.pathway_signatures import PATHWAY_SIGNATURES as P
    src = set(P["IRF3_direct"]["up"])
    dwn = set(P["IFNAR_induced"]["up"])
    ov = src & dwn
    return sorted(src - ov), sorted(dwn - ov)   # drop overlap (Ifit3)


# ---------------------------------------------------------------------------
def _analyze_one(above_baseline, time_hrs, gene_names, src_genes, dwn_genes, floor, margin, rng):
    act = tc.activation_time(above_baseline, time_hrs)
    ind = tc.induced_mask(above_baseline, floor)
    gi = {g: i for i, g in enumerate(gene_names)}
    src_idx = [gi[g] for g in src_genes if g in gi and ind[gi[g]]]
    dwn_idx = [gi[g] for g in dwn_genes if g in gi and ind[gi[g]]]
    v1 = tc.validate_source_downstream(act, src_idx, dwn_idx, rng=rng)
    # directed edges among induced source+downstream genes (the cascade we can validate)
    edges = tc.directed_edges(above_baseline, time_hrs, src_idx + dwn_idx, gene_names, act, margin=margin)
    v2 = tc.edge_direction_fraction(edges, src_genes, dwn_genes)
    return {"activation": act, "induced": ind, "src_idx": src_idx, "dwn_idx": dwn_idx,
            "V1": v1, "V2": v2, "edges": edges}


def run_synthetic(args, out_dir):
    _log("=== SYNTHETIC self-test: planted early(source)->late(downstream) cascade ===")
    rng = np.random.default_rng(args.seed)
    ab, time_hrs, src, dwn = tc.simulate_time_cascade(rng=rng)
    gene_names = [f"src{i}" for i in src] + [f"dwn{i}" for i in range(len(dwn))] + \
                 [f"bg{i}" for i in range(ab.shape[0] - len(src) - len(dwn))]
    src_genes = gene_names[:len(src)]
    dwn_genes = gene_names[len(src):len(src) + len(dwn)]
    res = _analyze_one(ab, time_hrs, gene_names, src_genes, dwn_genes, floor=0.3, margin=0.5, rng=rng)
    v1, v2 = res["V1"], res["V2"]
    _log(f"  V1 AUC(src earlier)={v1['auc']:.3f} p={v1['p']:.3f}  (median src={v1['median_source']:.2f}h "
         f"down={v1['median_downstream']:.2f}h)")
    _log(f"  V2 edges(src<->down)={v2['n']}  frac_src_to_down={v2['frac_src_to_down']:.2f}")
    ok = (v1["p"] <= 0.05 and v1["auc"] > 0.5 and v2["frac_src_to_down"] >= 0.8)
    _log(f"  APPARATUS {'OK' if ok else 'FAIL'} (expect source earlier p<0.05 & edges run src->down)")
    out = {"V1": v1, "V2": v2, "n_edges": len(res["edges"]), "_verdict": "APPARATUS OK" if ok else "APPARATUS FAIL"}
    with open(out_dir / "synthetic.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def run_real(args, out_dir):
    import scanpy as sc
    _log("=== REAL: Sheu time-resolved cascade (PIC, LPS) ===")
    bp = _import_script("build_pseudotubes_sheu2024")
    crt = _import_script("compute_sheu_realtime_emergence")
    with open(args.hvg) as f:
        gene_names = json.load(f)
    _log(f"  loading raw Sheu time course from {args.raw_dir} ...")
    adata = bp.load_sheu_anndata(args.raw_dir)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = bp.relabel_to_pbs(adata)
    avail = set(adata.obs[crt.CYTOKINE_COL].unique())
    _log(f"  cells={adata.n_obs}  available stimuli={sorted(avail)}")
    src_genes, dwn_genes = _source_downstream()

    summary = {"stimuli": {}, "source_genes": src_genes, "downstream_genes": dwn_genes}
    rng = np.random.default_rng(args.seed)
    for stim in args.stimuli:
        label = crt._resolve_stimulus_label(stim, avail)
        if label is None:
            _log(f"  SKIP {stim}: not in data"); continue
        _log(f"\n-- {stim} (label={label})")
        ab, time_hrs, _pbs = crt._compute_time_series(adata, label, gene_names)
        res = _analyze_one(ab, time_hrs, gene_names, src_genes, dwn_genes, args.floor, args.margin, rng)
        v1, v2 = res["V1"], res["V2"]
        _log(f"   V1 AUC(src earlier)={v1['auc']:.3f} p={v1['p']:.3f} "
             f"(median src={v1['median_source']:.2f}h down={v1['median_downstream']:.2f}h; "
             f"n_src={v1['n_source']} n_down={v1['n_downstream']})")
        _log(f"   V2 edges(src<->down)={v2['n']} frac_src_to_down={v2['frac_src_to_down']:.2f}")
        # per-donor V1 AUC stability
        donor_aucs = []
        for d in sorted(adata.obs["pseudo_donor"].unique()):
            sub = adata[adata.obs["pseudo_donor"] == d]
            if (sub.obs[crt.CYTOKINE_COL] == label).sum() < 30:
                continue
            ab_d, th_d, _ = crt._compute_time_series(sub, label, gene_names)
            rd = _analyze_one(ab_d, th_d, gene_names, src_genes, dwn_genes, args.floor, args.margin, rng)
            donor_aucs.append(rd["V1"]["auc"])
        v1["donor_aucs"] = [float(x) for x in donor_aucs if np.isfinite(x)]
        v1["donor_consistent"] = bool(v1["donor_aucs"] and
                                      np.mean([a > 0.5 for a in v1["donor_aucs"]]) >= 0.5)
        summary["stimuli"][stim] = {"V1": v1, "V2": v2, "n_edges": len(res["edges"]),
                                    "time_hrs": list(map(float, time_hrs))}
        # save activation times + edges
        import pandas as pd
        act = res["activation"]
        roles = ["source" if g in set(src_genes) else "downstream" if g in set(dwn_genes) else "other"
                 for g in gene_names]
        pd.DataFrame({"gene": gene_names, "activation_hr": act, "role": roles,
                      "max_above_baseline": np.nanmax(ab, axis=1)}).to_csv(
            out_dir / f"activation_times_{stim}.csv", index=False)
        if res["edges"]:
            pd.DataFrame(res["edges"]).to_csv(out_dir / f"directed_edges_{stim}.csv", index=False)
        if HAVE_MPL:
            _plot(out_dir, stim, ab, time_hrs, gene_names, res, src_genes, dwn_genes)

    # verdict: V1 (p<=0.05, delta>0, donor-consistent) AND V2 (frac>=0.8) for BOTH stimuli
    def ok(s):
        st = summary["stimuli"].get(s)
        if not st:
            return False
        return (st["V1"]["p"] <= 0.05 and st["V1"]["auc"] > 0.5 and st["V1"].get("donor_consistent")
                and st["V2"]["frac_src_to_down"] >= 0.8)
    green = all(ok(s) for s in args.stimuli if s in summary["stimuli"])
    summary["_verdict"] = "GREEN" if green else "AMBER/RED"
    _log(f"\n  VERDICT: {summary['_verdict']}")
    with open(out_dir / "validation.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    return summary


def _plot(out_dir, stim, ab, time_hrs, gene_names, res, src_genes, dwn_genes):
    th = np.asarray(time_hrs)
    gi = {g: i for i, g in enumerate(gene_names)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # (1) mean kinetics source vs downstream
    for genes, lab, col in [(src_genes, "IRF3-direct (source)", "C3"),
                            (dwn_genes, "IFNAR-induced (downstream)", "C0")]:
        idx = [gi[g] for g in genes if g in gi]
        if idx:
            axes[0].plot(th, np.nanmean(ab[idx], axis=0), "-o", color=col, label=lab)
    axes[0].set_xlabel("time (hr)"); axes[0].set_ylabel("mean above-baseline")
    axes[0].set_title(f"{stim}: source vs downstream kinetics"); axes[0].legend()
    # (2) heatmap of induced genes ordered by activation time
    ind = res["induced"]; act = res["activation"]
    gidx = [i for i in range(len(gene_names)) if ind[i] and np.isfinite(act[i])]
    gidx = sorted(gidx, key=lambda i: act[i])
    if gidx:
        axes[1].imshow(ab[gidx], aspect="auto", cmap="magma",
                       extent=[th.min(), th.max(), len(gidx), 0])
        axes[1].set_xlabel("time (hr)"); axes[1].set_ylabel("induced genes (by activation time)")
        axes[1].set_title(f"{stim}: trajectory heatmap")
    fig.tight_layout(); fig.savefig(out_dir / f"cascade_{stim}.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Sheu time-resolved gene cascades")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--raw_dir", default="/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw")
    p.add_argument("--hvg", default="/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json")
    p.add_argument("--stimuli", nargs="+", default=["PIC", "LPS"])
    p.add_argument("--floor", type=float, default=0.15, help="min max-above-baseline for an 'induced' gene")
    p.add_argument("--margin", type=float, default=0.4, help="min activation-time gap (hr) for a directed edge")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    global _LOG
    args = _parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / "sheu_temporal_cascade"
    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG = open(out_dir / "run.log", "w")
    res = run_synthetic(args, out_dir) if args.synthetic else run_real(args, out_dir)
    _log(f"\nDONE -> {out_dir}")
    _LOG.close()


if __name__ == "__main__":
    main()
