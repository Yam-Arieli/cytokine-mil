"""
Analyze gene learning-order: do cascade-SOURCE genes emerge before DOWNSTREAM genes?

--synthetic : apparatus self-test on two planted regimes (cascade_order vs snr_confound).
              Validates that H1's effect-size control + H2 correctly distinguish a real
              cascade-order signal from an SNR/learnability artifact, BEFORE the cluster run.
--real      : read the per-epoch attribution trajectory + real-time emergence + the curated
              source/downstream gene labels; run H1/H2 per stimulus/seed; write a verdict.

See reports/gene_cascade_direction/LEARNING_ORDER_PREREGISTRATION.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis import learning_order as lo


def _log(m=""):
    print(m, flush=True)


# ---------------------------------------------------------------------------
def run_synthetic(args):
    _log("=== SYNTHETIC self-test: cascade_order (real) vs snr_confound (artifact) ===")
    results, ok_all = {}, True
    for regime in ["cascade_order", "snr_confound"]:
        rng = np.random.default_rng(args.seed)
        sim = lo.simulate_learning_trajectories(regime, rng=rng)
        emerg = lo.emergence_epoch(sim["traj"], sim["epochs"])
        h1 = lo.h1_source_first(emerg, sim["effsize"], sim["source_mask"],
                                sim["downstream_mask"], n_perm=2000, rng=rng)
        h2 = lo.h2_realtime(emerg, sim["realtime_emergence"], sim["effsize"],
                            n_perm=2000, rng=rng)
        _log(f"\n-- {regime}")
        _log(f"   H1 observed(down-src)={h1['observed']:.2f}  p_raw={h1['p_raw']:.3f}  "
             f"p_matched={h1['p_matched']:.3f}  (src med {h1['median_source']:.1f} vs "
             f"down med {h1['median_downstream']:.1f})")
        _log(f"   H2 spearman={h2['spearman']:.3f}  partial(|effsize)={h2['partial_spearman']:.3f}  "
             f"p_partial={h2['p_partial']:.3f}")
        if regime == "cascade_order":
            ok = (h1["p_matched"] <= 0.05) and (h2["partial_spearman"] > 0) and (h2["p_partial"] <= 0.05)
            _log(f"   expect H1_matched PASS & H2 PASS: {'PASS' if ok else 'FAIL'}")
        else:
            ok = (h1["p_raw"] <= 0.05) and (h1["p_matched"] > 0.05) and (h2["p_partial"] > 0.05)
            _log(f"   expect H1_raw signif but H1_matched & H2 NOT: {'PASS' if ok else 'FAIL'}")
        ok_all = ok_all and ok
        results[regime] = {"h1": h1, "h2": h2, "self_test_pass": bool(ok)}
    results["_verdict"] = "APPARATUS OK" if ok_all else "APPARATUS FAIL"
    _log(f"\nSYNTHETIC VERDICT: {results['_verdict']} "
         "(controls catch the SNR confound; real cascade-order survives)")
    return results


# ---------------------------------------------------------------------------
def _source_downstream_genes():
    from cytokine_mil.analysis.pathway_signatures import PATHWAY_SIGNATURES as P
    src = set(P["IRF3_direct"]["up"])
    dwn = set(P["IFNAR_induced"]["up"])
    overlap = src & dwn
    return sorted(src - overlap), sorted(dwn - overlap)   # drop ambiguous (e.g. Ifit3)


def run_real(args):
    import pandas as pd
    _log("=== REAL: gene learning-order on Sheu (per stimulus / seed) ===")
    traj_df = pd.read_parquet(args.trajectory)          # gene,epoch,stimulus,seed,attr
    rt = pd.read_csv(args.realtime).set_index("gene")   # gene -> realtime_emergence, log2fc...
    src_genes, dwn_genes = _source_downstream_genes()
    out = {"source_genes": src_genes, "downstream_genes": dwn_genes, "per": {}}
    for (stim, seed), sub in traj_df.groupby(["stimulus", "seed"]):
        piv = sub.pivot_table(index="gene", columns="epoch", values="attr", aggfunc="mean")
        genes = list(piv.index)
        epochs = sorted(sub["epoch"].unique())
        traj = np.abs(piv[epochs].to_numpy())
        emerg = lo.emergence_epoch(traj, np.array(epochs))
        gi = {g: i for i, g in enumerate(genes)}
        smask = np.array([g in set(src_genes) for g in genes])
        dmask = np.array([g in set(dwn_genes) for g in genes])
        effsize = np.array([rt["log2fc"].get(g, np.nan) for g in genes])
        rtem = np.array([rt["realtime_emergence"].get(g, np.nan) for g in genes])
        h1 = lo.h1_source_first(emerg, effsize, smask, dmask, n_perm=2000)
        h2 = lo.h2_realtime(emerg, rtem, effsize, n_perm=2000)
        out["per"][f"{stim}|seed{seed}"] = {"h1": h1, "h2": h2}
        _log(f"  {stim} seed{seed}: H1 p_matched={h1['p_matched']:.3f} | "
             f"H2 partial={h2['partial_spearman']:.3f} p={h2['p_partial']:.3f}")
    # verdict: H1_matched & H2 hold for polyIC AND LPS, majority of seeds
    def passes(stim):
        rows = [v for k, v in out["per"].items() if k.startswith(stim)]
        h1ok = np.mean([r["h1"]["p_matched"] <= 0.05 and r["h1"]["observed"] > 0 for r in rows])
        h2ok = np.mean([r["h2"]["p_partial"] <= 0.05 and r["h2"]["partial_spearman"] > 0 for r in rows])
        return h1ok >= 0.5 and h2ok >= 0.5
    green = all(passes(s) for s in ["polyIC", "LPS"] if any(k.startswith(s) for k in out["per"]))
    out["_verdict"] = "GREEN" if green else "AMBER/RED"
    _log(f"  VERDICT: {out['_verdict']}")
    return out


# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="gene learning-order analysis")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--trajectory", default=None, help="gene_attribution_trajectory.parquet")
    p.add_argument("--realtime", default=None, help="realtime_emergence.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else \
        REPO_ROOT / "results" / "gene_learning_order"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_synthetic(args) if args.synthetic else run_real(args)
    with open(out_dir / ("synthetic.json" if args.synthetic else "learning_order_results.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


if __name__ == "__main__":
    main()
