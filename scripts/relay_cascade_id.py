"""
Stage 1 driver — directed cross-cell relay cascades on the Immune Dictionary.

Two modes:
  --synthetic : local self-test of the apparatus (no data). Runs BOTH regimes:
                'cell_autonomous' (direction should be recoverable) and 'tube_level'
                (direction should collapse to symmetric). Validates the pipeline and
                empirically confirms the one-hop identifiability boundary.
  --real      : load RAW ID tubes for the pre-registered conditions, build per-tube
                leave-one-out C*G samples, fit the hollow-ridge relay influence, and
                evaluate gates G1-G4 for the NK -> responder IFN-gamma relay.

See reports/immune_dictionary/RELAY_PREREGISTRATION.md and the approved plan.
Reuses cytokine_mil/analysis/relay_cascade.py (numpy apparatus).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cytokine_mil.analysis import relay_cascade as rc

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


# ---------------------------------------------------------------------------
# Shared: fit + evaluate one tube set for one relay edge
# ---------------------------------------------------------------------------
def fit_and_eval(tubes, cell_types, n_genes, edge, alpha, held_frac, n_draws,
                 n_perm, n_boot, seed, pbs_label="PBS"):
    rng = np.random.default_rng(seed)
    cts = rc.usable_cell_types(tubes, cell_types, min_cells=10, min_tube_frac=0.6)
    # ensure the edge's types survive
    for t in (edge[0], edge[2]):
        if t not in cts:
            _log(f"    WARNING: edge type {t} not usable (dropped) — relay untestable")
    pbs_means = rc.compute_pbs_type_means(tubes, cts, n_genes, pbs_label)
    samples = rc.build_relay_samples(tubes, cts, n_genes, pbs_means,
                                      held_frac=held_frac, n_draws=n_draws, rng=rng)
    A, Y = samples["A"], samples["Y"]
    layout = rc.build_flat_layout(cts, n_genes)

    # honest R^2: split by tube-draw rows into train/test
    n = len(A)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    M_tr = rc.hollow_ridge_influence(A[tr], Y[tr], alpha)
    r2_test = rc.r2(M_tr, A[te], Y[te])
    r2_test_by_type = rc.r2_per_target_type(M_tr, A[te], Y[te], layout)

    # PRIMARY direction = predictability asymmetry, on the held-out TEST split (honest)
    direction_pred = rc.relay_direction_pred(M_tr, A[te], Y[te], layout, *edge)
    # secondary (attribution only; variance-confounded) coefficient asymmetry
    M_full = rc.hollow_ridge_influence(A, Y, alpha)
    direction_coef = rc.relay_direction(M_full, layout, *edge)
    sig = rc.signal_permutation_null(A, Y, alpha, n_perm=n_perm, rng=rng)
    boot = rc.direction_bootstrap(A, Y, layout, alpha, edge, n_boot=n_boot, rng=rng)
    return {"n_samples": n, "usable_cell_types": cts,
            "r2_test": r2_test, "r2_test_by_type": r2_test_by_type,
            "signal_null": sig, "direction_pred": direction_pred,
            "direction_coef": direction_coef, "direction_boot": boot}


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------
def run_synthetic(args, out_dir):
    _log("\n=== SYNTHETIC self-test: cell_autonomous (recoverable) vs tube_level (collapse) ===")
    results = {}
    for mode in ["cell_autonomous", "tube_level"]:
        _log(f"\n-- mode = {mode}")
        rng = np.random.default_rng(args.seed)
        sim = rc.simulate_relay_tubes(mode, beta=8.0, rng=rng)
        edge = (sim["src"][0], sim["src"][1], sim["tgt"][0], sim["tgt"][1])  # S.h0 -> T.g1
        res = fit_and_eval(sim["tubes"], sim["cell_types"], sim["n_genes"], edge,
                           alpha=args.alpha, held_frac=args.held_frac, n_draws=args.n_draws,
                           n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed)
        d, b, s = res["direction_pred"], res["direction_boot"], res["signal_null"]
        c = res["direction_coef"]
        _log(f"   R2_test={res['r2_test']:.3f}  signal p={s['p_emp']:.3f} "
             f"(obs {s['observed_r2']:.3f} vs q95 {s['null_q95']:.3f})")
        _log(f"   predictability  pred_tgt(T.g1)={d['pred_tgt']:.3f}  pred_src(S.h0)={d['pred_src']:.3f}  "
             f"asym={d['asymmetry']:.3f}  boot CI=[{b['asym_q025']:.3f},{b['asym_q975']:.3f}] reliable={b['reliable']}")
        _log(f"   (secondary coef asym, variance-confounded={c['asymmetry']:.3f})")
        # recovered = TARGET predictable AND SOURCE not (asym>0) AND reliable
        recovered = b["reliable"] and d["asymmetry"] > 0.1
        expectation = "recover (asym>0)" if mode == "cell_autonomous" else "collapse (asym~0)"
        ok = recovered if mode == "cell_autonomous" else (not recovered)
        _log(f"   expected to {expectation}: {'PASS' if ok else 'FAIL'}")
        results[mode] = {**res, "recovered": bool(recovered), "expectation": expectation,
                         "self_test_pass": bool(ok)}
    verdict = (results["cell_autonomous"]["self_test_pass"] and results["tube_level"]["self_test_pass"])
    results["_synthetic_verdict"] = "APPARATUS OK" if verdict else "APPARATUS FAIL"
    _log(f"\nSYNTHETIC VERDICT: {results['_synthetic_verdict']} "
         "(recovers cell-autonomous relay; collapses on tube-level — confirms one-hop ceiling)")
    return results


# ---------------------------------------------------------------------------
# Real ID data loading (cluster) — read RAW tubes, select panel, normalize
# ---------------------------------------------------------------------------
def load_relay_tubes_real(manifest_path, conditions, panel, rng):
    import anndata, json as _json
    with open(manifest_path) as f:
        manifest = _json.load(f)
    entries = [e for e in manifest if e["cytokine"] in conditions]
    panel_set = set(panel)
    tubes, present = [], None
    for e in entries:
        raw_path = e["path"].replace(".h5ad", "_raw.h5ad")
        p = raw_path if Path(raw_path).exists() else e["path"]
        ad = anndata.read_h5ad(p)
        cols = [g for g in panel if g in set(ad.var_names)]
        if present is None:
            present = cols
            _log(f"  panel genes present in ID: {len(cols)}/{len(panel)}")
        ad = ad[:, present]
        X = ad.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        X = X.astype(np.float64)
        # normalize_total (1e4) + log1p to match the pipeline (raw counts -> normalized)
        tot = X.sum(1, keepdims=True); tot[tot == 0] = 1.0
        X = np.log1p(X / tot * 1e4)
        ct = (ad.obs["cell_type"].astype(str).values
              if "cell_type" in ad.obs.columns else np.array(["unknown"] * len(X)))
        by_type = {}
        for t in np.unique(ct):
            by_type[str(t)] = X[ct == t].astype(np.float32)
        tubes.append({"cytokine": e["cytokine"], "donor": e["donor"], "by_type": by_type})
    return tubes, present


def run_real(args, out_dir):
    _log("\n=== REAL: Immune Dictionary NK -> responder IFN-gamma relay ===")
    panel = rc.panel_gene_list()
    conditions = args.conditions
    rng = np.random.default_rng(args.seed)
    tubes, present = load_relay_tubes_real(args.manifest, conditions, panel, rng)
    n_genes = len(present)
    pidx = rc.panel_indices(present)
    src_genes = np.unique(np.concatenate([pidx.get("ifng_producer", np.array([], int)),
                                          pidx.get("ifng_transactivator", np.array([], int))]))
    tgt_genes = pidx.get("isg_target", np.array([], int))
    _log(f"  tubes={len(tubes)} conditions={conditions} n_panel_genes={n_genes} "
         f"src(IFN-axis)={len(src_genes)} tgt(ISG)={len(tgt_genes)}")

    cand_types = ["NK_cell", "Macrophage", "cDC1", "B_cell", "T_cell_CD8", "T_cell_CD4", "Treg"]
    src_type = "NK_cell"
    out = {"args": vars(args), "conditions": conditions, "n_genes": n_genes,
           "n_src_genes": int(len(src_genes)), "n_tgt_genes": int(len(tgt_genes)), "relays": {}}

    # fit on the IFN-gamma-inducing conditions
    fit_conds = [c for c in ["IL-12", "IL-18", "IL-15"] if c in conditions]
    fit_tubes = [t for t in tubes if t["cytokine"] in fit_conds or t["cytokine"] == "PBS"]
    for tgt_type in ["Macrophage", "cDC1", "B_cell"]:
        edge = (src_type, list(src_genes), tgt_type, list(tgt_genes))
        try:
            res = fit_and_eval(fit_tubes, cand_types, n_genes, edge, args.alpha,
                               args.held_frac, args.n_draws, args.n_perm, args.n_boot, args.seed)
            d, b, s = res["direction_pred"], res["direction_boot"], res["signal_null"]
            _log(f"  [{src_type}->{tgt_type}] R2_test={res['r2_test']:.3f} signal_p={s['p_emp']:.3f} "
                 f"| pred_tgt={d['pred_tgt']:.3f} pred_src={d['pred_src']:.3f} asym={d['asymmetry']:.3f} "
                 f"reliable={b['reliable']}")
            out["relays"][f"{src_type}->{tgt_type}"] = res
        except Exception as ex:
            _log(f"  [{src_type}->{tgt_type}] FAILED: {ex}")
            out["relays"][f"{src_type}->{tgt_type}"] = {"error": str(ex)}

    # negative control: IL-4 condition
    if "IL-4" in conditions:
        neg_tubes = [t for t in tubes if t["cytokine"] in ("IL-4", "PBS")]
        edge = (src_type, list(src_genes), "Macrophage", list(tgt_genes))
        try:
            res = fit_and_eval(neg_tubes, cand_types, n_genes, edge, args.alpha,
                               args.held_frac, args.n_draws, args.n_perm, args.n_boot, args.seed)
            _log(f"  [NEG IL-4 {src_type}->Macrophage] asym={res['direction_pred']['asymmetry']:.3f} "
                 f"reliable={res['direction_boot']['reliable']}")
            out["relays"]["NEG_IL4_NK->Macrophage"] = res
        except Exception as ex:
            out["relays"]["NEG_IL4_NK->Macrophage"] = {"error": str(ex)}

    # gates
    pos = [out["relays"].get(f"NK_cell->{t}", {}) for t in ["Macrophage", "cDC1", "B_cell"]]
    g1 = any(r.get("signal_null", {}).get("p_emp", 1) < 0.05 for r in pos if "signal_null" in r)
    g2 = any(r.get("direction_boot", {}).get("reliable") and r.get("direction_pred", {}).get("asymmetry", 0) > 0.1
             for r in pos)
    neg = out["relays"].get("NEG_IL4_NK->Macrophage", {})
    g3 = not neg.get("direction_boot", {}).get("reliable", False)
    out["_gates"] = {"G1_signal": bool(g1), "G2_relay_direction": bool(g2), "G3_negative_clean": bool(g3)}
    out["_verdict"] = "GREEN" if (g1 and g2) else ("RED" if not g1 else "AMBER")
    _log(f"  GATES: {out['_gates']}  -> VERDICT {out['_verdict']} (single seed; seed-stability across array)")
    return out


# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="relay cascade (ID) — Stage 1")
    p.add_argument("--synthetic", action="store_true", help="run local apparatus self-test")
    p.add_argument("--manifest", default="/cs/labs/mornitzan/yam.arieli/datasets/"
                                         "ImmuneDictionary_pseudotubes/manifest.json")
    p.add_argument("--conditions", nargs="+",
                   default=["IL-12", "IL-18", "IL-15", "IFN-gamma", "IL-4", "PBS"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--held_frac", type=float, default=0.15)
    p.add_argument("--n_draws", type=int, default=4)
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    global _LOG
    args = _parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else \
        REPO_ROOT / "results" / "relay_cascade_id" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG = open(out_dir / "train.log", "w")
    t0 = time.time()
    _log(f"relay_cascade_id | seed={args.seed} synthetic={args.synthetic}")
    res = run_synthetic(args, out_dir) if args.synthetic else run_real(args, out_dir)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(res, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    _log(f"\nDONE in {res['elapsed_sec']}s -> {out_dir}")
    _LOG.close()


if __name__ == "__main__":
    main()
