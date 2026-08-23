#!/usr/bin/env python
"""Stage 4 of the Oesinghaus full-90 DAG — signature-space coupling for all 4005 pairs.

Every coupling value comes from `est.signature_coupling(...)` on the cascadir orchestrator
(cascadir-values SKILL.md: never the module-level function — it falls back to an
over-powered cell-level null and still returns a `coupled` column).

Settings, and why they are not preferences:
  donor_level=True    Oesinghaus has 10 train donors, above the ~8 threshold cascadir's own
                      code documents (pipeline.py:409-413). The donor count decides it.
  degree_correct=True Always. The validated hub/degree over-call fix (CLAUDE.md §28.2);
                      symmetric, so it never touches `cross_asym`.

The only arithmetic this script performs is Benjamini-Hochberg on the `donor_sign_p`
column cascadir emits — standard multiple-testing on its output, and the same procedure
behind `q_donor` / `coupled_q05` / `coupled_q10` in the published 276-pair file. The
family is now the neutral 4005 pairs, which is the point of the run.

Note on resolution: with 10 donors the one-sided binomial sign test is quantized —
p can only be 9.77e-4 (10/10 donors agree), 1.07e-2 (9/10), 5.47e-2 (8/10) — so the BH
threshold lands on one of those steps. Reported in the run meta.

Usage (cluster, CPU, high memory):
  python scripts/run_oesinghaus_full90_coupling.py --output_dir results/oes_full90
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _full90_config as C  # noqa: E402
from _full90_estimator import build_estimator  # noqa: E402


def bh_qvalues(pvals):
    """Benjamini-Hochberg step-up q-values (manual, no scipy — mirrors run_group_u_fdr.py).

    Applied to cascadir's `donor_sign_p`; NaNs pass through as NaN and are excluded from
    the family size.
    """
    import numpy as np

    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.where(np.isfinite(p))[0]
    m = finite.size
    if m == 0:
        return q
    order = finite[np.argsort(p[finite], kind="mergesort")]
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        prev = min(prev, p[idx] * m / rank)
        q[idx] = prev
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default="results/oes_full90")
    ap.add_argument("--coupling_alpha", type=float, default=C.COUPLING_ALPHA)
    args = ap.parse_args()

    import numpy as np

    out = Path(args.output_dir)
    est, provenance = build_estimator(out)

    n = provenance["n_conditions"]
    print(f"[coupling] {n} conditions -> {n*(n-1)//2} unordered pairs", flush=True)
    print(f"[coupling] donor_level=True (n_donors={len(provenance['donors'])}), "
          "degree_correct=True", flush=True)

    t0 = time.time()
    coupling = est.signature_coupling(
        donor_level=True, coupling_alpha=args.coupling_alpha, degree_correct=True
    )
    print(f"[coupling] est.signature_coupling done in {(time.time()-t0)/60:.1f} min "
          f"({len(coupling)} rows)", flush=True)

    if "donor_sign_p" not in coupling.columns:
        raise SystemExit(
            "FATAL: no donor_sign_p column — the donor-level gate did not run, so the "
            "`coupled` column came from the over-powered cell-level null. Refusing to "
            "write a coupling table gated that way."
        )

    coupling["q_donor"] = bh_qvalues(coupling["donor_sign_p"].to_numpy())
    for q in C.FDR_QS:
        tag = f"coupled_q{int(round(q*100)):02d}"
        coupling[tag] = (coupling["q_donor"] <= q) & (coupling["donor_coupling_mean"] > 0)

    csv_path = out / "coupling_donor_degree.csv"
    coupling.to_csv(csv_path, index=False)

    p = coupling["donor_sign_p"].to_numpy(dtype=float)
    steps = sorted({float(v) for v in p[np.isfinite(p)]})
    meta = {
        **provenance,
        "n_pairs": int(len(coupling)),
        "coupling_alpha": args.coupling_alpha,
        "donor_level": True,
        "degree_correct": True,
        "n_coupled_alpha": int(coupling["coupled"].sum()),
        "sign_test_p_levels": steps[:8],
        "n_donors_per_pair_median": float(np.nanmedian(coupling["n_donors"])),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    for q in C.FDR_QS:
        tag = f"coupled_q{int(round(q*100)):02d}"
        meta[f"n_{tag}"] = int(coupling[tag].sum())
        meta[f"frac_{tag}"] = float(coupling[tag].mean())
    C.write_json(out / "coupling_meta.json", meta)

    print(f"[gate] cascadir alpha={args.coupling_alpha}: {meta['n_coupled_alpha']} coupled", flush=True)
    for q in C.FDR_QS:
        tag = f"coupled_q{int(round(q*100)):02d}"
        print(f"[gate] BH q<={q}: {meta['n_'+tag]} coupled "
              f"({meta['frac_'+tag]:.1%} of {len(coupling)})", flush=True)
    print(f"[done] {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
