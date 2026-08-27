"""Stage 6 of the §40 dropout+curation run — signature-space coupling, one arm.

Rehydrates the fit with `CascadeDirection.from_artifacts` and calls the orchestrator's
`signature_coupling(donor_level=True, degree_correct=True)`:

  * **donor_level** — the unit of independence is the donor. The cell-level gene-set null
    is over-powered (thousands of cells make almost any nonzero asymmetry "significant"),
    and this run has 10 well-covered donors, comfortably past the ~8 the donor-level sign
    test needs (CLAUDE.md §28.2).
  * **degree_correct** — double-centring the coupling matrix removes the hub bias. It is
    symmetric, so it never touches direction.

BH-FDR is then applied to cascadir's emitted `donor_sign_p` over the whole 4005-pair
family. That is standard multiple-testing on cascadir's output, not a re-derivation of it.

`--arm` picks which signatures to score: `curated` (§40's result) or `raw` (the uncurated
top-200 control). Both arms are run; without the control there is no way to say whether the
curation helped, hurt, or did nothing. The arms may cover DIFFERENT condition sets —
curation drops conditions it empties — so their pair counts can differ and any comparison
between them must be made on the intersection.

No coupling arithmetic happens in this file. Calling the module-level
`signature_coupling()` instead of the orchestrator would silently fall back to the
over-powered cell-level null while still returning a `coupled` column — so the script
refuses to write if `donor_sign_p` is missing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascadir" / "src"))

import _oes90_dc_config as C  # noqa: E402
import _oes90_dc_estimator as E  # noqa: E402


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
    C.assert_agnostic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(C.OUT_DIR))
    ap.add_argument("--arm", default="curated", choices=sorted(C.ARMS))
    ap.add_argument("--alpha", type=float, default=C.COUPLING_ALPHA)
    args = ap.parse_args()

    out = Path(args.out_dir)
    est, prov = E.build_estimator(out, arm=args.arm)
    n_donors = len(prov["donors"])
    C.log(f"[gate] donor_level=True (n_donors={n_donors}), degree_correct=True")

    t0 = time.time()
    coupling = est.signature_coupling(
        donor_level=True, coupling_alpha=args.alpha, degree_correct=True
    )
    C.log(f"[coupling] {len(coupling)} pairs in {(time.time()-t0)/60:.1f} min")

    if "donor_sign_p" not in coupling.columns:
        raise AssertionError(
            "coupling table has no `donor_sign_p` — the donor-level gate did not run, so "
            "the `coupled` column came from the over-powered cell-level null. Refusing to "
            "write (CLAUDE.md §28.2)."
        )

    coupling["q_donor"] = bh_qvalues(coupling["donor_sign_p"].to_numpy())
    for q in C.FDR_QS:
        tag = f"coupled_q{int(round(q*100)):02d}"
        coupling[tag] = (coupling["q_donor"] <= q) & (coupling["donor_coupling_mean"] > 0)

    csv_path = out / f"coupling_donor_degree_{args.arm}.csv"
    coupling.to_csv(csv_path, index=False)

    n = len(coupling)
    C.log(f"\n[gates] arm={args.arm}, over the neutral {n}-pair family "
          f"({prov['n_conditions']} conditions):")
    C.log(f"  cascadir alpha={args.alpha}: {int(coupling['coupled'].sum())} coupled")
    for q in C.FDR_QS:
        tag = f"coupled_q{int(round(q*100)):02d}"
        k = int(coupling[tag].sum())
        C.log(f"  BH q<={q:.2f}: {k} coupled ({k/n:.1%})")

    import numpy as np

    finite_p = np.unique(coupling["donor_sign_p"].dropna().to_numpy())
    C.log(
        f"[note] the donor sign test is quantized: {len(finite_p)} distinct p-values with "
        f"{n_donors} donors, so a BH threshold lands on one of those steps, not an "
        "arbitrary cut."
    )

    C.write_json(out / f"coupling_meta_{args.arm}.json", {
        **prov,
        "n_pairs": n,
        "donor_level": True,
        "degree_correct": True,
        "coupling_alpha": args.alpha,
        "n_coupled_alpha": int(coupling["coupled"].sum()),
        **{
            f"n_coupled_q{int(round(q*100)):02d}": int(
                coupling[f"coupled_q{int(round(q*100)):02d}"].sum()
            )
            for q in C.FDR_QS
        },
        "distinct_donor_sign_p": [float(x) for x in finite_p[:20]],
        "elapsed_s": round(time.time() - t0, 1),
    })
    C.mark_done(out, f"coupling_{args.arm}")
    C.log(f"\n[done] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
