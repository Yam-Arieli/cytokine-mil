"""Step 8 — signature-space coupling + the degree (hub) correction.

The SECOND coupling path: build one cross-engagement matrix in signature space,
  M[a,b] = s(a, S_b) - s(PBS, S_b)         # a's cells engaging b's signature
  coupling(a,b)  = M[a,b] + M[b,a]         # SYMMETRIC   -> existence
  cross_asym(a,b)= M[a,b] - M[b,a]         # ANTISYMMETRIC -> direction (same as Path B)

The RAW coupling over-calls: a broadly-engaged ("hub") signature looks coupled to
everything. The **degree correction** (`degree_correct=True`, the default) double-centers
the coupling matrix -> pair-SPECIFIC residual. It is symmetric, so it changes only
coupling (existence), never `cross_asym` (direction).

This demo plants a hub (`make_hub_anndata`): CytA<->CytB are specifically coupled, CytC is
independent, and CytH is a hub (its program is engaged by everyone). Watch CytH's pairs
collapse under the correction while CytA-CytB survives.

NOTE on `donor_level`: the donor-level gate needs ~8+ well-covered donors. This synthetic
set has 3, so we use the cell-level degree-corrected gate (donor_level=False, the default).

Run:  python 08_signature_coupling.py
"""

from __future__ import annotations

from _common import banner, hub_anndata, CONTROL, COLS

import cascadir as cd


def _pair(df, a, b):
    lo, hi = sorted((a, b))
    row = df[(df["condition_a"] == lo) & (df["condition_b"] == hi)]
    return row.iloc[0] if len(row) else None


def main() -> None:
    banner("Step 8 — signature_coupling + degree (hub) correction")
    adata = hub_anndata()

    est = cd.CascadeDirection(
        control_label=CONTROL,
        tube_config=cd.TubeConfig(n_tubes=5, n_per_cell_type=20, min_cells=8),
        train_config=cd.TrainConfig(encoder_epochs=5, binary_epochs=40),
        cross_asym_config=cd.CrossAsymConfig(top_n=10, min_cells=8, n_null_perms=0),
        device="cpu",
        seed=42,
        **COLS,
    ).fit(adata, assume="raw")

    cols = ["condition_a", "condition_b", "coupling_raw", "coupling", "cross_asym"]

    raw = est.signature_coupling(degree_correct=False)          # M[a,b]+M[b,a]
    cor = est.signature_coupling(degree_correct=True)           # degree-corrected (default)

    print("RAW coupling (degree_correct=False) — hub CytH looks coupled to everyone:")
    print(raw[["condition_a", "condition_b", "coupling"]].to_string(index=False), "\n")

    print("DEGREE-CORRECTED coupling (default) — CytA-CytB survives, hub pairs collapse:")
    print(cor[cols].to_string(index=False), "\n")

    # read off the lesson on the three diagnostic pairs
    specific = _pair(cor, "CytA", "CytB")
    hub = _pair(cor, "CytA", "CytH")
    indep = _pair(cor, "CytA", "CytC")
    banner("what to notice")
    if specific is not None and hub is not None:
        print(f"specific CytA-CytB : coupling {specific['coupling']:+.3f} "
              f"(raw {specific['coupling_raw']:+.3f})  <- stays high")
        print(f"hub     CytA-CytH : coupling {hub['coupling']:+.3f} "
              f"(raw {hub['coupling_raw']:+.3f})  <- collapses after correction")
    if indep is not None:
        print(f"indep   CytA-CytC : coupling {indep['coupling']:+.3f}  <- ~0 / negative")
    ch = _pair(cor, "CytC", "CytH")
    if ch is not None:
        print(f"caveat  CytC-CytH : coupling {ch['coupling']:+.3f}  <- additive centering "
              "is imperfect: a node coupled ONLY to the hub keeps that one edge")
        print("        (mirrors the real-data finding that IL-15 is only PARTLY de-hubbed).")
    print("\ncross_asym (direction) is identical with/without the correction "
          "(degree centering is symmetric).")
    print("Gate it: report direction (cross_asym) only on pairs the coupling call keeps.")


if __name__ == "__main__":
    main()
