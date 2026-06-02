"""Step 1 — validate the data against the method's contract.

`validate_anndata` checks the obs columns, the control label, donor count, cell
counts, and the X state, and (in strict mode) raises with every problem listed.

Run:  python 01_validate.py
"""

from __future__ import annotations

from _common import CONTROL, COLS, banner, raw_anndata

import cascadir as cd


def main() -> None:
    banner("Step 1 — validate_anndata")
    adata = raw_anndata()

    # non-strict: get a report you can inspect
    report = cd.validate_anndata(
        adata, control_label=CONTROL, strict=False, n_hvgs=4000, **COLS
    )
    print(report.summary())
    print(f"\nok={report.ok}  x_state={report.x_state}  "
          f"per_condition_cells={report.per_condition_cells}\n")

    # strict mode raises with an actionable message when the contract is broken
    bad = adata.copy()
    bad.obs["cytokine"] = bad.obs["cytokine"].replace({"PBS": "rest"})  # remove control
    try:
        cd.validate_anndata(bad, control_label=CONTROL, **COLS)
    except cd.DataValidationError as exc:
        print("strict validation correctly rejected the bad data:")
        print(f"  {exc}")


if __name__ == "__main__":
    main()
