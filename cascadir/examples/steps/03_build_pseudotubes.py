"""Step 3 — build pseudo-tubes (in-memory bags of cells, stratified by cell type).

Run:  python 03_build_pseudotubes.py
"""

from __future__ import annotations

from _common import banner, preprocessed, CONTROL, COLS

import cascadir as cd


def main() -> None:
    banner("Step 3 — build_pseudotubes")
    proc = preprocessed()

    ts = cd.build_pseudotubes(
        proc,
        control_label=CONTROL,
        n_per_cell_type=20,
        min_cells=8,
        n_tubes=4,
        seed=0,
        **COLS,
    )
    print(f"tubes: {len(ts.tubes)}")
    print(f"conditions: {ts.conditions}  (stimuli: {ts.stimulus_conditions})")
    print(f"donors: {ts.donors}")
    print(f"cell types: {ts.cell_types}")
    print(f"genes: {len(ts.gene_names)}")

    t0 = ts.tubes[0]
    print(f"\nexample tube: condition={t0.condition} donor={t0.donor} "
          f"n_cells={t0.n_cells} cell_types_included={t0.cell_types_included}")

    # the grouping cross_asym consumes: {(condition, cell_type): array}
    cbp = ts.cells_by_pair()
    print(f"\ncells_by_pair groups: {len(cbp)}; "
          f"a few keys: {sorted(cbp)[:4]}")


if __name__ == "__main__":
    main()
