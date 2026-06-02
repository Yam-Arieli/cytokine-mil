"""Step 7 — Path B: cross_asym direction (who is upstream).

cross_asym(a, b) = s(a, S_b) - s(b, S_a), antisymmetric -> its sign is direction.
Here we use the full pipeline orchestrator for brevity, then read the direction.

Run:  python 07_path_b_direction.py
"""

from __future__ import annotations

from _common import banner, raw_anndata, CONTROL, COLS

import cascadir as cd


def main() -> None:
    banner("Step 7 — direction_table / direction (Path B)")
    adata = raw_anndata()

    est = cd.CascadeDirection(
        control_label=CONTROL,
        tube_config=cd.TubeConfig(n_tubes=5, n_per_cell_type=20, min_cells=8),
        train_config=cd.TrainConfig(encoder_epochs=5, binary_epochs=40),
        cross_asym_config=cd.CrossAsymConfig(top_n=10, min_cells=8, n_null_perms=30),
        device="cpu",
        seed=42,
        **COLS,
    ).fit(adata, assume="raw")

    print("direction table (sorted by |cross_asym|):")
    print(est.direction_table().to_string(index=False))

    call = est.direction("CytA", "CytB")
    print(f"\nCytA vs CytB -> {call.classification}, direction={call.direction}, "
          f"upstream={call.upstream}, cross_asym_median={call.cross_asym_median:+.4f}")


if __name__ == "__main__":
    main()
