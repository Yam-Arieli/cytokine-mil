"""Author a cascade -> forge a snapshot -> let cascadir recover the direction.

Run:
    pip install -e cascade_forge -e cascadir
    python cascade_forge/examples/quickstart.py

Prints cascadir's direction accuracy against the planted ground truth. The symmetric
directional_score control should sit near chance — the contrast is the proof that the
antisymmetric cross_asym is doing the work.
"""

from __future__ import annotations

import cascade_forge as cf


def main() -> None:
    # ---- 1. author the ground-truth cascade -------------------------------------
    cascades = {
        "AlphaKine": {"BetaKine": (0.75, 2.0)},     # A -> B (strength 0.75, delta 2.0)
        "BetaKine": {"GammaKine": (0.65, 2.0)},     # B -> C  (2-hop chain)
        "DeltaKine": {"EpsilonKine": (0.7, 1.0)},   # an independent edge
        "OmegaKine": {"BetaKine": (0.5, 1.0)},      # a second parent of B (fan-in)
        # GammaKine / EpsilonKine appear only downstream -> exist, no outgoing cascade
    }

    # ---- 2. forge the single-cell snapshot --------------------------------------
    sim = cf.CascadeSimulator(
        cascades,
        n_cell_types=4,
        n_cells_per_tube=300,
        n_donors=6,
        effect_size=0.30,        # weak program bump << cell-type marker gap (~0.9)
        responder_mode="all",
        output="raw",
        seed=0,
    )
    result = sim.simulate(snapshot_times=[4.0])   # a well-developed snapshot
    adata = result.adata
    print(f"forged AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes, "
          f"conditions={sorted(set(adata.obs['condition']))}")
    print(f"planted direct edges: {result.direct_edges}")

    # ---- 3. let cascadir recover the direction ----------------------------------
    import cascadir as cd

    cd.validate_anndata(
        adata, condition_col="condition", donor_col="donor",
        celltype_col="cell_type", control_label="PBS",
    )
    est = cd.CascadeDirection(
        condition_col="condition", donor_col="donor", celltype_col="cell_type",
        control_label="PBS",
    ).fit(adata, assume="raw")

    # feedback pairs are ambiguous by design; benchmark the unambiguous authored edges
    bidir = {frozenset(p) for p in result.bidirectional_pairs}
    edges = [e for e in result.direct_edges if frozenset(e) not in bidir]
    bench = est.benchmark(edges)

    print("\n=== cascadir vs planted ground truth ===")
    print(f"labeled edges          : {bench.n_labeled}  (found {bench.n_found})")
    print(f"cross_asym accuracy    : {bench.cross_accuracy_all:.0%}   <- the direction call")
    print(f"directional_score ctrl : {bench.dirscore_accuracy:.0%}   <- symmetric, ~chance")
    print(f"classification breakdown: {bench.classification_counts}")
    print("\nper-pair:")
    cols = ["condition_a", "condition_b", "expected_upstream",
            "called_upstream", "cross_asym_median", "cross_correct"]
    print(bench.table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
