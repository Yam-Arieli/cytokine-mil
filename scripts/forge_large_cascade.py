#!/usr/bin/env python
"""Forge the large 20-label cascade_forge dataset (one config per invocation).

Authored ground truth (see cascade_forge/LARGE_EXPERIMENT_PLAN.md): a deep chain
A->B->C->D, a chain E->F->G, a fan-out H->{I,J,K} with mixed delays, a fan-in {L,M}->N,
a feedback loop O<->P, and four isolated negative-control labels Q,R,S,T. 20 labels + PBS.

Usage (one (responder_mode, effect_size, snapshot_times) config -> one output dir):
    python scripts/forge_large_cascade.py \
        --responder_mode all --effect_size 0.30 --snapshot_times 3,6 \
        --out results/cascade_forge_large/all_eff0.30
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import cascade_forge as cf

# ---- authored ground truth (locked) ----------------------------------------------
LARGE_CASCADES = {
    "A": {"B": (0.75, 2.0)}, "B": {"C": (0.65, 2.0)}, "C": {"D": (0.55, 2.0)},  # deep chain
    "E": {"F": (0.70, 1.5)}, "F": {"G": (0.60, 1.5)},                            # chain
    "H": {"I": (0.70, 1.0), "J": (0.60, 2.0), "K": (0.50, 3.0)},                 # fan-out
    "L": {"N": (0.65, 1.0)}, "M": {"N": (0.60, 1.0)},                            # fan-in
    "O": {"P": (0.60, 1.0)}, "P": {"O": (0.45, 1.0)},                            # feedback
}
ISOLATED_LABELS = ("Q", "R", "S", "T")   # negative controls (no cascade)


def scramble_labels(cascades, isolated, seed):
    """Relabel every node to a name whose SORT ORDER is decoupled from cascade order.

    Critical for a fair benchmark: in the authored graph the upstream label is always
    alphabetically before the downstream one (A->B, ...), so a trivial "first = upstream"
    rule — and the symmetric directional_score control — would score 100% for free.
    Scrambling names breaks that so cross_asym (antisymmetric) must do the real work and
    the symmetric control drops to ~chance. Returns (cascades', isolated', mapping).
    """
    labels = sorted(
        set(cascades) | {d for dn in cascades.values() for d in dn} | set(isolated)
    )
    names = [f"L{i:02d}" for i in range(len(labels))]
    random.Random(seed).shuffle(names)
    m = dict(zip(labels, names))
    new_cascades = {m[s]: {m[d]: v for d, v in dn.items()} for s, dn in cascades.items()}
    new_isolated = tuple(m[x] for x in isolated)
    return new_cascades, new_isolated, m


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responder_mode", choices=["all", "receptor"], default="all")
    p.add_argument("--effect_size", type=float, default=0.30)
    p.add_argument("--snapshot_times", default="3,6",
                   help="comma-separated pseudo-times, e.g. '3,6'")
    p.add_argument("--n_cell_types", type=int, default=8)
    p.add_argument("--n_cells_per_tube", type=int, default=5000)
    p.add_argument("--n_donors", type=int, default=10)
    p.add_argument("--output", choices=["raw", "lognorm"], default="raw")
    p.add_argument("--no_sparse", action="store_true", help="store dense X (default sparse)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scramble_seed", type=int, default=None,
                   help="if set, relabel nodes so sort-order != cascade-order (fair benchmark)")
    p.add_argument("--out", required=True, help="output directory for this config")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    times = [float(t) for t in str(args.snapshot_times).split(",") if t != ""]
    out = Path(args.out)

    cascades, isolated = LARGE_CASCADES, ISOLATED_LABELS
    if args.scramble_seed is not None:
        cascades, isolated, mapping = scramble_labels(cascades, isolated, args.scramble_seed)
        print(f"[forge] scrambled labels (seed={args.scramble_seed}): {mapping}", flush=True)

    sim = cf.CascadeSimulator(
        cascades,
        isolated_labels=isolated,
        n_cell_types=args.n_cell_types,
        n_cells_per_tube=args.n_cells_per_tube,
        n_donors=args.n_donors,
        effect_size=args.effect_size,
        responder_mode=args.responder_mode,
        output=args.output,
        sparse=not args.no_sparse,
        seed=args.seed,
    )
    print(f"[forge] {len(sim.graph.labels)} labels ({len(ISOLATED_LABELS)} isolated), "
          f"{len(sim.graph.direct)} direct edges, {sim.model.n_genes} genes | "
          f"mode={args.responder_mode} eff={args.effect_size} times={times}", flush=True)
    result = sim.simulate(snapshot_times=times)

    # Validate every snapshot against the cascadir data contract before saving.
    try:
        import cascadir as cd
        for t, adata in result.adatas.items():
            cd.validate_anndata(adata, condition_col="condition", donor_col="donor",
                                celltype_col="cell_type", control_label="PBS")
            print(f"[forge] t={t}: {adata.n_obs} cells x {adata.n_vars} genes — "
                  "validate_anndata OK", flush=True)
    except ImportError:
        warnings.warn("cascadir not installed; skipping validate_anndata", RuntimeWarning)

    paths = result.save(str(out))
    for pth in paths:
        print(f"[forge] wrote {pth}", flush=True)
    print(f"[forge] direct edges: {result.direct_edges}", flush=True)
    print(f"[forge] bidirectional (excluded from signed acc): {result.bidirectional_pairs}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
