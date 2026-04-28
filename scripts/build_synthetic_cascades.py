"""
CLI wrapper to build the synthetic cytokine cascade dataset.

Example:
    python scripts/build_synthetic_cascades.py \
        --out /cs/labs/mornitzan/yam.arieli/datasets/synthetic_cascades_v1 \
        --seed 0
"""

import argparse
from pathlib import Path

from cytokine_mil.data.synthetic_cascade_sim import (
    SimConfig,
    default_cascade_graph,
    generate_dataset,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=str, help="Output directory.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_donors", type=int, default=6)
    p.add_argument("--n_pseudo_tubes", type=int, default=8)
    p.add_argument("--n_per_cell_type", type=int, default=30)
    p.add_argument("--n_cell_types", type=int, default=12)
    p.add_argument("--n_cytokines", type=int, default=20)
    p.add_argument("--n_genes", type=int, default=512)
    p.add_argument("--no_log1p", action="store_true",
                   help="Save raw (clipped) values; default is to apply log1p.")
    args = p.parse_args()

    cfg = SimConfig(
        n_cell_types=args.n_cell_types,
        n_cytokines=args.n_cytokines,
        n_genes=args.n_genes,
        n_donors=args.n_donors,
        n_pseudo_tubes=args.n_pseudo_tubes,
        n_per_cell_type=args.n_per_cell_type,
        apply_log1p=not args.no_log1p,
        seed=args.seed,
    )

    print(f"Generating synthetic cascade dataset → {args.out}")
    print(f"  cytokines={cfg.n_cytokines} (+PBS), cell_types={cfg.n_cell_types}, "
          f"genes={cfg.n_genes}, donors={cfg.n_donors}, "
          f"tubes/(donor,cyt)={cfg.n_pseudo_tubes}")

    manifest_path = generate_dataset(
        out_dir=args.out, cfg=cfg, graph=default_cascade_graph()
    )

    out = Path(args.out)
    print(f"Done. Wrote:")
    print(f"  {manifest_path}")
    print(f"  {out/'cascade_ground_truth.json'}")
    print(f"  {out/'cytokine_programs.json'}")
    print(f"  {out/'hvg_list.json'}")
    print(f"  {out/'sim_config.json'}")


if __name__ == "__main__":
    main()
