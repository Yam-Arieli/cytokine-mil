#!/bin/bash
#SBATCH --job-name=cell_degree_sheu
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sig_ablation/slurm_celldeg_sheu_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sig_ablation/slurm_celldeg_sheu_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Cell-level degree-corrected coupling on Sheu 3hr (targeted 500-gene panel). The
# donor-level gate is inapplicable to Sheu (4 donors), but the DEGREE CORRECTION can be
# tested at the cell level where all 21 pairs (incl. LPS-IFNb, PIC-IFNb) are testable.
# Question: does HUB (degree-corrected) keep the textbook IFN cascades top-ranked while
# cutting the RAW over-call? Read by coupling_hub RANKING (null_p_hub is permissive for
# small-positive pairs -- the known magnitude-floor caveat). Research code only.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

IG=results/sheu_cascade/3hr/binary_ig/binary_ig.parquet
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json

$PY scripts/run_cell_degree_coupling.py \
    --dataset sheu \
    --binary_ig_parquet "$IG" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --time_filter 3hr \
    --top_n 50 --min_cells 10 --n_perm 500 \
    --output_dir results/sig_ablation/sheu3hr/cell_degree

echo "DONE. report at results/sig_ablation/sheu3hr/cell_degree/cell_degree_report.md"
