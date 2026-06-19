#!/bin/bash
#SBATCH --job-name=id_celldeg
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_celldeg_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_celldeg_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=short

# Cell-level degree-corrected coupling on the Immune Dictionary (Cui 2024; mouse
# in-vivo lymph node, 4 h, whole-transcriptome 4000-HVG). The third-dataset test of
# the coupling-gate fix (CLAUDE.md §28.2): ID has only 3 donors for benchmark
# cytokines (rep01/02/03) -> the donor-level null is inapplicable (p-floor 1/2^3),
# so this uses the CELL-LEVEL degree correction validated on Sheu, now on a BROAD
# in-vivo panel (vs Sheu's targeted 500-gene panel).
#
# Question: does HUB (degree-corrected) keep the known cascades coupled
# (IFNb-IFNg, IFNg-IL2/IL15/IL18, IL1b-IL6, IL6-TNFa, IFNg-IL12) while cutting the
# RAW over-call, and suppress the antagonisms (IFNg-IL4, IL10-IL12)? Read by
# coupling_hub RANKING (null_p_hub is permissive for small-positive pairs -- the
# known magnitude-floor caveat). Research code only. Reuses the existing §26
# binary_ig signatures S_X (12 benchmark cytokines). No training, no GPU.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

IG=results/id_cascade/binary_ig/binary_ig.parquet
MAN=/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/hvg_list.json
AXES=reports/immune_dictionary/id_axes_labeled.csv

mkdir -p results/id_cascade/coupling_cell_degree

$PY scripts/run_cell_degree_coupling.py \
    --dataset id \
    --binary_ig_parquet "$IG" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --axes_csv "$AXES" \
    --top_n 50 --min_cells 10 --n_perm 500 \
    --output_dir results/id_cascade/coupling_cell_degree

echo "DONE. report at results/id_cascade/coupling_cell_degree/cell_degree_report.md"
