#!/bin/bash
#SBATCH --job-name=gu_coup_oes
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_coup_oes_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_coup_oes_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=short

# Signature-space coupling on Oesinghaus (CLAUDE.md §28). Uses the all-45-cytokine
# signatures from the Group-U ig_merge step (binary_ig_all45). Computes the full
# cross-engagement matrix over all 45 cytokines -> coupling axes (gene-set-null
# gated) + cross_asym direction; compares the coupling axis set + literature
# support to latent-geometry Path A (cytokine_axes.csv). Train donors only
# (D2/D3 excluded), matching Path A's PBS-RC split.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

IG=results/group_u/binary_ig_all45/binary_ig.parquet
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json
AXES=reports/cascade_pairs/cytokine_axes.csv

$PY scripts/run_signature_coupling.py \
    --dataset oesinghaus \
    --binary_ig_parquet "$IG" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --axes_csv "$AXES" \
    --exclude_donors Donor2 Donor3 \
    --top_n 50 --min_cells 10 --n_perm 1000 \
    --output_dir results/group_u/coupling_oes

echo "DONE. report at results/group_u/coupling_oes/coupling_report.md"
