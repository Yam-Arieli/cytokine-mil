#!/bin/bash
#SBATCH --job-name=vac_apptree
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_apptree_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_apptree_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/vaccine_progression/apparatus_tree
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

# 4th synthetic scenario (tree cascade) + the 3 ladder scenarios for comparison.
$PY scripts/apparatus_tree_cascade.py \
    --output_dir results/vaccine_progression/apparatus_tree
