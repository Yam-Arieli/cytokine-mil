#!/bin/bash
#SBATCH --job-name=cov_apparatus
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/covid_progression/slurm_apparatus_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/covid_progression/slurm_apparatus_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/covid_progression/apparatus
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/apparatus_cross_asym_ladder.py \
    --output_dir results/covid_progression/apparatus
