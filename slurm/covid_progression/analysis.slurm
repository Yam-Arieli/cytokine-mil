#!/bin/bash
#SBATCH --job-name=cov_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/covid_progression/slurm_analysis_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/covid_progression/slurm_analysis_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/covid_progression/plots
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/analyze_covid_progression.py \
    --fit_dir results/covid_progression/fit \
    --output_dir results/covid_progression \
    --n_boot 1000 --seed 42
