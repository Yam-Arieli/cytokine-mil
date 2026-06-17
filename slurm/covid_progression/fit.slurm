#!/bin/bash
#SBATCH --job-name=cov_fit
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/covid_progression/slurm_fit_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/covid_progression/slurm_fit_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/covid_progression/fit
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/run_covid_cascadir.py \
    --prepared /cs/labs/mornitzan/yam.arieli/datasets/COVID_Haniffa/prepared/covid_haniffa_prepared.h5ad \
    --output_dir results/covid_progression/fit \
    --device cuda --assume auto --seed 42
