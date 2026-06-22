#!/bin/bash
#SBATCH --job-name=vac_fit_state
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_fit_state_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_fit_state_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/vaccine_progression/fit_state
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/run_vaccine_cascadir.py \
    --prepared /cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/prepared/vaccine_tcell_prepared.h5ad \
    --condition_col state --control_label Resting \
    --output_dir results/vaccine_progression/fit_state \
    --device cuda --assume auto --seed 42
