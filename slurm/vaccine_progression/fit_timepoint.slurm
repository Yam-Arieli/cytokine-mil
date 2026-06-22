#!/bin/bash
#SBATCH --job-name=vac_fit_tp
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_fit_tp_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_fit_tp_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/vaccine_progression/fit_timepoint
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/run_vaccine_cascadir.py \
    --prepared /cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/prepared/vaccine_tcell_prepared.h5ad \
    --condition_col timepoint --control_label D0 \
    --output_dir results/vaccine_progression/fit_timepoint \
    --device cuda --assume auto --seed 42
