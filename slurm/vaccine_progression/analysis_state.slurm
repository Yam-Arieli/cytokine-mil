#!/bin/bash
#SBATCH --job-name=vac_anal_state
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_anal_state_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_anal_state_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/vaccine_progression/state/plots
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/analyze_vaccine_progression.py \
    --fit_dir results/vaccine_progression/fit_state \
    --output_dir results/vaccine_progression/state \
    --conditions Naive,Effector,Memory --control_label Resting --label state \
    --report_path reports/vaccine_progression/VACCINE_PROGRESSION_RESULTS.md \
    --n_boot 1000 --seed 42
