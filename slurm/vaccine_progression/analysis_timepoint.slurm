#!/bin/bash
#SBATCH --job-name=vac_anal_tp
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_anal_tp_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_anal_tp_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/vaccine_progression/timepoint/plots
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/analyze_vaccine_progression.py \
    --fit_dir results/vaccine_progression/fit_timepoint \
    --output_dir results/vaccine_progression/timepoint \
    --conditions D2,D11,D28 --control_label D0 --label timepoint \
    --report_path reports/vaccine_progression/VACCINE_PROGRESSION_RESULTS_timepoint.md \
    --n_boot 1000 --seed 42
