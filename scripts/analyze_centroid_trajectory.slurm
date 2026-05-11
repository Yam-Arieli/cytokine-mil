#!/bin/bash
#SBATCH --job-name=centroid_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --partition=short

# Analysis of 10-seed centroid trajectory experiment.
# Loads dynamics_stage3.pkl for each seed, computes trajectory bias per donor,
# fits OLS slopes, runs Wilcoxon significance tests, reports benchmark recovery.

WORKDIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil
PYTHON=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
RESULTS_DIR=${WORKDIR}/results/centroid_trajectory
OUTPUT_DIR=${RESULTS_DIR}/analysis

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p ${OUTPUT_DIR}

cd ${WORKDIR}

echo "========================================"
echo "Job: ${SLURM_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "========================================"

${PYTHON} scripts/analyze_centroid_trajectory.py \
    --results_dir ${RESULTS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --alpha 0.05 \
    --direction_mode global

echo "Done."
