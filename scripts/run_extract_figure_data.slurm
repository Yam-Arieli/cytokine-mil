#!/bin/bash
#SBATCH --job-name=extract_fig_data
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/extract_fig_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/extract_fig_%j.err
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

WORKDIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil
PYTHON=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

cd ${WORKDIR}
mkdir -p logs

echo "Starting figure data extraction at $(date)"

${PYTHON} scripts/extract_figure_data.py \
    --results_dir results/oesinghaus_full_v2 \
    --seeds 42 123 7 \
    --output results/figure_data.json

echo "Done at $(date)"
