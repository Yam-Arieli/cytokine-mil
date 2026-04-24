#!/bin/bash
#SBATCH --job-name=geo_aggregate
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/geo_aggregate_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/geo_aggregate_%j.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

# Aggregates PBS-RC results across all 12 seeds (3 existing + 9 new).
# Submitted with --dependency=afterok:<geo_job_id>.
# Saves summary.json, report.txt, rho_plot.png, poi_ranks_plot.png to
#   results/oesinghaus_full/geo_ensemble_summary/

echo "========================================================"
echo "PBS-RC Ensemble Aggregation"
echo "========================================================"

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

mkdir -p results/oesinghaus_full/geo_ensemble_summary

python scripts/aggregate_geo_ensemble.py

echo "Done."
