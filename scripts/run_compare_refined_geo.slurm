#!/bin/bash
#SBATCH --job-name=compare_refined
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/compare_refined_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/compare_refined_%j.err
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

python scripts/compare_refined_geo.py

echo "Done — comparison complete."
