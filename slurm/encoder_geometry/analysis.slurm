#!/bin/bash
#SBATCH --job-name=eg_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encoder_geometry/slurm_analysis_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encoder_geometry/slurm_analysis_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --partition=short

# Stage 2: attach each fit's measured signature diversity, evaluate B1-B4, render figures.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/analyze_encoder_geometry.py \
    --out_dir results/encoder_geometry \
    --report reports/encoder_geometry/ENCODER_GEOMETRY.md

touch results/encoder_geometry/DONE_analysis
