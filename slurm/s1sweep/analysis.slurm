#!/bin/bash
#SBATCH --job-name=s1_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/s1sweep/slurm_analysis_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/s1sweep/slurm_analysis_%A_%a.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --partition=short


# Stage 4: merge each arm's signature chunks and write the arm comparison.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/analyze_encsweep.py --out_dir results/s1sweep
