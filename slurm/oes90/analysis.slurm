#!/bin/bash
#SBATCH --job-name=o90_ana
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_analysis_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_analysis_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Stage 6: join, score against the neutral background, write reports/oesinghaus_full90/RESULTS.md.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/analyze_oesinghaus_full90.py --output_dir results/oes_full90

touch results/oes_full90/DONE_analysis
