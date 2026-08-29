#!/bin/bash
#SBATCH --job-name=p2_ana
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/code_path/slurm_p2ana_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/code_path/slurm_p2ana_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=short

# Phase 1+2 stage C: score the four arms and apply the pre-registered rule, including the
# seed control and the live falsification branch.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/analyze_phase2.py
