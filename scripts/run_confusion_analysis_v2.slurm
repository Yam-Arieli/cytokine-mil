#!/bin/bash
#SBATCH --job-name=confusion_v2
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/confusion_analysis_v2

# Analyse seed 123 (300 epochs, lr=0.003) — single seed so no stability rho yet
/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/analyze_confusion_dynamics.py \
    --results_dir results/oesinghaus_full_v2 \
    --output_dir results/confusion_analysis_v2
