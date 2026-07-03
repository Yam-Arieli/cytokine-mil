#!/bin/bash
#SBATCH --job-name=confdir_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/confusion_direction/slurm_analysis_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/confusion_direction/slurm_analysis_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Confusion-direction analysis of the signature-restricted 3-seed run (afterok train):
# asymmetry -> 17-pair benchmark accuracy + temporal early/late, comparable to the
# full-gene control and cross_asym 88%.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY -u scripts/score_confusion_direction.py \
    --base_dir results/confusion_direction/restricted --seeds 42 123 7 \
    --label signature_restricted \
    --out reports/confusion_direction/CONFUSION_DIRECTION_RESULTS.md
echo "ANALYSIS DONE"
