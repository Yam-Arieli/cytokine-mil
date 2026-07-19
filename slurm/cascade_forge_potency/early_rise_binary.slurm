#!/bin/bash
#SBATCH --job-name=cfp_early_rise
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency_binary/slurm_early_rise_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency_binary/slurm_early_rise_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --partition=short

# Cheap follow-up: no new training, just re-reads the already-collected binary
# dynamics.pkl trajectories to test whether an early-epoch (first 50) rise-rate feature,
# alone or combined with source_potency, separates true sources from isolated negatives
# better than source_potency alone.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY -u scripts/analyze_early_rise_cascade_forge.py \
    --seeds_dir results/cascade_forge_potency_binary --seeds 42 123 7 \
    --early_window 50 \
    --out reports/cascade_forge_potency_binary/EARLY_RISE_ANALYSIS.md

echo "EARLY RISE ANALYSIS DONE"
