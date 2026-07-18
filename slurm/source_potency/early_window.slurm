#!/bin/bash
#SBATCH --job-name=source_potency_early
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/source_potency/early_slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/source_potency/early_slurm_%j.err
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Recompute source_potency on the pre-overfitting window (truncate at the epoch where the
# aggregate val D2/D3 curve peaks, data-driven from the same dynamics.pkl -- see
# reports/source_potency/OVERFIT_DIAGNOSIS.md) and re-run P1/P2/P3 validation.
# CPU-only, memory-heavy (loads 3x ~1.3GB dynamics.pkl with train+val records).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/source_potency reports/source_potency
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY -u scripts/recompute_source_potency_early_window.py \
    --base_dir results/attention_dynamics --seeds 42 123 7 \
    --out reports/source_potency/EARLY_WINDOW_RESULTS.md
echo "EARLY_WINDOW DONE"
