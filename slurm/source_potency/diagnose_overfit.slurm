#!/bin/bash
#SBATCH --job-name=source_potency_diag
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/source_potency/diag_slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/source_potency/diag_slurm_%j.err
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Diagnose why source_potency scored RED (reports/source_potency/SOURCE_POTENCY_RESULTS.md):
# checks train/val overfitting per cytokine on the same 3-seed multiclass dynamics.pkl.
# CPU-only, memory-heavy (loads 3x ~1.3GB dynamics.pkl with train+val records).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/source_potency reports/source_potency
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY -u scripts/diagnose_source_potency_overfit.py \
    --base_dir results/attention_dynamics --seeds 42 123 7 \
    --out reports/source_potency/OVERFIT_DIAGNOSIS.md
echo "OVERFIT_DIAGNOSIS DONE"
