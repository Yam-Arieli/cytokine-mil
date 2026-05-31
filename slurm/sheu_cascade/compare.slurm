#!/bin/bash
#SBATCH --job-name=sheu_cmp
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascade/slurm_cmp_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascade/slurm_cmp_%j.err
#
# Cross-time bottom line. Depends on all per-time bridge jobs. Env: RES
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

echo "=== Sheu cross-time comparison (BOTTOM LINE) ==="
$PY scripts/compare_timepoints.py \
    --base_dir "$RES" \
    --timepoints 1hr 3hr 5hr \
    --labels_csv reports/sheu_cascade/sheu_axes_labeled.csv \
    --out reports/sheu_cascade/timepoint_comparison.md
echo "=== DONE compare ==="
cat reports/sheu_cascade/timepoint_comparison.md
