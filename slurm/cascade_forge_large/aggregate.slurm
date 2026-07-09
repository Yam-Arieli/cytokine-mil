#!/bin/bash
#SBATCH --job-name=cfl_agg
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_large/slurm_agg_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_large/slurm_agg_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --partition=short

# Aggregate all per-config metrics -> results/cascade_forge_large/RESULTS.md.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/aggregate_large_cascade.py --results_dir results/cascade_forge_large
echo "=== RESULTS.md ==="
cat results/cascade_forge_large/RESULTS.md
