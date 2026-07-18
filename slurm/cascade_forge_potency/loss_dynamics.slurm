#!/bin/bash
#SBATCH --job-name=cfp_loss
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency/slurm_loss_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency/slurm_loss_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
mkdir -p reports/cascade_forge_potency

$PY -u scripts/analyze_loss_dynamics_cascade_forge.py \
    --seeds_dir results/cascade_forge_potency --seeds 42 123 7 \
    --out reports/cascade_forge_potency/LOSS_POTENCY_VALIDATION.md

echo "LOSS ANALYSIS DONE"
