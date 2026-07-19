#!/bin/bash
#SBATCH --job-name=cfp_validate_lowlr
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency_lowlr/slurm_validate_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency_lowlr/slurm_validate_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
mkdir -p reports/cascade_forge_potency_lowlr

$PY -u scripts/validate_source_potency_cascade_forge.py \
    --seeds_dir results/cascade_forge_potency_lowlr --seeds 42 123 7 \
    --out reports/cascade_forge_potency_lowlr/SOURCE_POTENCY_VALIDATION.md

$PY -u scripts/analyze_loss_dynamics_cascade_forge.py \
    --seeds_dir results/cascade_forge_potency_lowlr --seeds 42 123 7 \
    --out reports/cascade_forge_potency_lowlr/LOSS_POTENCY_VALIDATION.md

$PY -u scripts/plot_cascade_forge_trajectories.py \
    --seeds_dir results/cascade_forge_potency_lowlr --seeds 42 123 7 \
    --zoom_epochs 500 \
    --out reports/cascade_forge_potency_lowlr/label_trajectories.png

echo "VALIDATE DONE (low-LR)"
