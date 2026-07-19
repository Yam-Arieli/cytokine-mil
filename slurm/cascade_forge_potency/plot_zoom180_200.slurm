#!/bin/bash
#SBATCH --job-name=cfp_zoom
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency_lowlr/slurm_zoom_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency_lowlr/slurm_zoom_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY -u scripts/plot_cascade_forge_trajectories.py \
    --seeds_dir results/cascade_forge_potency_lowlr --seeds 42 123 7 \
    --zoom_start 180 --zoom_end 200 \
    --out reports/cascade_forge_potency_lowlr/label_trajectories_zoom180_200.png

echo "ZOOM PLOT DONE"
