#!/bin/bash
#SBATCH --job-name=cfp_plot_traj
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency/slurm_plot_traj_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency/slurm_plot_traj_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
mkdir -p reports/cascade_forge_potency

$PY -u scripts/plot_cascade_forge_trajectories.py \
    --seeds_dir results/cascade_forge_potency --seeds 42 123 7 \
    --out reports/cascade_forge_potency/label_trajectories.png

echo "PLOT DONE"
