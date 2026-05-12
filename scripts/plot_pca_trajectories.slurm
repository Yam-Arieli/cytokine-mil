#!/bin/bash
#SBATCH --job-name=pca_traj
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --partition=short

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p ${BASE}/figures

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    /cs/labs/mornitzan/yam.arieli/cytokine-mil/scripts/plot_centroid_pca_trajectories.py \
    --seed_dirs \
        ${BASE}/seed_42 \
        ${BASE}/seed_123 \
        ${BASE}/seed_7 \
        ${BASE}/seed_17 \
        ${BASE}/seed_31 \
        ${BASE}/seed_53 \
        ${BASE}/seed_97 \
        ${BASE}/seed_137 \
    --source "IL-12" \
    --target "IFN-gamma" \
    --trajectory_type attn_weighted \
    --output_dir ${BASE}/figures
