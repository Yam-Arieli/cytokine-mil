#!/bin/bash
#SBATCH --job-name=pca_traj
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-01:00:00
#SBATCH --partition=short

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/figures

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    /cs/labs/mornitzan/yam.arieli/cytokine-mil/scripts/plot_centroid_pca_trajectories.py \
    --seed_dir /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/seed_42 \
    --source "IL-12" \
    --target "IFN-gamma" \
    --trajectory_type attn_weighted \
    --output_dir /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory/figures
