#!/bin/bash
#SBATCH --job-name=pca_traj_arr
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%A_%a.err
#SBATCH --array=0-8
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-01:00:00
#SBATCH --partition=short

# 9 seeds (all except seed_42 which was already plotted)
seeds=(123 7 17 31 53 97 137 191 257)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p ${BASE}/figures

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    /cs/labs/mornitzan/yam.arieli/cytokine-mil/scripts/plot_centroid_pca_trajectories.py \
    --seed_dirs ${BASE}/seed_${SEED} \
    --source "IL-12" \
    --target "IFN-gamma" \
    --trajectory_type attn_weighted \
    --output_dir ${BASE}/figures

echo "Done: seed=${SEED}"
