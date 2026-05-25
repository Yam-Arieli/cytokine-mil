#!/bin/bash
#SBATCH --job-name=sheu_build_8hr
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_build_8hr_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_build_8hr_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Build Sheu pseudotubes filtered to {0hr, 8hr} for the kinetic trajectory.
# 8hr has the broadest donor coverage in Sheu (8 pseudo-donors).

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/build_pseudotubes_sheu2024.py \
    --raw_dir /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw \
    --base_path /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_8hr_pseudotubes \
    --time_points 0hr 8hr \
    --seed 42
