#!/bin/bash
#SBATCH --job-name=sheu_build_1hr
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_build_1hr_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_build_1hr_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Track B: build Sheu pseudotubes filtered to {0hr, 1hr} instead of {0hr, 3hr}.
# Tests root cause #3: at 1hr the primary cascade hasn't merged with the
# secondary response yet, so the directional signal should be cleaner.

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/build_pseudotubes_sheu2024.py \
    --raw_dir /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw \
    --base_path /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_1hr_pseudotubes \
    --time_points 0hr 1hr \
    --seed 42
