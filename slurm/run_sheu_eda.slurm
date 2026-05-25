#!/bin/bash
#SBATCH --job-name=sheu_eda
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_eda_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_eda_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Pair-level EDA benchmark on Sheu phase-1 pseudotubes.
# Computes the statistic battery, AUC vs permutation null, and plots.

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_sheu_eda_benchmark.py \
    --manifest /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json \
    --out_dir  /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_eda \
    --min_cells 10 \
    --n_permutations 2000 \
    --seed 42
