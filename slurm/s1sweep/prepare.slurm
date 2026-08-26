#!/bin/bash
#SBATCH --job-name=s1_prepare
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/s1sweep/slurm_prepare_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/s1sweep/slurm_prepare_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=short


# Stage 0: build ONE unique-cell bank, then carve a budget-matched Stage-1 set for each of
# the four arms. Also fixes the seeded panel and the nested encoder subsets.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/prepare_s1sweep.py --out_dir results/s1sweep
