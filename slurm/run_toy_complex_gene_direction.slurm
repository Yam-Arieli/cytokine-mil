#!/bin/bash
#SBATCH --job-name=toy_cplx_dir
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/toy_complex_gene_direction/slurm_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/toy_complex_gene_direction/slurm_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --array=0-2

# Toy: complex-valued directional gene->gene influence from a cell "soup".
# Self-contained simulated experiment (no real data, no h5ad). Decides whether a
# complex hollow-diagonal bilinear operator recovers planted cross-cell cascade
# DIRECTION (Stage 1) and whether complex beats real on composition (Stage 2).
# Seeds 42/123/7 in parallel (array) for stability. Tiny job — minutes.

set -e
seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/toy_complex_gene_direction

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/toy_complex_gene_direction.py \
    --stage both \
    --seed $SEED \
    --output_dir results/toy_complex_gene_direction/seed_$SEED
