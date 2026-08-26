#!/bin/bash
#SBATCH --job-name=eg_probe
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encoder_geometry/slurm_probe_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encoder_geometry/slurm_probe_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --partition=short

# Stage 1: probe every saved Stage-1 encoder's gene -> embedding map.
# Read-only over existing artifacts: no training, no checkpoints, no GPU.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
export OMP_NUM_THREADS=8

PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/probe_encoder_gene_geometry.py \
    --out_dir results/encoder_geometry \
    --n_jac_cells "${N_JAC_CELLS:-8}" \
    --device cpu

touch results/encoder_geometry/DONE_probe
