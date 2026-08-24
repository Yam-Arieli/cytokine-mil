#!/bin/bash
#SBATCH --job-name=op_enc
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure/slurm_encoder_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure/slurm_encoder_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=short

# Stage 1: the ONE shared Stage-1 encoder (2x wide), early-stopped on a held-out
# cell-type-stratified split and run past the plateau so the history shows overfitting.
# Its sha256 is what every later stage asserts (CLAUDE.md §27.6).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/train_oes90_pure_encoder.py --out_dir results/oes90_pure --device cuda
