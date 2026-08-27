#!/bin/bash
#SBATCH --job-name=dc_enc
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_encoder_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_encoder_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=short

# Stage 1 (§40): the ONE shared Stage-1 encoder — 2x wide (embed 1024) WITH 50% dropout
# before the embedding layer, trained for a FIXED 20 epochs. The held-out
# 10% cell-type-stratified validation split is RECORDED but does not pick the checkpoint
# (restore_best=False), so Stage-1 length is pinned rather than being a hidden variable.
# Its sha256 is what every later stage asserts (CLAUDE.md §27.6).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/train_oes90_dc_encoder.py --out_dir results/oes90_dc --device cuda
