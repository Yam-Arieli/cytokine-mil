#!/bin/bash
#SBATCH --job-name=s1_encoder
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/s1sweep/slurm_encoder_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/s1sweep/slurm_encoder_%A_%a.err
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:a5000
#SBATCH --array=0-3

# Stage 1: one Stage-1 encoder per arm. The arms differ ONLY in how many distinct
# perturbations their cell set contains; total cells and epochs are matched.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

ARMS=(pub_replica pub_replica_clean vol_small vol_large)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "[slurm] arm=$ARM"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/train_encsweep_encoder.py \
    --out_dir results/s1sweep --arm "$ARM" --device cuda
