#!/bin/bash
#SBATCH --job-name=p2_enc
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/code_path/slurm_p2enc_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/code_path/slurm_p2enc_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000
#SBATCH --time=6:00:00
#SBATCH --partition=short
#SBATCH --array=0-5%3

# Phase 1+2 stage A: one Stage-1 encoder per (path, seed). Six tasks = 2 paths x 3 seeds.
# Both paths train on the SAME cells (stage1_cells.h5ad), so the arms differ only in the
# encoder-construction + training code.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

PATHS=(cm cm cm cd cd cd)
SEEDS=(42 123 7 42 123 7)
P=${PATHS[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "[stage A] path=$P seed=$S"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/phase2_train_encoder.py \
    --path "$P" --seed "$S" --device cuda
