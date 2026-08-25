#!/bin/bash
#SBATCH --job-name=e50_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure_ep50/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure_ep50/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --partition=short
#SBATCH --array=0-8%3

# Stage 1 of the epoch-50 re-run: the SAME 90 binary AB-MIL models on the SAME shared
# frozen encoder and the SAME persisted embeddings as results/oes90_pure -- stopped at
# epoch 50 instead of 250. Constant-LR SGD with no scheduler, so this is the first 50
# epochs of that run reproduced exactly, not a different optimisation.
# 1/5 the gradient steps, so 2h replaces the 8h the full schedule needed.
# Concurrency capped at 3 so the run never monopolises the a5000 nodes.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/train_oes90_pure_chunk.py \
    --out_dir results/oes90_pure_ep50 \
    --chunk_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --n_chunks 9 \
    --epochs 50 \
    --device cuda
