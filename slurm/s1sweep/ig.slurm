#!/bin/bash
#SBATCH --job-name=s1_ig
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/s1sweep/slurm_ig_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/s1sweep/slurm_ig_%A_%a.err
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:a5000
#SBATCH --array=0-15%3

# Stage 3: Integrated Gradients signatures (top-50) per arm. Same task->(arm, chunk) map
# as train. IG runs the full model from gene space, so it cannot use the embedding cache.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

ARMS=(pub_replica pub_replica_clean vol_small vol_large)
TID=${SLURM_ARRAY_TASK_ID:-0}
ARM=${ARMS[$((TID / 4))]}
CHUNK=$((TID % 4))
echo "[slurm] task=$TID arm=$ARM chunk=$CHUNK"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/ig_encsweep.py \
    --out_dir results/s1sweep --arm "$ARM" \
    --chunk_id "$CHUNK" --n_chunks 4 --device cuda
