#!/bin/bash
#SBATCH --job-name=o90_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --array=0-8%3

# Stage 2: 90 binary AB-MIL models (cytokine vs PBS) + Integrated Gradients, 10 per task.
# Concurrency capped at 3 so the run never monopolises the a5000 nodes. Each task loads
# only its own conditions + PBS (~7 GB of the ~64 GB tube set).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/train_oesinghaus_full90_chunk.py \
    --output_dir results/oes_full90 \
    --chunk_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --n_chunks 9 \
    --device cuda

touch results/oes_full90/DONE_train_${SLURM_ARRAY_TASK_ID:-0}
