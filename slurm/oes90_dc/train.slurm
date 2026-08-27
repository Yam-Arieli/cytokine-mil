#!/bin/bash
#SBATCH --job-name=dc_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=short
#SBATCH --array=0-8%3

# Stage 3: 90 binary AB-MIL models (cytokine vs PBS), 10 per task, on the shared frozen
# encoder. Concurrency capped at 3 so the run never monopolises the a5000 nodes.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/train_oes90_dc_chunk.py \
    --out_dir results/oes90_dc \
    --chunk_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --n_chunks 9 \
    --device cuda
