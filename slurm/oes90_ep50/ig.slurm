#!/bin/bash
#SBATCH --job-name=e50_ig
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure_ep50/slurm_ig_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure_ep50/slurm_ig_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=short
#SBATCH --array=0-8%3

# Stage 4: Integrated Gradients over the saved models, twice — MAIN tubes (the run's
# signatures) and the disjoint RESERVE tubes (the memorisation check). IG attributes back
# to genes, so it runs the full model from gene inputs and cannot use the embedding cache.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/ig_oes90_pure.py \
    --out_dir results/oes90_pure_ep50 \
    --chunk_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --n_chunks 9 \
    --device cuda
