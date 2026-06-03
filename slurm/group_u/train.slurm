#!/bin/bash
#SBATCH --job-name=gu_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/group_u/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --array=0-7

# Group-U (§27.1): train binary AB-MIL for the cytokines MISSING from
# binary_ig_all24 (= cytokines in cytokine_axes.csv not yet covered; ~21 of 45),
# so the pipeline can direct all 121 Path A axes. CHUNKED across a GPU array so
# the ~21 models stay well under the 12h short partition; each task trains a
# contiguous slice and writes per-cytokine model_<cyt>.pt to a shared dir.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/group_u/binary_missing

PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/train_oesinghaus_binary_groupu.py \
    --output_dir results/group_u/binary_missing \
    --n_chunks 8 \
    --chunk_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --embed_dim 512 \
    --hidden_dims 512 512 \
    --attention_hidden_dim 128
