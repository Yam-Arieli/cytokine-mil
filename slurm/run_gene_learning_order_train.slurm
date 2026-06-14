#!/bin/bash
#SBATCH --job-name=glo_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/gene_learning_order/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/gene_learning_order/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# Gene learning-order experiment — Stage 2 training (binary AB-MIL, per-epoch checkpoints).
# CLAUDE.md §2.5 + reports/gene_cascade_direction/LEARNING_ORDER_PREREGISTRATION.md.
# Seeds 42, 123, 7 run in parallel via array (indices 0-2).
# Each seed trains a shared Stage-1 encoder + per-stimulus binary models for polyIC and LPS,
# checkpointing EVERY Stage-2 epoch so the attribution extractor can build the full trajectory.

seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/gene_learning_order/seed_$SEED

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/train_sheu_binary_learning_order.py \
    --seed $SEED \
    --output_dir results/gene_learning_order/seed_$SEED
