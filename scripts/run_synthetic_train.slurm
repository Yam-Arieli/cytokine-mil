#!/bin/bash
#SBATCH --job-name=synth_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_train_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-5

# Stage 1 (cell-type encoder, 50 ep) + Stage 2 MIL (frozen encoder, 100 ep).
# MIL training is required for PBS-RC geometry to detect cytokine cascade direction.

SEEDS=(42 123 7 1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/synthetic_cascades/seed${SEED}"

echo "========================================================"
echo "Synthetic cascade — Stage 1 + Stage 2 MIL"
echo "  array_task : ${SLURM_ARRAY_TASK_ID}"
echo "  seed       : ${SEED}"
echo "  out_dir    : ${OUT_DIR}"
echo "========================================================"

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/train_synthetic_cascades.py \
    --seed        "${SEED}" \
    --output_dir  "${OUT_DIR}" \
    --stage1_epochs 50 \
    --stage2_epochs 150 \
    --lr          0.01 \
    --stage1_lr   0.01 \
    --log_every   5

echo "Done — seed=${SEED}."
