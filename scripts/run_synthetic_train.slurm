#!/bin/bash
#SBATCH --job-name=synth_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_train_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-5

# Train synthetic cascade AB-MIL for 6 seeds.
# Each task: Stage 1 encoder (50 epochs) + Stage 2 MIL (100 epochs, frozen encoder).
# Synthetic data is fast — single GPU run expected in < 2h.

SEEDS=(42 123 7 1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RESULTS_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/synthetic_cascades
OUT_DIR=${RESULTS_DIR}/seed${SEED}

echo "========================================================"
echo "Synthetic cascade training"
echo "  array_task : ${SLURM_ARRAY_TASK_ID}"
echo "  seed       : ${SEED}"
echo "  out_dir    : ${OUT_DIR}"
echo "========================================================"

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/train_synthetic_cascades.py \
    --seed "${SEED}" \
    --output_dir "${OUT_DIR}" \
    --stage1_epochs 50 \
    --stage2_epochs 100

echo "Done — seed=${SEED}."
