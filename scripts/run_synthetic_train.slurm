#!/bin/bash
#SBATCH --job-name=synth_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_train_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-5

# Stage 1 only: cell-type encoder pre-training (50 epochs).
# PBS-RC geometry is purely geometric — it uses the encoder trained on cell types,
# not a cytokine-supervised MIL model. Stage 2 is not needed and is not run here.

SEEDS=(42 123 7 1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/synthetic_cascades/seed${SEED}"

echo "========================================================"
echo "Synthetic cascade — Stage 1 encoder (PBS-RC geometry)"
echo "  array_task : ${SLURM_ARRAY_TASK_ID}"
echo "  seed       : ${SEED}"
echo "  out_dir    : ${OUT_DIR}"
echo "========================================================"

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/train_synthetic_encoder_only.py \
    --seed        "${SEED}" \
    --output_dir  "${OUT_DIR}" \
    --stage1_epochs 50

echo "Done — seed=${SEED}."
