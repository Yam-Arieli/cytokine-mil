#!/bin/bash
#SBATCH --job-name=synth_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_train_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-5

# Train Stage 1 encoder only (cell-type classification, 50 epochs) on the
# synthetic cascade dataset.  No Stage 2 MIL — PBS-RC geometry runs on
# encoder embeddings directly, exactly as in the Oesinghaus robustness report.

SEEDS=(42 123 7 1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "========================================================"
echo "Synthetic cascade — Stage 1 encoder only"
echo "  array_task : ${SLURM_ARRAY_TASK_ID}"
echo "  seed       : ${SEED}"
echo "========================================================"

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/train_synthetic_encoder_only.py \
    --seed "${SEED}" \
    --stage1_epochs 50

echo "Done — seed=${SEED}."
