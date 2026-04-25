#!/bin/bash
#SBATCH --job-name=instr_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/instr_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/instr_train_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:a5000
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4

# Instrumented training: 5 new seeds (11-15) with checkpoint saving.
# Checkpoints saved at epochs 25, 50, 75, 100 for dynamics analysis.

SEEDS=(11 12 13 14 15)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/cs/labs/mornitzan/yam.arieli
OUTPUT_DIR=${BASE}/cytokine-mil/results/oesinghaus_full/instrumented_seed${SEED}

echo "========================================================"
echo "Instrumented Training — Seed ${SEED}"
echo "  array_task : ${SLURM_ARRAY_TASK_ID}"
echo "  output_dir : ${OUTPUT_DIR}"
echo "  checkpoints: 25,50,75,100"
echo "========================================================"

mkdir -p ${OUTPUT_DIR}
set -e

source ${BASE}/venvs/biovenv/bin/activate
cd ${BASE}/cytokine-mil

python scripts/train_oesinghaus_full.py \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --stage1_epochs 50 \
    --stage2_epochs 100 \
    --lr 0.01 \
    --log_every 1 \
    --checkpoint_epochs "25,50,75,100"

echo "Done — seed=${SEED}."
