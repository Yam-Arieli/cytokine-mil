#!/bin/bash
#SBATCH --job-name=exp3_v4_F
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/exp3_v4_F_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/exp3_v4_F_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-14

# 15-seed ensemble: Approach F — geometry in frozen h_i (encoder) space.
# Decoder (hidden_dim=256) is trained normally but used only as a per-cell
# cytokine confidence weight, not as an embedding space.
# This eliminates rotation ambiguity by construction:
#   h_i is frozen across all seeds, so centroids are in a fixed coordinate frame.
# MIL seed: task 0-4 -> 42, task 5-9 -> 123, task 10-14 -> 7
MIL_SEEDS=(42 123 7)
MIL_SEED=${MIL_SEEDS[$((SLURM_ARRAY_TASK_ID / 5))]}
DECODER_SEED=${SLURM_ARRAY_TASK_ID}

RESULTS_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full

RUN_DIR=$(ls -d ${RESULTS_DIR}/run_*_seed${MIL_SEED} 2>/dev/null | tail -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: No run directory found for MIL seed=${MIL_SEED} in ${RESULTS_DIR}"
    exit 1
fi

echo "========================================================"
echo "Experiment 3 v4 (Approach F: h_i space, hidden_dim=256)"
echo "  array_task  : ${SLURM_ARRAY_TASK_ID}"
echo "  mil_seed    : ${MIL_SEED}"
echo "  decoder_seed: ${DECODER_SEED}"
echo "  run_dir     : ${RUN_DIR}"
echo "========================================================"

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/run_experiment3.py \
    --run_dir "${RUN_DIR}" \
    --device cpu \
    --epochs 50 \
    --lr 0.01 \
    --seed ${DECODER_SEED} \
    --hidden_dim 256 \
    --n_permutations 1000 \
    --min_confidence 0.5 \
    --encoder_space \
    --exp_name "experiment3_v4/dec${DECODER_SEED}"

echo "Done — mil_seed=${MIL_SEED} decoder_seed=${DECODER_SEED}."
