#!/bin/bash
#SBATCH --job-name=exp3_v3_256
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/exp3_v3_256_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/exp3_v3_256_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-14

# 15-seed ensemble: hidden_dim=256 (vs v2 which used hidden_dim=64).
# Same MIL checkpoints, same confidence filter, same seed structure.
# Purpose: compare whether larger decoder capacity recovers IL-12→IFN-γ.
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
echo "Experiment 3 v3 (hidden_dim=256, min_confidence=0.5)"
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
    --exp_name "experiment3_v3/dec${DECODER_SEED}"

echo "Done — mil_seed=${MIL_SEED} decoder_seed=${DECODER_SEED}."
