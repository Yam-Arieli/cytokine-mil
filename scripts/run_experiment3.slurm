#!/bin/bash
#SBATCH --job-name=exp3_ensemble
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/exp3_ensemble_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/exp3_ensemble_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-14

# 15-seed ensemble: cycle over 3 MIL checkpoints, vary decoder init seed.
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
echo "Experiment 3 ensemble (hidden_dim=64, min_confidence=0.5)"
echo "  array_task  : ${SLURM_ARRAY_TASK_ID}"
echo "  mil_seed    : ${MIL_SEED}"
echo "  decoder_seed: ${DECODER_SEED}"
echo "  run_dir     : ${RUN_DIR}"
echo "========================================================"

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/run_experiment3.py \
    --run_dir "${RUN_DIR}" \
    --device cpu \
    --epochs 50 \
    --lr 0.01 \
    --seed ${DECODER_SEED} \
    --n_permutations 1000 \
    --min_confidence 0.5 \
    --exp_name "experiment3_v2/dec${DECODER_SEED}"

echo "Done — mil_seed=${MIL_SEED} decoder_seed=${DECODER_SEED}."
