#!/bin/bash
#SBATCH --job-name=instr_geo
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/instr_geo_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/instr_geo_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4

# Method 1: Fast PBS-RC geometry at each checkpoint (no permutations).
# CPU-only. Dependent on instrumented training completing.

SEEDS=(11 12 13 14 15)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/cs/labs/mornitzan/yam.arieli
RUN_DIR=${BASE}/cytokine-mil/results/oesinghaus_full/instrumented_seed${SEED}

echo "========================================================"
echo "Geo Trajectory Extraction — Seed ${SEED}"
echo "  run_dir : ${RUN_DIR}"
echo "========================================================"

if [ ! -d "${RUN_DIR}/checkpoints" ]; then
    echo "ERROR: checkpoints/ not found in ${RUN_DIR}"
    exit 1
fi

set -e

source ${BASE}/venvs/biovenv/bin/activate
cd ${BASE}/cytokine-mil

python scripts/run_geo_trajectory.py \
    --run_dir "${RUN_DIR}" \
    --device cpu

echo "Done — seed=${SEED}."
