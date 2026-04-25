#!/bin/bash
#SBATCH --job-name=geo_seeds45
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/geo_seeds45_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/geo_seeds45_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1

# Retry PBS-RC geometry for seeds 4 and 5 which timed out at 6h.
# Doubled the time limit to 12h.

SEEDS=(4 5)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RUN_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/new_seeds_seed${SEED}

echo "========================================================"
echo "PBS-RC Geometry Retry (seeds 4 and 5)"
echo "  array_task : ${SLURM_ARRAY_TASK_ID}"
echo "  seed       : ${SEED}"
echo "  run_dir    : ${RUN_DIR}"
echo "========================================================"

if [ ! -f "${RUN_DIR}/model_stage2.pt" ]; then
    echo "ERROR: model_stage2.pt not found in ${RUN_DIR}"
    exit 1
fi

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/run_experiment_geo.py \
    --run_dir "${RUN_DIR}" \
    --device cpu \
    --n_permutations 1000

echo "Done — seed=${SEED}."
