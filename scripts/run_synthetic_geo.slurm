#!/bin/bash
#SBATCH --job-name=synth_geo
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_geo_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_geo_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5

# PBS-RC geometry for synthetic cascade dataset (6 seeds).
# CPU-only; deterministic given the MIL checkpoint.

SEEDS=(42 123 7 1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RUN_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/synthetic_cascades/seed${SEED}
HVG_PATH=/cs/labs/mornitzan/yam.arieli/datasets/synthetic_cascades_v2/hvg_list.json

echo "========================================================"
echo "PBS-RC geometry — synthetic cascade"
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
    --hvg_path "${HVG_PATH}" \
    --n_permutations 1000

echo "Done — seed=${SEED}."
