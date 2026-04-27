#!/bin/bash
#SBATCH --job-name=offset1_geo
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/offset1_geo_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/offset1_geo_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11

# PBS-RC geometry experiment on each of the 12 offset-1 seeds.
# Submit with --dependency=afterok:<train_job_id> so it only starts
# after ALL 12 training tasks complete successfully.

SEEDS=(11 12 13 14 15 16 17 18 19 20 21 22)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RUN_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/offset1_seed${SEED}

echo "========================================================"
echo "PBS-RC Geometry Experiment (offset-1 seeds)"
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

echo "Done — seed=${SEED}, offset=1."
