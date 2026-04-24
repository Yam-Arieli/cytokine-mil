#!/bin/bash
#SBATCH --job-name=new_seeds_geo
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/new_seeds_geo_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/new_seeds_geo_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8

# PBS-RC geometry experiment on each of the 9 new MIL seeds.
# Submitted with --dependency=afterok:<train_job_id> so it only starts
# after ALL 9 training tasks complete successfully.

SEEDS=(1 2 3 4 5 6 8 9 10)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RUN_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/new_seeds_seed${SEED}

echo "========================================================"
echo "PBS-RC Geometry Experiment (new seeds)"
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
