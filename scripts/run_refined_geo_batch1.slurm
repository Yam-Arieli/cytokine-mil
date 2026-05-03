#!/bin/bash
#SBATCH --job-name=refined_geo_b1
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/refined_geo_b1_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/refined_geo_b1_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11

# Re-run PBS-RC geometry with the refined per-donor Wilcoxon pipeline on all
# Batch-1 (offset=1) trained models. Overwrites the old latent_geometry.pkl.

SEEDS=(11 12 13 14 15 16 17 18 19 20 21 22)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RUN_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/offset1_seed${SEED}

echo "========================================================"
echo "Refined PBS-RC Geometry — Batch 1"
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

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

python scripts/run_experiment_geo.py \
    --run_dir "${RUN_DIR}" \
    --device cpu \
    --n_permutations 1000

echo "Done — seed=${SEED}."
