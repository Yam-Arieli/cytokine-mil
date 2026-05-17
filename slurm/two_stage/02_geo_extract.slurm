#!/bin/bash
#SBATCH --job-name=two_stage_geo
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage/geo_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage/geo_%A_%a.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --array=0-7

SEEDS=(42 123 7 456 789 314 271 618)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$IDX]}
BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/two_stage_pipeline
EXP_DIR=${BASE}/exp_${IDX}_seed${SEED}

echo "========================================================"
echo "Two-Stage Geo Extract"
echo "  array_task : ${IDX}"
echo "  seed       : ${SEED}"
echo "  exp_dir    : ${EXP_DIR}"
echo "========================================================"

set -e

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/run_geo_extract.py \
    --exp_dir ${EXP_DIR} \
    --output_dir ${EXP_DIR} \
    --top_pct 0.05

echo "Done — geo extract exp=${IDX}, seed=${SEED}."
