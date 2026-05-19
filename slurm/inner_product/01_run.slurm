#!/bin/bash
#SBATCH --job-name=ip_pairs
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/inner_product/ip_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/inner_product/ip_%A_%a.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --array=0-7

SEEDS=(42 123 7 456 789 314 271 618)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$IDX]}
BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/two_stage_pipeline
EXP_DIR=${BASE}/exp_${IDX}_seed${SEED}

echo "========================================================"
echo "Inner-Product Pair Detection"
echo "  array_task : ${IDX}"
echo "  seed       : ${SEED}"
echo "  exp_dir    : ${EXP_DIR}"
echo "========================================================"

set -e

mkdir -p /cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/inner_product
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/run_inner_product_pairs.py \
    --exp_dir ${EXP_DIR} \
    --output_dir ${EXP_DIR} \
    --top_pct 0.20 \
    --pca_dims 6 24 \
    --seed ${SEED}

echo "Done — inner-product pairs exp=${IDX}, seed=${SEED}."
