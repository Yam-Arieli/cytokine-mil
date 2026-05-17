#!/bin/bash
#SBATCH --job-name=two_stage_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage/train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage/train_%A_%a.err
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=28:00:00
#SBATCH --partition=short
#SBATCH --array=0-7

SEEDS=(42 123 7 456 789 314 271 618)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$IDX]}
BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/two_stage_pipeline
OUT=${BASE}/exp_${IDX}_seed${SEED}

echo "========================================================"
echo "Two-Stage Train"
echo "  array_task : ${IDX}"
echo "  seed       : ${SEED}"
echo "  output_dir : ${OUT}"
echo "========================================================"

set -e

mkdir -p /cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/two_stage
mkdir -p ${OUT}

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/train_oesinghaus_full.py \
    --seed ${SEED} \
    --output_dir ${OUT} \
    --stage1_full_donors \
    --stage1_epochs 50 \
    --stage2_epochs 20 \
    --stage3_epochs 30 \
    --stage1_lr 0.01 \
    --lr 0.001

echo "Done — seed=${SEED}, exp=${IDX}."
