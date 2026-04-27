#!/bin/bash
#SBATCH --job-name=oesing_offset1_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/offset1_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/offset1_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=28:00:00
#SBATCH --partition=short
#SBATCH --array=0-11

# 12 seeds with donor_offset=1 (Stage 1 manifest shifted by one donor).
# For each cytokine i, Stage 1 encoder is trained on tube from donor (i+1)%N
# instead of donor i%N.  Produces completely disjoint training cells from the
# offset=0 batch (seeds 42,123,7,1-10), enabling data-level reproducibility.
#
# Seeds chosen to be distinct from all existing runs.

SEEDS=(11 12 13 14 15 16 17 18 19 20 21 22)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/offset1_seed${SEED}

echo "========================================================"
echo "Offset-1 Training"
echo "  array_task  : ${SLURM_ARRAY_TASK_ID}"
echo "  seed        : ${SEED}"
echo "  donor_offset: 1"
echo "  output_dir  : ${OUTPUT_DIR}"
echo "========================================================"

set -e

mkdir -p /cs/labs/mornitzan/yam.arieli/cytokine-mil/logs
mkdir -p ${OUTPUT_DIR}

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/train_oesinghaus_full.py \
    --seed ${SEED} \
    --donor_offset 1 \
    --output_dir ${OUTPUT_DIR}

echo "Done — seed=${SEED}, donor_offset=1."
