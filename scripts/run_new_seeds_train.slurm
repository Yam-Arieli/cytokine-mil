#!/bin/bash
#SBATCH --job-name=oesing_new_seeds
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/new_seeds_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/new_seeds_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=28:00:00
#SBATCH --partition=short
#SBATCH --array=0-8

# 9 new MIL seeds for PBS-RC ensemble stability analysis.
# Seeds chosen to differ from existing runs (42, 123, 7).
# Output dirs are fixed (no timestamp) so dependent geo job can find them directly.

SEEDS=(1 2 3 4 5 6 8 9 10)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/new_seeds_seed${SEED}

echo "========================================================"
echo "New-seeds MIL Training"
echo "  array_task : ${SLURM_ARRAY_TASK_ID}"
echo "  seed       : ${SEED}"
echo "  output_dir : ${OUTPUT_DIR}"
echo "========================================================"

set -e

mkdir -p /cs/labs/mornitzan/yam.arieli/cytokine-mil/logs
mkdir -p ${OUTPUT_DIR}

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/train_oesinghaus_full.py \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR}

echo "Done — seed=${SEED}."
