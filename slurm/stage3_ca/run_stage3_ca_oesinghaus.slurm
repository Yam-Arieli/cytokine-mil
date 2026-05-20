#!/bin/bash
#SBATCH --job-name=stage3_ca_oesing
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/stage3_ca/seed_%a_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/stage3_ca/seed_%a_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --array=0-1

# Stage 3 CA-only training on full Oesinghaus 91-class data, reusing
# existing Stage 2 models. Initial array: 2 seeds (42, 123) where Stage 2
# models exist on disk. Once SA/CA separation is verified, the remaining
# 6 seeds will be trained.

SEEDS=(42 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil
STAGE2_DIR=${BASE}/results/oesinghaus_full_v2/seed_${SEED}
OUTPUT_DIR=${BASE}/results/oesinghaus_stage3_ca/seed_${SEED}

mkdir -p ${BASE}/logs/stage3_ca ${OUTPUT_DIR}
cd ${BASE}
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

echo "===================================================================="
echo "Stage 3 CA-only on full Oesinghaus 91-class"
echo "  Seed:        ${SEED}"
echo "  Stage 2 dir: ${STAGE2_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Started:     $(date)"
echo "===================================================================="

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate

python scripts/train_stage3_ca_oesinghaus_full.py \
    --seed ${SEED} \
    --stage2_dir ${STAGE2_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --stage3_epochs 100 \
    --lr 0.0002 \
    --momentum 0.95

echo "===================================================================="
echo "Done:        $(date)"
echo "===================================================================="
