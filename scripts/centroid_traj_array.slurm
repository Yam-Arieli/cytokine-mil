#!/bin/bash
#SBATCH --job-name=centroid_traj
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%A_%a.err
#SBATCH --array=0-9
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --partition=short

# Centroid trajectory experiment — 10 seeds, Stage 2a (frozen) + Stage 3 (unfrozen).
#
# Protocol (per CLAUDE.md cascade plan):
#   Stage 1  : 50 epochs, lr=0.01  — cell-type encoder pre-training
#   Stage 2  : 20 epochs, lr=0.001 — frozen warmup (Stage 2a), settles attention/FCN
#   Stage 3  : 100 epochs, lr=0.001 — full fine-tuning (Stage 2b), centroid trajectories
#
# Small LR (0.001) is critical: keeps centroid trajectories smooth, not spiky.
# Stage 1 manifest: one tube per cytokine, rotating across the 10 training donors
# (Donor1, Donor4..Donor12). Val donors Donor2+Donor3 are observer-only.

seeds=(42 123 7 17 31 53 97 137 191 257)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

WORKDIR=/cs/labs/mornitzan/yam.arieli/cytokine-mil
PYTHON=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
OUTPUT_BASE=${WORKDIR}/results/centroid_trajectory
OUTPUT_DIR=${OUTPUT_BASE}/seed_${SEED}

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p ${OUTPUT_BASE}
mkdir -p ${OUTPUT_DIR}

cd ${WORKDIR}

echo "========================================"
echo "Job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Seed: ${SEED}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "========================================"

${PYTHON} scripts/train_oesinghaus_full.py \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --stage1_epochs 50 \
    --stage1_lr 0.01 \
    --stage2_epochs 20 \
    --lr 0.001 \
    --stage3_epochs 100 \
    --stage3_lr 0.001 \
    --encoder_lr_factor 0.1 \
    --stage3_warmup 5 \
    --log_every 5 \
    --centroid_log_every 10 \
    --checkpoint_epochs_stage3 50,100 \
    --donor_offset 0

echo "Done: seed=${SEED}"
