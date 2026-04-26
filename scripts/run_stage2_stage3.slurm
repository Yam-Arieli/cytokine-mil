#!/bin/bash
#SBATCH --job-name=s2s3_train
#SBATCH --array=0-4
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%A_%a.err
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --partition=short

seeds=(11 12 13 14 15)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

WORKDIR=/cs/labs/mornitzan/yam.arieli
PYTHON=${WORKDIR}/venvs/biovenv/bin/python
REPO=${WORKDIR}/cytokine-mil
OUT_DIR=${REPO}/results/oesinghaus_full/stage3_seed${SEED}

export MPLCONFIGDIR=${WORKDIR}/.matplotlib_cache
mkdir -p ${REPO}/results

# Stage 3 checkpoint epochs: every 5 epochs for 150 epochs
S3_CKPTS="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150"

cd ${REPO}
${PYTHON} scripts/train_oesinghaus_full.py \
    --seed ${SEED} \
    --output_dir ${OUT_DIR} \
    --stage1_epochs 50 \
    --stage2_epochs 15 \
    --stage3_epochs 150 \
    --encoder_lr_factor 0.1 \
    --lr 0.01 \
    --log_every 1 \
    --checkpoint_epochs_stage3 ${S3_CKPTS}
