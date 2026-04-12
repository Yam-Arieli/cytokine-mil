#!/bin/bash
#SBATCH --job-name=oesing_full
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/slurm_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/slurm_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/train_oesinghaus_full.py \
    --seed $SEED
