#!/bin/bash
#SBATCH --job-name=oelen_lr_sweep
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen_binary_lr_sweep/slurm_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen_binary_lr_sweep/slurm_%A_%a.err
#SBATCH --array=1-10
#SBATCH --gres=gpu:a5000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=short

# LR values indexed 1-10 (bash arrays are 0-indexed, task IDs are 1-indexed)
LR_VALUES=(0.00001 0.000015 0.00002 0.000025 0.00003 0.000035 0.00004 0.00005 0.00007 0.0001)
LR=${LR_VALUES[$SLURM_ARRAY_TASK_ID-1]}

echo "Array task ${SLURM_ARRAY_TASK_ID}/10 — Stage 2 LR=${LR}"

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/train_oelen_binary.py \
    --stage2_lr ${LR} \
    --output_dir /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen_binary_lr_sweep/run_lr${LR}
