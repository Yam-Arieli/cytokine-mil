#!/bin/bash
#SBATCH --job-name=sheu_aux
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_aux_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_aux_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# Track A: Sheu phase 1 with receptor-adapter auxiliary head in Stage 1.
seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/sheu2024_aux
OUT_DIR=results/sheu2024_aux/seed_${SEED}
mkdir -p "$OUT_DIR"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/train_sheu2024_stage12_aux.py \
    --seed "$SEED" \
    --output_dir "$OUT_DIR"
