#!/bin/bash
#SBATCH --job-name=sheu_p1
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# Sheu 2024 phase 1 axis-discovery training (Stage 1 + Stage 2).
# CLAUDE.md §21 specifies seeds 42, 123, 7. Job array runs all three in parallel.
#
# Resource sizing: smaller than Oesinghaus (~295K cells, 500 genes vs Oesinghaus
# ~3.4M cells × 4000 HVGs). 64G mem + 12h time should be ample headroom.

seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/sheu2024_full

# Use a stable output directory per seed so results are easy to locate
# (vs the timestamped default which is unique per launch).
OUT_DIR=results/sheu2024_full/seed_${SEED}
mkdir -p "$OUT_DIR"

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/train_sheu2024_stage12.py \
    --seed "$SEED" \
    --output_dir "$OUT_DIR"
