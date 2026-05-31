#!/bin/bash
#SBATCH --job-name=sheu_s12
#SBATCH --gres=gpu:a5000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --array=0-2
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascade/slurm_s12_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascade/slurm_s12_%A_%a.err
#
# Path A Stage 1+2 multiclass MIL, 3 seeds (array tasks 0-2) for one time point.
# Env (set via sbatch --export): TP, MAN, HVG, RES
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

SEEDS=(42 123 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=$RES/$TP/stage12/seed_${SEED}
mkdir -p "$OUT"

echo "=== Sheu Stage1+2  TP=$TP  seed=$SEED ==="
$PY scripts/train_sheu2024_stage12.py \
    --seed "$SEED" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --output_dir "$OUT"
echo "=== DONE s12 TP=$TP seed=$SEED ==="
