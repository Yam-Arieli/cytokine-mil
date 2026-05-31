#!/bin/bash
#SBATCH --job-name=sheu_bin
#SBATCH --gres=gpu:a5000
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascade/slurm_bin_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascade/slurm_bin_%j.err
#
# Bridge: binary AB-MIL (stimulus vs PBS) for all stimuli at one time point.
# Env: TP, MAN, HVG, RES
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

OUT=$RES/$TP/binary
mkdir -p "$OUT"
echo "=== Sheu binary AB-MIL  TP=$TP ==="
$PY scripts/train_sheu2024_binary.py \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --output_dir "$OUT"
echo "=== DONE binary TP=$TP ==="
