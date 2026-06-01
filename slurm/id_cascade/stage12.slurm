#!/bin/bash
#SBATCH --job-name=id_s12
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --array=0-2
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_s12_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_s12_%A_%a.err
#
# Path A Stage 1+2 multiclass AB-MIL, 3 seeds (array tasks 0-2).
# Env (sbatch --export): MAN, HVG, RES
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

RES=${RES:-results/id_cascade}
MAN=${MAN:-/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/manifest.json}
HVG=${HVG:-/cs/labs/mornitzan/yam.arieli/datasets/ImmuneDictionary_pseudotubes/hvg_list.json}

SEEDS=(42 123 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=$RES/stage12/seed_${SEED}
mkdir -p "$OUT"

echo "=== ID Stage1+2  seed=$SEED  MAN=$MAN  HVG=$HVG ==="
$PY scripts/train_immune_dictionary_stage12.py \
    --seed "$SEED" \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --output_dir "$OUT"
echo "=== DONE s12 seed=$SEED ==="
