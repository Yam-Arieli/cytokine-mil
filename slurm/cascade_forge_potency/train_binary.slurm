#!/bin/bash
#SBATCH --job-name=cfp_train_binary
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency_binary/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency_binary/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# Per-label BINARY (label vs PBS) re-run of the cascade_forge dynamics experiment: removes
# the multiclass cross-label-confusability confound the user raised (two labels similar to
# EACH OTHER, not deep-cascade-wise, can take longer to separate in a 21-class softmax).
# One shared frozen Stage-1 encoder, then 20 independent binary Stage-2 models per seed
# (same lr=0.001 / stage2_epochs=4000 as the low-LR multiclass re-run, for a fair comparison).
# Reuses the existing pseudotubes (results/cascade_forge_potency/pseudotubes) -- no rebuild.
# ~20 sequential binary models per task stay well under the 12h short partition (matches
# the slurm/group_u/train.slurm precedent for a comparable per-task model count).

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
IN=results/cascade_forge_potency/pseudotubes

SEEDS=(42 123 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/cascade_forge_potency_binary/seed_${SEED}
mkdir -p "$OUT"

$PY -u scripts/train_cascade_forge_binary.py \
    --manifest "$IN/manifest.json" --gene_list "$IN/gene_list.json" \
    --val_donors donor_8,donor_9 \
    --output_dir "$OUT" --seed "$SEED" \
    --stage1_epochs 30 --stage2_epochs 4000 --lr 0.001

echo "TRAIN DONE seed=$SEED (binary per-label)"
