#!/bin/bash
#SBATCH --job-name=cfp_train_lowlr
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency_lowlr/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency_lowlr/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# Low-LR re-run: same cascade_forge pseudo-tubes as results/cascade_forge_potency (reused,
# no rebuild), but lr=0.001 (10x lower) and stage2_epochs=4000 (10x more) -- tests whether
# the fast (~30-50 epoch) convergence in the lr=0.01/400-epoch run was collapsing a
# staged (primary-then-secondary) learning order that a slower optimizer might reveal.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
IN=results/cascade_forge_potency/pseudotubes

SEEDS=(42 123 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/cascade_forge_potency_lowlr/seed_${SEED}
mkdir -p "$OUT"

$PY -u scripts/train_cascade_forge_full.py \
    --manifest "$IN/manifest.json" --gene_list "$IN/gene_list.json" \
    --val_donors donor_8,donor_9 \
    --output_dir "$OUT" --seed "$SEED" \
    --stage1_epochs 30 --stage2_epochs 4000 --lr 0.001

echo "TRAIN DONE seed=$SEED (low-LR)"
