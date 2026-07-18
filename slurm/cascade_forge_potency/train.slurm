#!/bin/bash
#SBATCH --job-name=cfp_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# One array task per seed (42, 123, 7). 21-class AB-MIL (20 cascade_forge labels + PBS),
# ~210 tubes total -- far smaller than Oesinghaus (91-class, 10920 tubes), so this should
# converge to a genuine train plateau well within the epoch budget below (unlike the
# borrowed lr=0.001/250-epoch Oesinghaus "warmup" regime -- see reports/source_potency/
# OVERFIT_DIAGNOSIS.md). Val donors donor_8/donor_9 (i.i.d. synthetic donors, no
# deliberate-outlier design, so val here is a trustworthy generalization check).

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
IN=results/cascade_forge_potency/pseudotubes

SEEDS=(42 123 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/cascade_forge_potency/seed_${SEED}
mkdir -p "$OUT"

$PY -u scripts/train_cascade_forge_full.py \
    --manifest "$IN/manifest.json" --gene_list "$IN/gene_list.json" \
    --val_donors donor_8,donor_9 \
    --output_dir "$OUT" --seed "$SEED" \
    --stage1_epochs 30 --stage2_epochs 400 --lr 0.01

echo "TRAIN DONE seed=$SEED"
