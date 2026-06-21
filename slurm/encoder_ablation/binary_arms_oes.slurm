#!/bin/bash
#SBATCH --job-name=enc_abl_tr
#SBATCH --array=0-1
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encoder_ablation/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/encoder_ablation/slurm_train_%A_%a.err
#
# Encoder-ablation: does the Stage-1 cell-type PRE-TRAINING matter, or does a single
# end-to-end train match it? Two arms on Oesinghaus (same 24 cytokines, seed, data):
#   task 0 = two_stage    (Stage-1 cell-type pretrain -> frozen Stage-2 MIL; current)
#   task 1 = single_stage (random-init encoder, end-to-end, encoder unfrozen, no Stage-1)
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json
IGREF=results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet

TARGETS=$($PY -c "import pandas as pd; print(' '.join(sorted(pd.read_parquet('$IGREF').cytokine.astype(str).unique())))")

case $SLURM_ARRAY_TASK_ID in
  0) ARM=two_stage;    FLAG="" ;;
  1) ARM=single_stage; FLAG="--single_stage --single_stage_lr 1e-3" ;;
  *) echo "bad idx"; exit 2 ;;
esac

OUT=results/encoder_ablation/$ARM/binary
mkdir -p "$OUT"
echo "=== encoder-ablation arm=$ARM  targets=[$TARGETS] ==="
$PY scripts/train_immune_dictionary_binary.py \
    --manifest_path "$MAN" --hvg_path "$HVG" \
    --targets $TARGETS --val_donors Donor2 Donor3 \
    $FLAG \
    --output_dir "$OUT"
echo "=== DONE arm=$ARM ==="
