#!/bin/bash
#SBATCH --job-name=attnreg_extract
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_reg/slurm_extract_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_reg/slurm_extract_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# §33 — per-cell-type attention trajectory per seed. A/B frozen (cache H once);
# B drops the same rare cell types as training; C reads checkpoints_stage3 and
# auto-detects the unfrozen encoder (recompute H per checkpoint), strided x5.
seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
OUT=results/attn_reg/$EXP/seed_$SEED

case $EXP in
  A) $PY -u scripts/extract_attention_trajectory.py --run_dir $OUT --device cpu ;;
  B) $PY -u scripts/extract_attention_trajectory.py --run_dir $OUT --device cpu \
        --exclude_cell_types pDC,ILC,Plasmablast ;;
  C) $PY -u scripts/extract_attention_trajectory.py --run_dir $OUT --device cpu \
        --checkpoint_subdir checkpoints_stage3 --checkpoint_stride 5 ;;
  *) echo "ERROR: unknown EXP='$EXP'"; exit 1 ;;
esac
