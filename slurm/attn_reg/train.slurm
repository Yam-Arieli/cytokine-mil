#!/bin/bash
#SBATCH --job-name=attnreg_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_reg/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_reg/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# §33 attention-collapse interventions. EXP (A|B|C) passed via sbatch --export.
#   A entropy:         attention-entropy penalty (lambda=0.1), data as-is
#   B entropy_hygiene: A + drop QC-dropped rare cell types
#   C unfreeze:        Stage1 -> short frozen warmup (20) -> unfreeze (230), no penalty
seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/attn_reg/$EXP
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
OUT=results/attn_reg/$EXP/seed_$SEED

case $EXP in
  A)
    CKPTS=$(seq -s, 1 1 250)
    $PY scripts/train_oesinghaus_full.py --seed $SEED --output_dir $OUT \
        --stage2_epochs 250 --log_every 10 --checkpoint_epochs $CKPTS \
        --attn_entropy_lambda 0.1
    ;;
  B)
    CKPTS=$(seq -s, 1 1 250)
    $PY scripts/train_oesinghaus_full.py --seed $SEED --output_dir $OUT \
        --stage2_epochs 250 --log_every 10 --checkpoint_epochs $CKPTS \
        --attn_entropy_lambda 0.1 --exclude_cell_types pDC,ILC,Plasmablast
    ;;
  C)
    CKPTS3=$(seq -s, 1 1 230)
    $PY scripts/train_oesinghaus_full.py --seed $SEED --output_dir $OUT \
        --stage2_epochs 20 --log_every 10 \
        --stage3_epochs 230 --checkpoint_epochs_stage3 $CKPTS3 \
        --encoder_lr_factor 0.1 --stage3_lr 0.001 --stage3_warmup 5
    ;;
  *)
    echo "ERROR: unknown EXP='$EXP' (expected A|B|C)"; exit 1 ;;
esac
