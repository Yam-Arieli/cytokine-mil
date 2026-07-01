#!/bin/bash
#SBATCH --job-name=attnlam_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_lambda/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attn_lambda/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# §33 attention-entropy penalty LAMBDA SWEEP (Exp-A recipe only; escalate the
# penalty strength that was too weak at lambda=0.1). LAM (1|10|100) passed via
# sbatch --export=ALL,LAM=<value>. Data as-is, encoder frozen, 250 epochs,
# checkpoint every epoch, log every 10 — identical to attn_reg Exp A except lambda.
seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/attn_lambda/L$LAM
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
OUT=results/attn_lambda/L$LAM/seed_$SEED

CKPTS=$(seq -s, 1 1 250)
$PY scripts/train_oesinghaus_full.py --seed $SEED --output_dir $OUT \
    --stage2_epochs 250 --log_every 10 --checkpoint_epochs $CKPTS \
    --attn_entropy_lambda $LAM
