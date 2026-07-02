#!/bin/bash
#SBATCH --job-name=selfattn_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/selfattn/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/selfattn/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# §34 self-attention over cells — Part A (checkpointed MULTICLASS Stage-2).
# Identical to §33's train (frozen encoder, 250 epochs, every-epoch checkpoints,
# SGD) EXCEPT --model_type set_transformer: cells interact via a self-attention
# block before the existing AB-MIL pooling. The saved state_dicts hold the SAB +
# pooling params; the frozen encoder means attention + the cell×cell interaction
# matrix are reconstructable per checkpoint from cached embeddings.

seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/selfattn/seed_$SEED

STAGE2_EPOCHS=250
CKPTS=$(seq -s, 1 1 $STAGE2_EPOCHS)

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/train_oesinghaus_full.py \
    --seed $SEED \
    --output_dir results/selfattn/seed_$SEED \
    --model_type set_transformer \
    --sab_heads 4 --sab_layers 1 \
    --stage2_epochs $STAGE2_EPOCHS \
    --log_every 10 \
    --checkpoint_epochs $CKPTS
