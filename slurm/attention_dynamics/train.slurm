#!/bin/bash
#SBATCH --job-name=attn_dyn_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attention_dynamics/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/attention_dynamics/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# §33 attention training-dynamics — Part A (checkpointed MULTICLASS Stage-2).
# The multiclass model is required: attention over all 91 stimuli encodes
# specificity, and the relay-lag direction stat needs all stimuli in one model.
# Checkpoints every 10 epochs feed scripts/extract_attention_trajectory.py.

seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/attention_dynamics/seed_$SEED

# Pre-registered: checkpoints every 10 over a 250-epoch Stage-2 (25 checkpoints).
# --stage2_epochs MUST be passed: the trainer default is 20, which would leave
# only checkpoints 10 & 20 and collapse the recruitment-timing resolution.
STAGE2_EPOCHS=250
CKPTS=$(seq -s, 10 10 $STAGE2_EPOCHS)

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/train_oesinghaus_full.py \
    --seed $SEED \
    --output_dir results/attention_dynamics/seed_$SEED \
    --stage2_epochs $STAGE2_EPOCHS \
    --checkpoint_epochs $CKPTS
