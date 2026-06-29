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

CKPTS=$(seq -s, 10 10 250)

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/train_oesinghaus_full.py \
    --seed $SEED \
    --output_dir results/attention_dynamics/seed_$SEED \
    --checkpoint_epochs $CKPTS
