#!/bin/bash
#SBATCH --job-name=confdir_train
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/confusion_direction/slurm_train_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/confusion_direction/slurm_train_%A_%a.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --array=0-2

# Signature-restricted multiclass training for confusion-direction (§ pivot).
# Same as the standard 91-class multiclass run EXCEPT --hvg_path = the signature
# union U (∪ S_X ∩ HVG), so the model must discriminate on specific programs and
# the cascade leakage (A carries S_B) becomes the dominant confusion structure.
# softmax_trajectory is logged (train_mil) -> confusion tensor downstream.

seeds=(42 123 7)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/confusion_direction/restricted/seed_$SEED
UGENES=datasets/signature_union/gene_list_signature_union.json

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/train_oesinghaus_full.py \
    --seed $SEED \
    --output_dir results/confusion_direction/restricted/seed_$SEED \
    --hvg_path $UGENES \
    --stage2_epochs 250 \
    --log_every 10
