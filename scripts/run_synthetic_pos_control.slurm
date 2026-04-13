#!/bin/bash
#SBATCH --job-name=syn_pos_ctrl
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/synthetic_cascade_control

# GO/NO-GO gate: inject IFN-gamma cells into IL-12 tubes at alpha=0.2.
# 10-cytokine subset, 10 stage1 epochs, 20 stage2 epochs.
# Expected: C(IL-12, IFN-gamma, t) > C(IFN-gamma, IL-12, t) at late epochs.
/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/synthetic_cascade_control.py \
    --pair_a IL-12 \
    --pair_b IFN-gamma \
    --alpha 0.0 0.2 \
    --seed 42 \
    --stage1_epochs 10 \
    --stage2_epochs 20 \
    --lr 0.003 \
    --subset_n_cytokines 10
