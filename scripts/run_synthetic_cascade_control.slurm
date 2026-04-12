#!/bin/bash
#SBATCH --job-name=syn_cascade
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/synthetic_cascade_control

# Positive cascade pair: IL-12 -> IFN-gamma (pre-registered biological control)
/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/synthetic_cascade_control.py \
    --pair_a IL-12 \
    --pair_b IFN-gamma \
    --alpha 0.1 0.2 0.3 \
    --seed 42

# Negative control pair: IL-6 / IL-10 (shared STAT3, no documented cascade)
/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/synthetic_cascade_control.py \
    --pair_a IL-6 \
    --pair_b IL-10 \
    --alpha 0.1 0.2 0.3 \
    --seed 42
