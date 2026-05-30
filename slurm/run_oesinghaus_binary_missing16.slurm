#!/bin/bash
#SBATCH --job-name=oesing_bin_m16
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_binary_missing16/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_binary_missing16/slurm_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=short

# Train binary AB-MIL models for the 16 cytokines NOT in the existing
# oesinghaus_binary trained pool, so the Path A → Bridge → Path B pipeline
# can be evaluated against the full 17 KNOWN_DIRECTIONAL + 2 PRE_REGISTERED
# axes (currently only 3 of 121 axes are evaluable).
#
# HPs match the existing wide-variant trained models so the IG probe can
# load all 24 binary models uniformly via state-dict shape inference.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/oesinghaus_binary_missing16

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/train_oesinghaus_binary_missing16.py \
    --embed_dim 512 \
    --hidden_dims 512 512 \
    --attention_hidden_dim 128
