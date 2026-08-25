#!/bin/bash
#SBATCH --job-name=e50_prep
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure_ep50/slurm_setup_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure_ep50/slurm_setup_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --partition=short

# Stage 0: share (symlink + digest-verify) the encoder, tube split and embedding cache
# from results/oes90_pure. Nothing is retrained upstream -- the ONLY difference between
# this fit and the §37 fit is that Stage-2 stops at epoch 50.
set -e
mkdir -p /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure_ep50
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/prepare_oes90_pure_ep50.py \
    --src_dir results/oes90_pure \
    --out_dir results/oes90_pure_ep50 \
    --epochs 50
