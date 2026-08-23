#!/bin/bash
#SBATCH --job-name=o90_enc
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_encoder_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_encoder_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=short

# Stage 1: the ONE shared Stage-1 encoder + its sha256. Every chunk in stage 2 verifies
# the digest and refuses to run on a mismatch — the CLAUDE.md §27.6 guard.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/train_oesinghaus_full90_encoder.py --output_dir results/oes_full90 --device cuda

touch results/oes_full90/DONE_encoder
