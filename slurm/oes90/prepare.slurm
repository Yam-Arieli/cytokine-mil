#!/bin/bash
#SBATCH --job-name=o90_prep
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_prepare_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_prepare_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=short

# Stage 0: materialise the committed Oesinghaus pseudo-tubes as (donor, condition) shards
# + the unique-cell AnnData for Stage-1. Tubes are built ONCE so every training chunk sees
# byte-identical tubes (CLAUDE.md §27.6).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/oes_full90

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/prepare_oesinghaus_full90.py --output_dir results/oes_full90

touch results/oes_full90/DONE_prepare
