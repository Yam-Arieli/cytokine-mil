#!/bin/bash
#SBATCH --job-name=o90_merge
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_merge_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes_full90/slurm_merge_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Stage 3: merge chunk signatures; abort unless every chunk records the same encoder
# sha256 and the same tube sha256.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/merge_full90_signatures.py --output_dir results/oes_full90

touch results/oes_full90/DONE_merge
