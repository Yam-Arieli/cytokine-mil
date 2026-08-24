#!/bin/bash
#SBATCH --job-name=op_merge
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure/slurm_merge_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure/slurm_merge_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00
#SBATCH --partition=short

# Stage 5: merge the per-chunk signatures, assert uniform provenance across all nine
# chunks, and record the main-vs-reserve signature stability.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/merge_oes90_pure_signatures.py --out_dir results/oes90_pure --n_chunks 9
