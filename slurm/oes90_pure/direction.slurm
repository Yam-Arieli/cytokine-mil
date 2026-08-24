#!/bin/bash
#SBATCH --job-name=op_dir
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure/slurm_direction_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure/slurm_direction_%j.err
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --partition=short

# Stage 7: cross_asym direction for all 4005 pairs + the per-cell-type engagement values.
# ~2 tube-set copies at ~25 GB each; the per-pair random-gene-set null dominates runtime.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/run_oes90_pure_direction.py --out_dir results/oes90_pure
