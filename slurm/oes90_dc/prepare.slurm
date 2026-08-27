#!/bin/bash
#SBATCH --job-name=dc_prep
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_prepare_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_dc/slurm_prepare_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=short

# Stage 0 (§40): §37's prepare script run VERBATIM into this run's out_dir — the tube
# split and Stage-1 cell set are therefore provably identical to §37's.
# Verify the reused §36 tube shards, fix the main/reserve tube split, and build
# the equal-weight Stage-1 cell set. No tubes are re-materialised.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/prepare_oes90_pure.py --out_dir results/oes90_dc
