#!/bin/bash
#SBATCH --job-name=op_coupR
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure/slurm_coupR_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure/slurm_coupR_%j.err
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --partition=short

# Track 3: re-run coupling over the 54 responders only, so the degree correction and the
# BH family are both computed on the uncontaminated matrix. ~60% of the conditions, so
# ~90G against the 110G the full 91-condition run needed. Direction is pairwise and is
# NOT recomputed — read it from direction_table.csv.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_oes90_pure_coupling_responders.py --out_dir results/oes90_pure
