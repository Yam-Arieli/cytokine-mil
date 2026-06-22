#!/bin/bash
#SBATCH --job-name=vac_apparatus
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_apparatus_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_apparatus_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/vaccine_progression/apparatus
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

# Independent synthetic GO/NO-GO gate (reused from §30, dataset-agnostic):
# (A) distinct-program ladder MUST recover order; (B) monotone-intensity ladder
# characterizes whether cross_asym is magnitude-fooled (calibrates interpretation).
$PY scripts/apparatus_cross_asym_ladder.py \
    --output_dir results/vaccine_progression/apparatus
