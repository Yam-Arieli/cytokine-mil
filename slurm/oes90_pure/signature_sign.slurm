#!/bin/bash
#SBATCH --job-name=op_sgn
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure/slurm_sigsign_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oes90_pure/slurm_sigsign_%j.err
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --partition=short

# Track 1 diagnosis: are the top-100 IG genes UP or DOWN vs PBS, per condition?
# Holds one k=4 tube-set copy (~25 GB) plus the regrouped per-(condition, cell type)
# cell arrays, so ~2 copies at peak.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/diagnose_oes90_pure_signature_sign.py --out_dir results/oes90_pure
