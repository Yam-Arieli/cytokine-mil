#!/bin/bash
#SBATCH --job-name=vac_prepare
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_prepare_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_prepare_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/vaccine_progression
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/prepare_vaccine_tcell.py \
    --raw /cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/raw/vaccine_cite_raw.h5ad \
    --out /cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/prepared/vaccine_tcell_prepared.h5ad \
    --state_source auto
