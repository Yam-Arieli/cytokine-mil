#!/bin/bash
#SBATCH --job-name=cov_prepare
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/covid_progression/slurm_prepare_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/covid_progression/slurm_prepare_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/covid_progression
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/prepare_covid_haniffa.py \
    --raw /cs/labs/mornitzan/yam.arieli/datasets/COVID_Haniffa/raw/haniffa21.processed.h5ad \
    --out /cs/labs/mornitzan/yam.arieli/datasets/COVID_Haniffa/prepared/covid_haniffa_prepared.h5ad
