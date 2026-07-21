#!/bin/bash
#SBATCH --job-name=sheu_cd_prep
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascadir_native/slurm_prepare_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascadir_native/slurm_prepare_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/sheu_cascadir_native
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/prepare_sheu_cascadir.py \
    --manifest_path /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_5hr_pseudotubes/manifest.json \
    --time_filter 5hr \
    --out /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_5hr_pseudotubes/prepared/sheu_5hr_prepared.h5ad
