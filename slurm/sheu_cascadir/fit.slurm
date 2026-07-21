#!/bin/bash
#SBATCH --job-name=sheu_cd_fit
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascadir_native/slurm_fit_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascadir_native/slurm_fit_%j.err
#SBATCH --gres=gpu:a5000
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/sheu_cascadir_native/5hr
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/run_sheu_cascadir.py \
    --prepared /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_5hr_pseudotubes/prepared/sheu_5hr_prepared.h5ad \
    --output_dir results/sheu_cascadir_native/5hr \
    --device cuda --assume lognorm --seed 42
