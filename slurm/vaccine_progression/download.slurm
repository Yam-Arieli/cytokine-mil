#!/bin/bash
#SBATCH --job-name=vac_download
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_download_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_download_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/vaccine_progression

bash scripts/download_vaccine_multimodal.sh
