#!/bin/bash
#SBATCH --job-name=oelen_dl
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen/slurm_download_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen/slurm_download_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --partition=short

# Download + inspect the Oelen 2022 1M-scBloodNL human PBMC dataset (free; molgenis
# portal, cert expired -> --no-check-certificate). 120-donor PBMC. The .out captures
# the metadata structure (condition x time, donors, cell types). CPU-only.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/oelen
bash scripts/download_oelen2022.sh
echo "=== DONE oelen_dl ==="
