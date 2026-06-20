#!/bin/bash
#SBATCH --job-name=cg_download
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cano_gamez/slurm_download_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cano_gamez/slurm_download_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --partition=short

# Download + unzip + inspect the Cano-Gamez CD4 T-cell cytokine scRNA-seq
# (BioStudies S-BSST2978, ~8 GB). The .out captures the extracted file tree and
# text-file peeks so the adapter (build_pseudotubes_cano_gamez.py) can be designed
# from the actual format. CPU-only.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/cano_gamez
bash scripts/download_cano_gamez.sh
echo "=== DONE cg_download ==="
