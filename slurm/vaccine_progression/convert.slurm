#!/bin/bash
#SBATCH --job-name=vac_convert
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_convert_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/vaccine_progression/slurm_convert_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=short

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/vaccine_progression
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
RAW=/cs/labs/mornitzan/yam.arieli/datasets/SARSCoV2_Vaccine/raw

# R may be a module on compute nodes; make Rscript available defensively.
if ! command -v Rscript >/dev/null 2>&1; then
    module load R 2>/dev/null || module load r 2>/dev/null || module load R/4.3 2>/dev/null || true
fi
command -v Rscript >/dev/null 2>&1 || { echo "ABORT: Rscript not found (need R + Matrix, ideally Seurat)"; exit 2; }

# 1. Seurat .rds -> flat files (counts.mtx / genes / barcodes / adt / meta)
Rscript scripts/convert_vaccine_rds_to_h5ad.R \
    --rds "$RAW/PBMC_vaccine_CITE.rds" \
    --out_dir "$RAW/flat"

# 2. flat files -> AnnData
$PY scripts/assemble_vaccine_h5ad.py \
    --flat_dir "$RAW/flat" \
    --out "$RAW/vaccine_cite_raw.h5ad"
