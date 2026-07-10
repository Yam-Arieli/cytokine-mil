#!/bin/bash
#SBATCH --job-name=umap_donor1
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/umap_donor1/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/umap_donor1/slurm_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --partition=short

# Two-panel donor UMAP (CLAUDE.md figure): cells separate by CELL TYPE while
# CYTOKINES are intermixed. CPU-only. Oesinghaus Donor1, first 3 pseudo-tubes
# per cytokine. Compute the embedding, then render the figure from the parquet.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/umap_donor1

PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json

$PY scripts/umap_donor_embedding.py \
    --manifest_path "$MAN" \
    --donor Donor1 \
    --tubes_per_cytokine 3 \
    --n_pcs 50 \
    --out_dir results/umap_donor1

$PY scripts/plot_umap_donor.py \
    --coords_path results/umap_donor1/umap_coords.parquet \
    --out_dir results/umap_donor1 \
    --dpi 220

echo "DONE. figure at results/umap_donor1/umap.png (+ .pdf), coords at results/umap_donor1/umap_coords.parquet"
