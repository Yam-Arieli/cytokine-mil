#!/bin/bash
#SBATCH --job-name=umap_signature
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/umap_signature/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/umap_signature/slurm_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --partition=short

# Signature-restricted UMAP (thesis Results, non-circular signature check):
# 3 panels (all_hvg / signature S_X / random size-matched genes), same cells
# (one cytokine vs PBS), coloured by condition. CPU-only.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/umap_signature

PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json
HVG=/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json

$PY scripts/umap_signature_embedding.py \
    --manifest_path "$MAN" \
    --hvg_path "$HVG" \
    --binary_ig_parquet results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet \
    --cytokine IFN-gamma \
    --top_n 50 \
    --max_tubes_per_condition 40 \
    --n_pcs 50 \
    --out_dir results/umap_signature

$PY scripts/plot_umap_signature.py \
    --coords_path results/umap_signature/umap_coords.parquet \
    --stats_path results/umap_signature/stats.json \
    --out_dir results/umap_signature \
    --dpi 220

echo "DONE. figure at results/umap_signature/umap_signature.png (+ .pdf)"
