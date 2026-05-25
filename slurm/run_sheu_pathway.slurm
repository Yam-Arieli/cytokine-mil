#!/bin/bash
#SBATCH --job-name=sheu_pathway
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_pathway_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_pathway_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Pathway-signature cascade analysis on Sheu phase-1 pseudotubes.
# Curated adaptor-specific gene sets; cascade penetration; pre-registered
# IFNAR binary test; magnitude test for LPS->TNF on NF-kB.

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_sheu_pathway_signatures.py \
    --manifest /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json \
    --out_dir  /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_pathway \
    --min_cells 10 \
    --n_ctrl 20 \
    --seed 42
