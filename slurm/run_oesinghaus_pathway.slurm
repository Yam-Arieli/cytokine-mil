#!/bin/bash
#SBATCH --job-name=oes_pathway
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_oesinghaus_pathway_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_oesinghaus_pathway_%j.err
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Pathway-signature cascade analysis on Oesinghaus 24h PBMC.
# Subsamples to train donors × 1 tube per (donor, cytokine), uses HUMAN
# pathway library. Pre-registered: IFN cytokines vs all others on
# IFNAR_induced penetration.

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_oesinghaus_pathway_signatures.py \
    --manifest /cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json \
    --out_dir  /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_pathway \
    --val_donors Donor2 Donor3 \
    --tubes_per_pair 1 \
    --min_cells 10 \
    --n_ctrl 20 \
    --seed 42
