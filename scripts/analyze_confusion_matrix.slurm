#!/bin/bash
#SBATCH --job-name=conf_asym
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --partition=short

BASE=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/centroid_trajectory
OUT=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/directionality

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p ${OUT}

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    /cs/labs/mornitzan/yam.arieli/cytokine-mil/scripts/analyze_confusion_matrix.py \
    --seed_dirs \
        ${BASE}/seed_42 \
        ${BASE}/seed_123 \
        ${BASE}/seed_7 \
        ${BASE}/seed_17 \
        ${BASE}/seed_31 \
        ${BASE}/seed_53 \
        ${BASE}/seed_97 \
        ${BASE}/seed_137 \
        ${BASE}/seed_191 \
        ${BASE}/seed_257 \
    --output_dir ${OUT}/confusion_asymmetry
