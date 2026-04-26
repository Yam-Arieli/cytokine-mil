#!/bin/bash
#SBATCH --job-name=quartet_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

WORKDIR=/cs/labs/mornitzan/yam.arieli
PYTHON=${WORKDIR}/venvs/biovenv/bin/python
REPO=${WORKDIR}/cytokine-mil

export MPLCONFIGDIR=${WORKDIR}/.matplotlib_cache

cd ${REPO}
${PYTHON} scripts/analyze_quartet.py \
    --seeds_dir ${REPO}/results/oesinghaus_full \
    --seeds 11 12 13 14 15 \
    --seed_prefix stage3_seed
