#!/bin/bash
#SBATCH --job-name=sheu_build
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_build_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_build_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Build Sheu 2024 pseudotubes on the cluster.
# Inputs:  /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw/
# Outputs: /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/
#          manifest.json, hvg_list.json, <pseudo_donor>/<cytokine>/pseudotube_N.h5ad

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/build_pseudotubes_sheu2024.py \
    --raw_dir /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024/raw \
    --base_path /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes \
    --seed 42
