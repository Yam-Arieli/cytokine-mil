#!/bin/bash
#SBATCH --job-name=sheu_pathway_traj
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_pathway_traj_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_pathway_traj_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Time-trajectory pathway analysis: run penetration on multiple Sheu
# pseudotube manifests (each a different time point), combine into a
# single trajectory dataset + plots.

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_pathway_trajectory.py \
    --trajectory \
        "1hr:1.0:/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_1hr_pseudotubes/manifest.json" \
        "3hr:3.0:/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json" \
    --out_dir /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_pathway_trajectory \
    --min_cells 10 \
    --seed 42
