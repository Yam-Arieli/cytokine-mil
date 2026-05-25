#!/bin/bash
#SBATCH --job-name=sheu_audit
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_audit_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_audit_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Adversarial audit of the pathway-signature cascade methodology (§23).
# Runs 4 independent validations: permutation null, random-pathway null,
# per-donor inference, directional asymmetry.

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_pathway_audit.py \
    --manifest /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json \
    --out_dir  /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_pathway_audit \
    --min_cells 10 \
    --n_permutations 2000 \
    --n_random_pathways 200 \
    --seed 42
