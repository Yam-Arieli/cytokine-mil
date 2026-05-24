#!/bin/bash
#SBATCH --job-name=sheu_aux_gate
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_aux_gate_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_aux_gate_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Track A: axis-discovery gate on adapter-aux checkpoints.

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p reports/sheu2024_aux

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_sheu_axis_gate.py \
    --seed_dirs results/sheu2024_aux/seed_42 \
                results/sheu2024_aux/seed_123 \
                results/sheu2024_aux/seed_7 \
    --hvg_path /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json \
    --output_dir reports/sheu2024_aux \
    --alpha 0.05 \
    --min_seed_rho 0.7 \
    --direction_mode global \
    --relax_gate \
    --device cpu
