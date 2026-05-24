#!/bin/bash
#SBATCH --job-name=sheu_gate
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_gate_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_gate_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Sheu 2024 phase 1 axis-discovery gate.
# Loads 3 trained seeds; runs PBS-RC + per-donor Wilcoxon + cross-seed Spearman;
# writes reports/sheu2024/AXIS_GATE_VERDICT.md.

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_sheu_axis_gate.py \
    --seed_dirs results/sheu2024_full/seed_42 \
                results/sheu2024_full/seed_123 \
                results/sheu2024_full/seed_7 \
    --hvg_path /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json \
    --output_dir reports/sheu2024 \
    --alpha 0.05 \
    --min_seed_rho 0.7 \
    --direction_mode global \
    --relax_gate \
    --device cpu
