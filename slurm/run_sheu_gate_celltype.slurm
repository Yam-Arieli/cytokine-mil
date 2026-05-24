#!/bin/bash
#SBATCH --job-name=sheu_gate_ct
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_gate_ct_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_sheu_gate_ct_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Track C: axis-discovery gate on existing 3hr narrowed checkpoints,
# with direction_mode=cell_type (T-specific direction vectors instead of global).

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p reports/sheu2024_celltype

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_sheu_axis_gate.py \
    --seed_dirs results/sheu2024_full/seed_42 \
                results/sheu2024_full/seed_123 \
                results/sheu2024_full/seed_7 \
    --hvg_path /cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json \
    --output_dir reports/sheu2024_celltype \
    --alpha 0.05 \
    --min_seed_rho 0.7 \
    --direction_mode cell_type \
    --relax_gate \
    --device cpu
