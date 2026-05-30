#!/bin/bash
#SBATCH --job-name=ab_pipe
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_ab_pipe_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_ab_pipe_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# End-to-end Path A -> Bridge -> Path B pipeline.
# Loads cytokine_axes.csv (Path A), binary_ig.parquet (Bridge), and
# Oesinghaus pseudotubes, then calls §24 directional_asymmetry_test for
# each axis where both endpoints have binary IG coverage.
#
# Expected outputs in results/gene_dynamics_phase0/pipeline_a_b/:
#   per_celltype.csv     incremental, one row per (axis, cell_type, direction)
#   per_axis_summary.csv incremental, one row per axis
#   verdict.md           ground-truth sign-accuracy summary

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/gene_dynamics_phase0/pipeline_a_b

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_pipeline_a_bridge_b.py \
    --axes_csv          reports/cascade_pairs/cytokine_axes.csv \
    --binary_ig_parquet results/gene_dynamics_phase0/binary_ig/binary_ig.parquet \
    --manifest_path     /cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json \
    --hvg_path          /cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json \
    --output_dir        results/gene_dynamics_phase0/pipeline_a_b \
    --top_n             50 \
    --min_cells         10
