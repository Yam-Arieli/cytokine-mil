#!/bin/bash
#SBATCH --job-name=bin_ig
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_bin_ig_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_bin_ig_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --partition=short

# Binary-MIL IG probe (Probe B, step 1 of the hybrid gene-set discovery).
# Loads pre-trained binary AB-MIL models from a previous oesinghaus_binary run
# and computes integrated gradients per cytokine over ALL HVGs.
#
# Output: results/gene_dynamics_phase0/binary_ig/{binary_ig.parquet,
#          binary_top_genes_summary.md, binary_marker_hits.csv}

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/gene_dynamics_phase0/binary_ig

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_binary_ig_probe.py \
    --binary_run_dir results/oesinghaus_binary/run_20260412_114413_pid3728447 \
    --output_dir     results/gene_dynamics_phase0/binary_ig \
    --top_n 50 \
    --n_ig_steps 20 \
    --max_tubes_per_cytokine 10 \
    --device cpu
