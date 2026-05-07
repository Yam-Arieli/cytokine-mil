#!/bin/bash
#SBATCH --job-name=lookup_benchmark
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j_lookup_benchmark.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/slurm_%j_lookup_benchmark.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=short

cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

mkdir -p results/oesinghaus_full/geo_refined_comparison

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/lookup_benchmark_pairs.py
