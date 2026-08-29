#!/bin/bash
#SBATCH --job-name=cp_0a
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/code_path/slurm_gate0a_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/code_path/slurm_gate0a_%j.err
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --partition=short

# Phase 0a — tube-identity gate (SPEC.md §2). Streams: one .h5ad at a time on the
# manifest side, one condition's shards at a time on the cascadir side, so the ask is set
# by the largest single condition (~10 donors x 10 tubes) and not by the ~64 GB full set.
# Exits nonzero on mismatch, which is what makes it a gate rather than a report.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python scripts/phase0_tube_identity.py \
    --out_dir results/code_path/phase0
