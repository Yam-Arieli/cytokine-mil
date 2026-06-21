#!/bin/bash
#SBATCH --job-name=rig_analysis
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/recurrent_ig/slurm_analysis_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/recurrent_ig/slurm_analysis_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=short

# Recurrent-IG (CLAUDE.md §31, Part C): depends on the train array. Loads the 3
# per-seed ig_traj.parquet + Oesinghaus cells (train donors) + audited labels,
# computes P-A..P-E + the §26 regression check, renders figures, and writes the
# verdict report. CPU-only, memory-heavy (loads all 45 cytokines' cells).

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/recurrent_ig reports/recurrent_ig

PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/analyze_recurrent_ig.py \
    --ig_dir results/recurrent_ig \
    --output_dir results/recurrent_ig \
    --report reports/recurrent_ig/RECURRENT_IG_RESULTS.md
