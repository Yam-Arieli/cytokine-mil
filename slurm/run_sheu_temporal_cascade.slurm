#!/bin/bash
#SBATCH --job-name=sheu_tcasc
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_temporal_cascade/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_temporal_cascade/slurm_%j.err
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --partition=short

# Time-resolved gene cascades on the Sheu time course (PIC, LPS).
# Loads the raw Sheu time course once (~295K cells, 500 genes), builds per-gene
# PBS-corrected trajectories over 0.25->8hr, orders genes by activation time, infers
# directed precedence edges, and validates the known IFN cascade (V1/V2/V3).
# CPU only. See reports/sheu2024_temporal/TEMPORAL_CASCADE_PREREGISTRATION.md.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/sheu_temporal_cascade

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_sheu_temporal_cascade.py \
    --stimuli PIC LPS \
    --output_dir results/sheu_temporal_cascade
