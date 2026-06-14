#!/bin/bash
#SBATCH --job-name=sheu_genestats
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_temporal_gene_stats/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_temporal_gene_stats/slurm_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Early-vs-late gene DISTRIBUTIONAL statistics (beyond the mean) on the Sheu time course.
# Loads the raw Sheu time course once (~295K cells, 500 genes), keeps RAW counts +
# log1p(normalize_total), and per stimulus {LPS, PIC} / per pseudo-donor / per timepoint
# computes variance, dispersion (Fano + mean-decoupled Fano-residual), bimodality (Sarle BC
# + expressing-cells-only), recruitment (frac_expressing + detection-residual), inequality
# (Gini), robust spread (IQR/QCD), and per-gene timing features. Runs H1-H6 at DONOR level
# (n_eff=4) with the mean-variance coupling trap controlled, and renders the figures.
# CPU only, no model training. See reports/sheu2024_temporal/EARLY_LATE_STATS_PREREGISTRATION.md.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/sheu_temporal_gene_stats

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/run_sheu_temporal_gene_stats.py \
    --stimuli LPS PIC \
    --min_cells 80 \
    --cap 1500 \
    --output_dir results/sheu_temporal_gene_stats
