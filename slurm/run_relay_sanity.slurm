#!/bin/bash
#SBATCH --job-name=relay_sanity
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/relay_cascade_id/sanity_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/relay_cascade_id/sanity_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short

# Sanity-check the RED verdict: sweep alpha x cell-type-set x panel-size x seed on the
# NK -> Macrophage relay. Loads raw ID tubes once, sweeps configs in-process.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/relay_cascade_id

/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python \
    scripts/relay_cascade_id.py \
    --sanity \
    --alphas 1,10,100 \
    --sanity_seeds 42,123 \
    --output_dir results/relay_cascade_id/sanity
