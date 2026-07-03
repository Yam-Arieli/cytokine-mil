#!/bin/bash
#SBATCH --job-name=source_potency
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/source_potency/slurm_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/source_potency/slurm_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Source-potency ranking from the 3-seed multiclass dynamics (analysis-only, no GPU).
# dynamics.pkl are ~1.3 GB each (stored softmax trajectories) so this needs a batch
# node, not the gateway. Reads results/attention_dynamics/seed_*/dynamics.pkl.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
mkdir -p results/source_potency reports/source_potency
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY -u scripts/compute_source_potency.py \
    --base_dir results/attention_dynamics --seeds 42 123 7 \
    --out reports/source_potency/SOURCE_POTENCY_RESULTS.md
echo "SOURCE_POTENCY DONE"
