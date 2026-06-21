#!/bin/bash
#SBATCH --job-name=rig_prune
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/recurrent_ig/slurm_prune_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/recurrent_ig/slurm_prune_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short

# §31 follow-up: does a stability-pruned S_X beat the raw top-50 on the §26
# direction benchmark? CPU-only; loads cells for the labeled cytokines (train donors).

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

$PY scripts/run_signature_pruning_test.py \
    --ig_dir results/recurrent_ig \
    --output_dir results/recurrent_ig/pruning_test
