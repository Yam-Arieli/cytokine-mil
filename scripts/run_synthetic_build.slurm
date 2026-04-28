#!/bin/bash
#SBATCH --job-name=synth_build
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_build_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/logs/synth_build_%j.err
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# Build the synthetic cascade dataset on the cluster.
# Must complete before training jobs start.
# Output: /cs/labs/mornitzan/yam.arieli/datasets/synthetic_cascades_v1/

OUT_DIR=/cs/labs/mornitzan/yam.arieli/datasets/synthetic_cascades_v1

echo "========================================================"
echo "Building synthetic cascade dataset"
echo "  out_dir : ${OUT_DIR}"
echo "========================================================"

set -e

source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil

python scripts/build_synthetic_cascades.py \
    --out "${OUT_DIR}" \
    --seed 0 \
    --n_donors 6 \
    --n_pseudo_tubes 8 \
    --n_per_cell_type 30 \
    --n_cell_types 12 \
    --n_cytokines 20 \
    --n_genes 512

echo "Done — dataset written to ${OUT_DIR}"
