#!/bin/bash
#SBATCH --job-name=latent_geo
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/latent_geometry/slurm_%A_%a.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/latent_geometry/slurm_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=short

RUN_DIRS=(
    "/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/run_20260412_161758_seed42"
    "/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/run_20260412_161803_seed123"
    "/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oesinghaus_full/run_20260412_161803_seed7"
)
RUN_DIR="${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"

PYTHON=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
REPO=/cs/labs/mornitzan/yam.arieli/cytokine-mil

export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache

mkdir -p /cs/labs/mornitzan/yam.arieli/cytokine-mil/results/latent_geometry

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Run dir:    $RUN_DIR"
echo "Started:    $(date)"

cd "$REPO"

$PYTHON scripts/run_latent_geometry.py \
    --run_dir "$RUN_DIR" \
    --output_dir "$RUN_DIR" \
    --n_permutations 1000 \
    --fdr_alpha 0.05 \
    --device cpu

echo "Finished: $(date)"
