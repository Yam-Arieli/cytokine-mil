#!/bin/bash
#SBATCH --job-name=cfp_validate
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency/slurm_validate_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency/slurm_validate_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --partition=short

# Compute source_potency (3-seed averaged) and validate against the EXACT authored
# cascade_forge graph (out_degree, cascade_size, source-vs-leaf/isolated).

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
mkdir -p reports/cascade_forge_potency

$PY -u scripts/validate_source_potency_cascade_forge.py \
    --seeds_dir results/cascade_forge_potency --seeds 42 123 7 \
    --out reports/cascade_forge_potency/SOURCE_POTENCY_VALIDATION.md

echo "VALIDATE DONE"
