#!/bin/bash
#SBATCH --job-name=oelen_build
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen/slurm_build_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/oelen/slurm_build_%j.err
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=short

# Build Oelen 1M-scBloodNL pseudo-tubes (~928k cells, 2 chemistries, 121 donors).
# High mem for the two large mtx loads + concat + HVG on the full gene set.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
MAN=/cs/labs/mornitzan/yam.arieli/datasets/Oelen2022_pseudotubes/manifest.json

if [ -f "$MAN" ]; then
    echo "manifest exists at $MAN; skipping build"
else
    $PY scripts/build_pseudotubes_oelen2022.py
fi
echo "=== DONE oelen_build ==="
