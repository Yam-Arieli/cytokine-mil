#!/bin/bash
#SBATCH --job-name=cg_build
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cano_gamez/slurm_build_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cano_gamez/slurm_build_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --partition=short

# Build Cano-Gamez CD4 T-cell pseudo-tubes from the mtx + metadata.
# 5 conditions (PBS[UNS]/Th0/Th2/Th17/iTreg), 4 donors, 2 cell types (Naive/Memory).
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
MAN=/cs/labs/mornitzan/yam.arieli/datasets/CanoGamez_pseudotubes/manifest.json

if [ -f "$MAN" ]; then
    echo "manifest exists at $MAN; skipping build"
else
    $PY scripts/build_pseudotubes_cano_gamez.py
fi
echo "=== DONE cg_build ==="
