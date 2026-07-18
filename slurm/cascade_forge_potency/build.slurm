#!/bin/bash
#SBATCH --job-name=cfp_build
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency/slurm_build_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/cascade_forge_potency/slurm_build_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Forge the cascade_forge dataset (LARGE_CASCADES ground truth, effect_size=0.30, t=6,
# 10 donors x 21 conditions x 2000 cells/tube) and write pseudo-tubes + manifest.json in
# the cytokine_mil convention.

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
OUT=results/cascade_forge_potency/pseudotubes
mkdir -p results/cascade_forge_potency

$PY -u scripts/build_pseudotubes_cascade_forge.py \
    --out_dir "$OUT" \
    --n_cell_types 8 --n_cells_per_tube 2000 --n_donors 10 \
    --effect_size 0.30 --responder_mode all --snapshot_time 6 --seed 0

echo "BUILD DONE"
