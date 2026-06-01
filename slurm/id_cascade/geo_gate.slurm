#!/bin/bash
#SBATCH --job-name=id_geo
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_geo_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/id_cascade/slurm_geo_%j.err
#
# Path A latent geometry (per seed) -> cytokine coupling axes.
# Secondary check only — Path B cross_asym is the primary direction metric.
# Env: RES
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

RES=${RES:-results/id_cascade}
GEO=$RES/geometry
mkdir -p "$GEO"

echo "=== ID latent geometry (per seed) ==="
# Best-effort per seed; continue on failure (geometry is secondary).
for SEEDDIR in $RES/stage12/seed_*; do
    [ ! -d "$SEEDDIR" ] && continue
    RUN=$(ls -dt "$SEEDDIR"/run_* 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then
        # Sometimes the stage12 script writes directly under seed_* (no run_*
        # subdir) — try that as a fallback.
        if [ -f "$SEEDDIR/model_stage2.pt" ]; then RUN=$SEEDDIR; fi
    fi
    [ -z "$RUN" ] && { echo "skip $SEEDDIR — no run_* / model_stage2.pt"; continue; }
    SEEDNAME=$(basename "$SEEDDIR")
    echo "[$(date -Iseconds)] geometry for $SEEDNAME  RUN=$RUN"
    $PY scripts/run_latent_geometry.py \
        --run_dir "$RUN" \
        --output_dir "$GEO/$SEEDNAME" \
        --n_permutations 1000 || echo "geometry failed for $SEEDDIR (continuing)"
done
echo "=== DONE geometry ==="
