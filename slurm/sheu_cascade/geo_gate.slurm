#!/bin/bash
#SBATCH --job-name=sheu_geo
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --output=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascade/slurm_geo_%j.out
#SBATCH --error=/cs/labs/mornitzan/yam.arieli/cytokine-mil/results/sheu_cascade/slurm_geo_%j.err
#
# Path A latent geometry (per seed) + §21 axis gate for one time point.
# Best-effort: latent geometry runs per seed dir; gate is defensive.
# Env: TP, RES
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
export MPLCONFIGDIR=/cs/labs/mornitzan/yam.arieli/.matplotlib_cache
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

GEO=$RES/$TP/geometry
mkdir -p "$GEO"
echo "=== Sheu latent geometry  TP=$TP ==="
# Run geometry for each completed stage12 seed dir (continue on failure).
for SEEDDIR in $RES/$TP/stage12/seed_*; do
    RUN=$(ls -dt "$SEEDDIR"/run_* 2>/dev/null | head -1)
    [ -z "$RUN" ] && continue
    SEEDNAME=$(basename "$SEEDDIR")
    $PY scripts/run_latent_geometry.py \
        --run_dir "$RUN" \
        --output_dir "$GEO/$SEEDNAME" \
        --n_permutations 1000 || echo "geometry failed for $SEEDDIR (continuing)"
done

echo "=== Sheu §21 axis gate  TP=$TP ==="
$PY scripts/eval_sheu_axis_gate.py \
    --geo_dir "$GEO" \
    --out "$RES/$TP/sheu_axis_gate_${TP}.md" \
    --time "$TP" || echo "axis gate skipped (Path B is primary)"
echo "=== DONE geo_gate TP=$TP ==="
