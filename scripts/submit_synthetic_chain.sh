#!/bin/bash
# Submit the full synthetic cascade v2 pipeline as a chained dependency.
# build (CPU) → train 6-seed array (GPU) → geo 6-seed array (CPU) → aggregate (CPU)
# Run from the cluster login node: bash cytokine-mil/scripts/submit_synthetic_chain.sh
set -e
cd /cs/labs/mornitzan/yam.arieli

BUILD_JID=$(sbatch --parsable cytokine-mil/scripts/run_synthetic_build.slurm)
echo "Build job:     ${BUILD_JID}"

TRAIN_JID=$(sbatch --parsable --dependency=afterok:${BUILD_JID} \
    cytokine-mil/scripts/run_synthetic_train.slurm)
echo "Train job:     ${TRAIN_JID}"

GEO_JID=$(sbatch --parsable --dependency=afterok:${TRAIN_JID} \
    cytokine-mil/scripts/run_synthetic_geo.slurm)
echo "Geo job:       ${GEO_JID}"

AGG_JID=$(sbatch --parsable \
    --dependency=afterok:${GEO_JID} \
    --job-name=synth_agg \
    --partition=short \
    --time=00:30:00 \
    --mem=8G \
    --output=cytokine-mil/logs/synth_agg_%j.out \
    --wrap="bash -c 'source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate && cd /cs/labs/mornitzan/yam.arieli/cytokine-mil && python scripts/aggregate_geo_synthetic.py'")
echo "Aggregate job: ${AGG_JID}"

echo ""
echo "Chain: build=${BUILD_JID} -> train=${TRAIN_JID} -> geo=${GEO_JID} -> agg=${AGG_JID}"
