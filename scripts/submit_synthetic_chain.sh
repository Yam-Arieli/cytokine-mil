#!/bin/bash
# Submit the full synthetic cascade v2 pipeline as a chained dependency.
# build (CPU) → train 6-seed array (GPU, ~5 min) → geo 6-seed array (CPU) → aggregate (CPU)
#
# PBS-RC geometry is purely geometric — no MIL Stage 2 needed.
# Train step: Stage 1 encoder only (cell-type classification, 50 epochs).
#
# Run from the cluster login node:
#   bash cytokine-mil/scripts/submit_synthetic_chain.sh
#
# To skip build (data already exists at synthetic_cascades_v2/):
#   bash cytokine-mil/scripts/submit_synthetic_chain.sh --skip-build
set -e
cd /cs/labs/mornitzan/yam.arieli

SKIP_BUILD=0
[[ "${1}" == "--skip-build" ]] && SKIP_BUILD=1

if [[ ${SKIP_BUILD} -eq 0 ]]; then
    BUILD_JID=$(sbatch --parsable cytokine-mil/scripts/run_synthetic_build.slurm)
    echo "Build job:     ${BUILD_JID}"
    TRAIN_DEP="--dependency=afterok:${BUILD_JID}"
else
    echo "Build skipped (--skip-build)"
    TRAIN_DEP=""
fi

TRAIN_JID=$(sbatch --parsable ${TRAIN_DEP} \
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
echo "Chain: train=${TRAIN_JID} -> geo=${GEO_JID} -> agg=${AGG_JID}"
