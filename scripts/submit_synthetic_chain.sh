#!/bin/bash
# Submit the full synthetic cascade pipeline as a chained dependency job.
# Run from the cluster login node: bash cytokine-mil/scripts/submit_synthetic_chain.sh
cd /cs/labs/mornitzan/yam.arieli

TRAIN_JID=$(sbatch --parsable cytokine-mil/scripts/run_synthetic_train.slurm)
echo "Train job:     ${TRAIN_JID}"

GEO_JID=$(sbatch --parsable --dependency=afterok:${TRAIN_JID} cytokine-mil/scripts/run_synthetic_geo.slurm)
echo "Geo job:       ${GEO_JID}"

AGG_JID=$(sbatch --parsable \
    --dependency=afterok:${GEO_JID} \
    --job-name=synth_agg \
    --partition=short \
    --time=00:30:00 \
    --mem=8G \
    --output=cytokine-mil/logs/synth_agg_%j.out \
    --wrap="source /cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/activate && cd /cs/labs/mornitzan/yam.arieli/cytokine-mil && python scripts/aggregate_geo_synthetic.py")
echo "Aggregate job: ${AGG_JID}"

echo ""
echo "Chain submitted: train=${TRAIN_JID} -> geo=${GEO_JID} -> agg=${AGG_JID}"
