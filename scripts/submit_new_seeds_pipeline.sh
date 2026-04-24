#!/bin/bash
# Submit the full 3-stage new-seeds PBS-RC pipeline with SLURM dependencies.
#
# Stage 1: Train 9 new MIL models (array, ~24h GPU each, parallel)
# Stage 2: Run PBS-RC on each new model (array, ~5h CPU, starts after ALL Stage 1 done)
# Stage 3: Aggregate all 12 seeds + report (1 job, starts after ALL Stage 2 done)
#
# Usage (run from the cluster repo root):
#   bash scripts/submit_new_seeds_pipeline.sh

set -e

REPO=/cs/labs/mornitzan/yam.arieli/cytokine-mil

echo "========================================================"
echo "Submitting new-seeds PBS-RC pipeline"
echo "========================================================"

# Ensure log directory exists
mkdir -p ${REPO}/logs

# ---- Stage 1: MIL training (9 tasks) ------------------------------------
JOB1=$(sbatch --parsable ${REPO}/scripts/run_new_seeds_train.slurm)
echo "Stage 1 (training)  : job ${JOB1}  (9 tasks, seeds 1-6,8-10, ~28h)"

# ---- Stage 2: PBS-RC geometry (9 tasks, after ALL of Stage 1) -----------
JOB2=$(sbatch --parsable \
    --dependency=afterok:${JOB1} \
    ${REPO}/scripts/run_new_seeds_geo.slurm)
echo "Stage 2 (PBS-RC geo): job ${JOB2}  (9 tasks, ~6h, starts after ALL Stage 1 tasks)"

# ---- Stage 3: Aggregation (1 task, after ALL of Stage 2) ----------------
JOB3=$(sbatch --parsable \
    --dependency=afterok:${JOB2} \
    ${REPO}/scripts/run_aggregate_geo.slurm)
echo "Stage 3 (aggregate) : job ${JOB3}  (1 task,  ~2h, starts after ALL Stage 2 tasks)"

echo ""
echo "========================================================"
echo "Pipeline submitted."
echo "  Total expected wall time: ~34h (28h train + 6h geo + 2h agg)"
echo "  Monitor with: squeue -u yam.arieli"
echo "  Results will be in:"
echo "    ${REPO}/results/oesinghaus_full/new_seeds_seed{1..10}/"
echo "    ${REPO}/results/oesinghaus_full/geo_ensemble_summary/"
echo "========================================================"
