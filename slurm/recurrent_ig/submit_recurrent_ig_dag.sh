#!/bin/bash
# Submit the recurrent-IG DAG (CLAUDE.md §31) with SLURM dependencies so it runs
# unattended.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/recurrent_ig/submit_recurrent_ig_dag.sh'"
#
# Dry-run (print sbatch commands instead of submitting):
#   SUBMIT=echo bash slurm/recurrent_ig/submit_recurrent_ig_dag.sh
#
# DAG:
#   train (GPU array 0-2: seeds 42/123/7 -> ig_traj.parquet per seed)
#     -> analysis (CPU 128G: P-A..P-E + §26 regression check + figures + verdict)
#
# Bottom line: reports/recurrent_ig/RECURRENT_IG_RESULTS.md
#              results/recurrent_ig/{stats,plots}
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/recurrent_ig reports/recurrent_ig

SUBMIT=${SUBMIT:-sbatch}

TRAIN=$($SUBMIT --parsable slurm/recurrent_ig/train.slurm)
ANAL=$($SUBMIT  --parsable --dependency=afterok:$TRAIN slurm/recurrent_ig/analysis.slurm)

echo ""
echo "Submitted recurrent-IG DAG:"
echo "  train    = $TRAIN  (GPU array 0-2: seeds 42/123/7, recurrent IG every 10 epochs)"
echo "  analysis = $ANAL   (CPU 128G: stats + figures + verdict, afterok train)"
echo ""
echo "Bottom line will land at:"
echo "  reports/recurrent_ig/RECURRENT_IG_RESULTS.md"
echo "  results/recurrent_ig/stats/*.csv  +  results/recurrent_ig/plots/*.png"
echo "Monitor: squeue -u yam.arieli"
