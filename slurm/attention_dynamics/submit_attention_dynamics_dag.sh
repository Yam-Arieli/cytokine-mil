#!/bin/bash
# Submit the §33 attention training-dynamics DAG with SLURM dependencies.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/attention_dynamics/submit_attention_dynamics_dag.sh'"
#
# Dry-run (print sbatch commands instead of submitting):
#   SUBMIT=echo bash slurm/attention_dynamics/submit_attention_dynamics_dag.sh
#
# DAG:
#   train   (GPU array 0-2: seeds 42/123/7, MULTICLASS Stage-2, checkpoints every 10)
#     -> extract (CPU array 0-2: attention_trajectory.pkl per seed, afterok train)
#     -> analysis (CPU: P1..P4 + figures + verdict per seed, afterok extract)
#
# Bottom line: reports/attention_dynamics/ATTENTION_DYNAMICS_RESULTS.md
#              results/attention_dynamics/seed_*/{attention_trajectory.pkl,plots}
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/attention_dynamics reports/attention_dynamics

SUBMIT=${SUBMIT:-sbatch}

TRAIN=$($SUBMIT  --parsable slurm/attention_dynamics/train.slurm)
EXTRACT=$($SUBMIT --parsable --dependency=afterok:$TRAIN slurm/attention_dynamics/extract.slurm)
ANAL=$($SUBMIT   --parsable --dependency=afterok:$EXTRACT slurm/attention_dynamics/analysis.slurm)

echo ""
echo "Submitted attention-dynamics DAG:"
echo "  train    = $TRAIN    (GPU array 0-2: seeds 42/123/7, checkpoints every 10)"
echo "  extract  = $EXTRACT  (CPU array 0-2: attention_trajectory.pkl, afterok train)"
echo "  analysis = $ANAL     (CPU: P1..P4 + figures + verdict, afterok extract)"
echo ""
echo "Bottom line will land at:"
echo "  reports/attention_dynamics/ATTENTION_DYNAMICS_RESULTS.md"
echo "  results/attention_dynamics/seed_*/{attention_trajectory.pkl,plots/*.png}"
echo "Monitor: squeue -u yam.arieli"
