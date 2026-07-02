#!/bin/bash
# Submit the §34 self-attention DAG with SLURM dependencies.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/selfattn/submit_selfattn_dag.sh'"
# Dry-run: SUBMIT=echo bash slurm/selfattn/submit_selfattn_dag.sh
#
# DAG:
#   train    (GPU array 0-2: seeds 42/123/7, MULTICLASS Stage-2, --model_type set_transformer,
#             every-epoch checkpoints)
#     -> extract  (CPU array 0-2: attention_trajectory.pkl + interaction_trajectory.pkl, afterok train)
#     -> analysis (CPU: §33 P1-P4 pooling + §34 interaction + 88%-comparable direction table, afterok extract)
#
# Bottom line:
#   reports/selfattn_dynamics/SELFATTN_RESULTS.md      (direction accuracy vs IG 88%)
#   reports/selfattn_dynamics/{POOLING_P1P4,INTERACTION}_RESULTS.md
#   results/selfattn/seed_*/{attention_trajectory.pkl,interaction_trajectory.pkl,plots/}
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/selfattn reports/selfattn_dynamics

SUBMIT=${SUBMIT:-sbatch}

TRAIN=$($SUBMIT   --parsable slurm/selfattn/train.slurm)
EXTRACT=$($SUBMIT --parsable --dependency=afterok:$TRAIN slurm/selfattn/extract.slurm)
ANAL=$($SUBMIT    --parsable --dependency=afterok:$EXTRACT slurm/selfattn/analysis.slurm)

echo ""
echo "Submitted self-attention DAG:"
echo "  train    = $TRAIN    (GPU array 0-2: seeds 42/123/7, --model_type set_transformer)"
echo "  extract  = $EXTRACT  (CPU array 0-2: both trajectories, afterok train)"
echo "  analysis = $ANAL     (CPU: P1-P4 + interaction + direction table, afterok extract)"
echo ""
echo "Bottom line will land at:"
echo "  reports/selfattn_dynamics/SELFATTN_RESULTS.md   (direction accuracy vs IG 88%)"
echo "  reports/selfattn_dynamics/{POOLING_P1P4,INTERACTION}_RESULTS.md"
echo "  results/selfattn/seed_*/{attention_trajectory,interaction_trajectory}.pkl + plots/"
echo "Monitor: squeue -u yam.arieli"
