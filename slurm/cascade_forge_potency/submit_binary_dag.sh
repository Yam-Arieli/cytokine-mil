#!/bin/bash
# Submit the per-label BINARY (label vs PBS) cascade_forge_potency re-run (unattended).
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/cascade_forge_potency/submit_binary_dag.sh'"
#
# Motivation: removes the multiclass cross-label-confusability confound (a 21-class
# softmax's plateau timing can depend on how similar a label is to OTHER labels, not just
# on its own cascade depth). Each label gets its own label-vs-PBS model, sharing one frozen
# Stage-1 encoder across all 20 labels per seed (matches the project's binary-IG
# convention). Same lr=0.001 / stage2_epochs=4000 as the low-LR multiclass re-run, for a
# direct, fair comparison against reports/cascade_forge_potency_lowlr/.
#
# DAG:
#   train_binary    (GPU array 0-2: seeds 42,123,7; 20 sequential binary models/seed;
#                    reuses the existing pseudo-tubes from results/cascade_forge_potency/)
#     -> validate_binary (CPU: accuracy-based + loss-based source_potency + trajectory plot,
#                          identical scripts/metrics as the multiclass runs -- same schema)
#
# Bottom line: reports/cascade_forge_potency_binary/{SOURCE_POTENCY_VALIDATION,LOSS_POTENCY_VALIDATION,label_trajectories}.{md,png}
#
# Dry-run: SUBMIT=echo bash slurm/cascade_forge_potency/submit_binary_dag.sh

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/cascade_forge_potency_binary reports/cascade_forge_potency_binary

SUBMIT=${SUBMIT:-sbatch}

TRAIN=$($SUBMIT --parsable slurm/cascade_forge_potency/train_binary.slurm)
VALIDATE=$($SUBMIT --parsable --dependency=afterok:$TRAIN slurm/cascade_forge_potency/validate_binary.slurm)

echo ""
echo "Submitted cascade_forge_potency binary (label-vs-PBS) DAG:"
echo "  train_binary    = $TRAIN     (GPU array 0-2: seeds 42,123,7; 20 binary models/seed)"
echo "  validate_binary = $VALIDATE  (CPU: accuracy+loss source_potency + trajectory plot)"
echo ""
echo "Bottom line: reports/cascade_forge_potency_binary/"
echo "Monitor: squeue -u yam.arieli"
