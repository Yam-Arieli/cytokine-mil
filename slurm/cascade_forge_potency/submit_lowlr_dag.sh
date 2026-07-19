#!/bin/bash
# Submit the low-LR cascade_forge_potency re-run (unattended).
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/cascade_forge_potency/submit_lowlr_dag.sh'"
#
# DAG:
#   train_lowlr    (GPU array 0-2: seeds 42,123,7; lr=0.001, stage2_epochs=4000;
#                   reuses the existing pseudo-tubes from results/cascade_forge_potency/)
#     -> validate_lowlr (CPU: accuracy-based + loss-based source_potency + trajectory plot)
#
# Bottom line: reports/cascade_forge_potency_lowlr/{SOURCE_POTENCY_VALIDATION,LOSS_POTENCY_VALIDATION}.md
#
# Dry-run: SUBMIT=echo bash slurm/cascade_forge_potency/submit_lowlr_dag.sh

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/cascade_forge_potency_lowlr reports/cascade_forge_potency_lowlr

SUBMIT=${SUBMIT:-sbatch}

TRAIN=$($SUBMIT --parsable slurm/cascade_forge_potency/train_lowlr.slurm)
VALIDATE=$($SUBMIT --parsable --dependency=afterok:$TRAIN slurm/cascade_forge_potency/validate_lowlr.slurm)

echo ""
echo "Submitted cascade_forge_potency low-LR DAG:"
echo "  train_lowlr    = $TRAIN     (GPU array 0-2: seeds 42,123,7; lr=0.001, 4000 epochs)"
echo "  validate_lowlr = $VALIDATE  (CPU: accuracy+loss source_potency + trajectory plot)"
echo ""
echo "Bottom line: reports/cascade_forge_potency_lowlr/"
echo "Monitor: squeue -u yam.arieli"
