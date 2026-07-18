#!/bin/bash
# Submit the cascade_forge source_potency validation DAG (unattended).
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/cascade_forge_potency/submit_cascade_forge_potency_dag.sh'"
#
# DAG:
#   build    (CPU: forge LARGE_CASCADES + write pseudo-tubes/manifest)
#     -> train    (GPU array 0-2: Stage1+Stage2, one seed each: 42,123,7)
#       -> validate (CPU: source_potency vs the EXACT cascade_forge ground truth)
#
# Bottom line: reports/cascade_forge_potency/SOURCE_POTENCY_VALIDATION.md
#
# Dry-run: SUBMIT=echo bash slurm/cascade_forge_potency/submit_cascade_forge_potency_dag.sh

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/cascade_forge_potency reports/cascade_forge_potency

SUBMIT=${SUBMIT:-sbatch}

BUILD=$($SUBMIT --parsable slurm/cascade_forge_potency/build.slurm)
TRAIN=$($SUBMIT --parsable --dependency=afterok:$BUILD slurm/cascade_forge_potency/train.slurm)
VALIDATE=$($SUBMIT --parsable --dependency=afterok:$TRAIN slurm/cascade_forge_potency/validate.slurm)

echo ""
echo "Submitted cascade_forge_potency DAG:"
echo "  build    = $BUILD     (CPU: forge + pseudo-tubes)"
echo "  train    = $TRAIN     (GPU array 0-2: seeds 42,123,7)"
echo "  validate = $VALIDATE  (CPU: source_potency vs exact ground truth)"
echo ""
echo "Bottom line: reports/cascade_forge_potency/SOURCE_POTENCY_VALIDATION.md"
echo "Monitor: squeue -u yam.arieli"
