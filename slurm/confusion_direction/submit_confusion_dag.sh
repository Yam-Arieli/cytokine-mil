#!/bin/bash
# Submit the confusion-direction DAG: build signature union U -> restricted multiclass
# training (GPU, 3 seeds) -> confusion-direction analysis (CPU, afterok train).
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/confusion_direction/submit_confusion_dag.sh'"
# Dry-run: SUBMIT=echo bash slurm/confusion_direction/submit_confusion_dag.sh
#
# The full-gene CONTROL (slurm/confusion_direction/control.slurm) is a separate,
# training-free baseline already run on results/attention_dynamics.
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/confusion_direction reports/confusion_direction datasets/signature_union
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python
SUBMIT=${SUBMIT:-sbatch}

# 45-cytokine Oesinghaus binary-IG signatures (broadest coverage for U).
BINARY_IG=results/group_u/binary_ig_all45/binary_ig.parquet
UGENES=datasets/signature_union/gene_list_signature_union.json

# Step 1 (inline, fast): build U = union of top-50 signatures ∩ HVG.
$PY scripts/build_signature_union.py --binary_ig $BINARY_IG --top_n 50 --out $UGENES

# Step 2/3: restricted training (GPU array) -> analysis (afterok).
TRAIN=$($SUBMIT --parsable slurm/confusion_direction/train.slurm)
ANAL=$($SUBMIT  --parsable --dependency=afterok:$TRAIN slurm/confusion_direction/analysis.slurm)

echo ""
echo "Submitted confusion-direction DAG:"
echo "  U genes  = $UGENES"
echo "  train    = $TRAIN   (GPU array 0-2: seeds 42/123/7, --hvg_path U)"
echo "  analysis = $ANAL    (CPU: confusion-direction benchmark + temporal, afterok train)"
echo ""
echo "Bottom line: reports/confusion_direction/CONFUSION_DIRECTION_RESULTS.md"
echo "  (compare to CONTROL_full_gene.md 8/17=47% and cross_asym 88%)"
echo "Monitor: squeue -u yam.arieli"
