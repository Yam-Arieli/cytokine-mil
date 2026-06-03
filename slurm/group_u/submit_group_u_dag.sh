#!/bin/bash
# Submit the full Group-U DAG (§27) with SLURM dependencies so it runs unattended.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/group_u/submit_group_u_dag.sh'"
#
# DAG:
#   train (GPU array 0-7: missing cytokines, chunked)
#     -> ig_merge (CPU: IG on new models + merge into binary_ig_all45)
#       -> pipeline (CPU: Path A->B over all 121 axes, pooled + train_only, dir null n=1000)
#         -> fdr (CPU: BH-FDR + pi0 + P1-P4 verdict)
#
# Bottom line: reports/cascade_pairs/GROUP_U_RESULTS.md
#              results/group_u/pipeline_full121_train_only/per_axis_summary.csv
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/group_u

# Dry-run: SUBMIT=echo prints the sbatch commands instead of submitting.
SUBMIT=${SUBMIT:-sbatch}

# Guard: the existing 24-cytokine parquet must exist (we merge onto it).
ALL24=results/gene_dynamics_phase0/binary_ig_all24/binary_ig.parquet
if [ ! -f "$ALL24" ]; then
    echo "WARN: $ALL24 not found — training will treat all axis cytokines as missing."
fi

TRAIN=$($SUBMIT --parsable slurm/group_u/train.slurm)
IGM=$($SUBMIT   --parsable --dependency=afterok:$TRAIN slurm/group_u/ig_merge.slurm)
PIPE=$($SUBMIT  --parsable --dependency=afterok:$IGM   slurm/group_u/pipeline.slurm)
FDR=$($SUBMIT   --parsable --dependency=afterok:$PIPE  slurm/group_u/fdr.slurm)

echo ""
echo "Submitted Group-U DAG:"
echo "  train    = $TRAIN  (GPU array 0-7: missing cytokines)"
echo "  ig_merge = $IGM    (IG on new models + merge -> binary_ig_all45)"
echo "  pipeline = $PIPE   (Path A->B over 121 axes, dir null n=1000, pooled+train_only)"
echo "  fdr      = $FDR    (BH-FDR + pi0 + P1-P4 verdict)"
echo ""
echo "Bottom line will land at:"
echo "  reports/cascade_pairs/GROUP_U_RESULTS.md  (primary, train_only)"
echo "  reports/cascade_pairs/GROUP_U_RESULTS_pooled.md  (§26 anchor)"
echo "  results/group_u/pipeline_full121_train_only/per_axis_summary.csv"
echo "Monitor: squeue -u yam.arieli"
