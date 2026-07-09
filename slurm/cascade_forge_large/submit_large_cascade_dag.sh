#!/bin/bash
# Submit the large cascade_forge experiment DAG with SLURM dependencies (unattended).
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/cascade_forge_large/submit_large_cascade_dag.sh'"
#
# DAG:
#   forge     (CPU: 5 configs -> 7 h5ads ~1.05M cells each + h5ad_manifest.txt)
#     -> benchmark (GPU array 0-6: cascadir fit + direction + coupling per h5ad)
#       -> aggregate (CPU: RESULTS.md)
#
# Bottom line: results/cascade_forge_large/RESULTS.md
#
# Dry-run: SUBMIT=echo bash slurm/cascade_forge_large/submit_large_cascade_dag.sh

set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/cascade_forge_large

SUBMIT=${SUBMIT:-sbatch}

FORGE=$($SUBMIT --parsable slurm/cascade_forge_large/forge.slurm)
BENCH=$($SUBMIT --parsable --dependency=afterok:$FORGE slurm/cascade_forge_large/benchmark.slurm)
AGG=$($SUBMIT   --parsable --dependency=afterok:$BENCH slurm/cascade_forge_large/aggregate.slurm)

echo ""
echo "Submitted cascade_forge_large DAG:"
echo "  forge     = $FORGE  (CPU: 5 configs -> 7 h5ads + manifest)"
echo "  benchmark = $BENCH  (GPU array 0-6: cascadir fit + direction + coupling)"
echo "  aggregate = $AGG    (CPU: RESULTS.md)"
echo ""
echo "Bottom line: results/cascade_forge_large/RESULTS.md"
echo "Monitor: squeue -u yam.arieli"
