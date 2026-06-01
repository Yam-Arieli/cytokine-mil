#!/bin/bash
# Gateway-safe entry point: verify the SCP<->GEO barcode join (fast, low memory,
# reads only barcode lists), and ONLY if it holds, submit the full DAG.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/id_cascade/verify_then_submit.sh'"
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
PY=/cs/labs/mornitzan/yam.arieli/venvs/biovenv/bin/python

echo "=== Step 1: barcode-join pre-check (gateway-safe) ==="
$PY scripts/check_id_barcode_join.py --channels 01,02,03,04,05 --min_frac 0.5

echo ""
echo "=== Step 2: join verified -> submitting DAG ==="
mkdir -p results/id_cascade
bash slurm/id_cascade/submit_id_cascade_dag.sh
