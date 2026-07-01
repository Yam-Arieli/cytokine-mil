#!/bin/bash
# Submit the §33 attention-entropy-penalty LAMBDA SWEEP DAG: 3 lambdas (1/10/100),
# each 3 seeds, run IN PARALLEL, each with its own extract->analysis chain
# (afterok), then a final compare job (afterok all 3 analyses). Exp-A recipe only
# (frozen encoder, 250 epochs, penalty) — escalates the lambda=0.1 that was too weak.
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/attn_lambda/submit_attn_lambda_dag.sh'"
# Dry-run: SUBMIT=echo bash slurm/attn_lambda/submit_attn_lambda_dag.sh
#
# Per lambda:  train (GPU array 0-2) -> extract (CPU array 0-2) -> analysis (CPU)
# Final:       compare (CPU, afterok all three analyses)
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/attn_lambda reports/attention_dynamics

SUBMIT=${SUBMIT:-sbatch}
ANALS=""

for LAM in 1 10 100; do
    T=$($SUBMIT --parsable --export=ALL,LAM=$LAM slurm/attn_lambda/train.slurm)
    E=$($SUBMIT --parsable --dependency=afterok:$T --export=ALL,LAM=$LAM slurm/attn_lambda/extract.slurm)
    A=$($SUBMIT --parsable --dependency=afterok:$E --export=ALL,LAM=$LAM slurm/attn_lambda/analysis.slurm)
    echo "LAM $LAM:  train=$T  extract=$E  analysis=$A"
    ANALS="$ANALS:$A"
done

CMP=$($SUBMIT --parsable --dependency=afterok${ANALS} slurm/attn_lambda/compare.slurm)
echo "compare=$CMP  (afterok${ANALS})"
echo ""
echo "Bottom line will land at:"
echo "  reports/attention_dynamics/LAMBDA_SWEEP_COMPARISON.md"
echo "  results/attn_lambda/L{1,10,100}/seed_*/{attention_trajectory.pkl,plots/,ATTENTION_DYNAMICS_RESULTS.md}"
echo "Monitor: squeue -u yam.arieli"
