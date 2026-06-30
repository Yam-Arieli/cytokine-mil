#!/bin/bash
# Submit the §33 attention-collapse intervention DAG: 3 experiments (A/B/C),
# each 3 seeds, run IN PARALLEL, each with its own extract->analysis chain
# (afterok), then a final compare job (afterok all 3 analyses).
#
# Run ON the cluster (login shell so sbatch is in PATH):
#   cluster_cmd "bash -l -c 'cd cytokine-mil && bash slurm/attn_reg/submit_attn_reg_dag.sh'"
# Dry-run: SUBMIT=echo bash slurm/attn_reg/submit_attn_reg_dag.sh
#
# Per experiment:  train (GPU array 0-2) -> extract (CPU array 0-2) -> analysis (CPU)
# Final:           compare (CPU, afterok all three analyses)
set -e
cd /cs/labs/mornitzan/yam.arieli/cytokine-mil
mkdir -p results/attn_reg reports/attention_dynamics

SUBMIT=${SUBMIT:-sbatch}
ANALS=""

for EXP in A B C; do
    T=$($SUBMIT --parsable --export=ALL,EXP=$EXP slurm/attn_reg/train.slurm)
    E=$($SUBMIT --parsable --dependency=afterok:$T --export=ALL,EXP=$EXP slurm/attn_reg/extract.slurm)
    A=$($SUBMIT --parsable --dependency=afterok:$E --export=ALL,EXP=$EXP slurm/attn_reg/analysis.slurm)
    echo "EXP $EXP:  train=$T  extract=$E  analysis=$A"
    ANALS="$ANALS:$A"
done

CMP=$($SUBMIT --parsable --dependency=afterok${ANALS} slurm/attn_reg/compare.slurm)
echo "compare=$CMP  (afterok${ANALS})"
echo ""
echo "Bottom line will land at:"
echo "  reports/attention_dynamics/INTERVENTION_COMPARISON.md"
echo "  results/attn_reg/{A,B,C}/seed_*/{attention_trajectory.pkl,plots/,ATTENTION_DYNAMICS_RESULTS.md}"
echo "Monitor: squeue -u yam.arieli"
